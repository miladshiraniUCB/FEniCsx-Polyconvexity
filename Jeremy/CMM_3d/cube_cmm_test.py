from mpi4py import MPI
import numpy as np
from dolfinx import mesh, fem, io
import ufl
from dolfinx.fem.petsc import NonlinearProblem, NewtonSolverNonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.plot import vtk_mesh
import pyvista as pv

from time import time

# -------------------------------------------------------------------
# 0. Basic setup
# -------------------------------------------------------------------

comm = MPI.COMM_WORLD
rank = comm.rank

elem_order = 2 # 1: linear, 2: quadratic

# Load mesh from XDMF file
mesh_file = "tube_mesh.xdmf"  # Replace with your mesh file path
with io.XDMFFile(comm, mesh_file, "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")


# Simple test mesh (replace with artery mesh)
# domain = mesh.create_box(comm, [
#     [0.0, 0.0, 0.0],
#     [10.0, 10.0, 10.0]],
#     [8, 8, 8], cell_type=mesh.CellType.hexahedron) # 10x10x10 [mm] cube

# domain = mesh.create_box(comm, [
#     [0.0, 0.0, 0.0],
#     [1.0, 1.0, 1.0]],
#     [8, 8, 8], cell_type=mesh.CellType.hexahedron) # 1x1x1 [mm] cube

# domain = mesh.create_box(comm, [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]], [8, 8, 8], cell_type=mesh.CellType.tetrahedron) # 10x10x10 [mm] tet cube
dim = domain.topology.dim


# V = fem.functionspace(domain, ("Lagrange", 1, (dim,))) # Linear vector space
V = fem.functionspace(domain, ("Lagrange", elem_order, (dim,))) # Quadratic vector space
u = fem.Function(V, name="u")
v = ufl.TestFunction(V)
du = ufl.TrialFunction(V)

I = ufl.Identity(dim)


# # Plotting with pyvista
# topology, cell_types, x = vtk_mesh(V)
# grid = pv.UnstructuredGrid(topology, cell_types, x)

# plotter = pv.Plotter()
# plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
# plotter.show()


# -------------------------------------------------------------------
# 1. Material parameters (from Table 1 in paper, scaled to your units)
#    You will probably read these from JSON or a config in practice.
# -------------------------------------------------------------------

# Pressure on "top" face
# Replace with pressure consistent with your geometry / Po in paper
# P0 = 14e3  # [Pa] (roughly the ~13.98 kPa in the paper)
P0 = 14

# Mass fractions at s=0
phi_e0 = 0.34 # Elastin subfraction
phi_m0 = 0.33 # Smooth muscle subfraction
phi_c0 = 0.33 # Collagen subfraction

# Collagen subfractions
beta_theta = 0.056 # circumferential
beta_z = 0.067 # axial
beta_d = 0.877  # diagonals

# Diagonal fiber angle (wrt axial direction)
alpha0_deg = 29.91 # degrees
alpha0 = alpha0_deg * np.pi / 180.0 # convert to radians

# Elastin parameter
c_e = 89.71e3  # [Pa]

# Smooth muscle parameters
c_m1 = 261.4e3  # [Pa]
c_m2 = 0.24

# Collagen parameters
c_c1 = 234.9e3  # [Pa]
c_c2 = 4.08

# Deposition stretches
G_e_theta = 1.9 # Circumferential deposition stretch
G_e_z = 1.62 # Axial deposition stretch
G_e_r = 1.0 / (G_e_theta * G_e_z)
G_e = np.diag([G_e_r, G_e_theta, G_e_z])

G_m = 1.20 # Smooth muscle deposition stretch (isotropic)
G_c = 1.25 # Collagen deposition stretch (isotropic)

# Penalty bulk modulus for Stage I
# kappa = 1e3 * c_e # Might be too stiff
# kappa = 1e2 * c_e
kappa = 1e1 * c_e # Solves!
# kappa = c_e

# -------------------------------------------------------------------
# 2. Helpers: kinematics, invariants, mass fractions
# -------------------------------------------------------------------

def kinematics(u):
    F = I + ufl.grad(u)
    C = F.T * F
    J = ufl.det(F)
    return F, C, J

def fiber_invariant(C, a0):
    """
    I4 = a0 · C a0 for a unit vector a0 (3D numpy array).
    """
    a = ufl.as_vector(a0)
    return ufl.dot(a, C * a)

# -------------------------------------------------------------------
# 3. Four-fiber strain energy pieces (per reference volume)
#    NOTE: We implement them as hyperelastic parts suitable for UFL AD.
# -------------------------------------------------------------------

def W_elastin(C):
    # C_e = G_e C G_e
    G = ufl.as_matrix(G_e)
    C_e = G * C * G
    I1_e = ufl.tr(C_e)
    return 0.5 * c_e * (I1_e - 3.0)

def W_smc(C):
    # Circumferential fiber ~ e_theta
    a_theta = np.array([0.0, 1.0, 0.0])
    I4m = fiber_invariant(C, a_theta)
    return (c_m1 / (4.0 * c_m2)) * (ufl.exp(c_m2 * (I4m - 1.0)**2) - 1.0)

def W_collagen(C):
    """
    Sum of θ, z, d collagen families, each weighted by β_j.
    Each uses the same (c_c1, c_c2) but different directions.
    """
    # Circumferential
    a_theta = np.array([0.0, 1.0, 0.0])
    I4_theta = fiber_invariant(C, a_theta)

    # Axial
    a_z = np.array([0.0, 0.0, 1.0])
    I4_z = fiber_invariant(C, a_z)

    # Diagonals: ±alpha0 in θ-z plane
    a_d1 = np.array([0.0,
                     np.sin(alpha0),
                     np.cos(alpha0)])
    a_d2 = np.array([0.0,
                     -np.sin(alpha0),
                     np.cos(alpha0)])
    I4_d1 = fiber_invariant(C, a_d1)
    I4_d2 = fiber_invariant(C, a_d2)

    # Common exponential function
    def W_f(I4):
        return (c_c1 / (4.0 * c_c2)) * (ufl.exp(c_c2 * (I4 - 1.0)**2) - 1.0)

    W_theta = beta_theta * W_f(I4_theta)
    W_z = beta_z * W_f(I4_z)
    W_d = beta_d * 0.5 * (W_f(I4_d1) + W_f(I4_d2))

    return W_theta + W_z + W_d

# -------------------------------------------------------------------
# 4. Stage I: Hyperelastic pre-stress
#    W_I = φ_e0 W_e(C) + φ_m0 W_m(C) + φ_c0 W_c(C) + U_vol(J)
#    This is a standard nearly-incompressible hyperelastic solve.
# -------------------------------------------------------------------

def volumetric_energy(J):
    # U(ln J) = 0.5 * kappa * (ln J)^2
    lnJ = ufl.ln(J)
    return 0.5 * kappa * lnJ**2

def strain_energy_stage_I(u):
    F, C, J = kinematics(u)
    W = (phi_e0 * W_elastin(C) +
         phi_m0 * W_smc(C) +
         phi_c0 * W_collagen(C) +
         volumetric_energy(J))
    return W

# Boundary conditions & loading
# For simplicity: "bottom" fixed, "top" loaded with traction/pressure
bottom = mesh.locate_entities_boundary(
    domain, dim-1,
    lambda x: np.isclose(x[2], 0.0, atol=1e-8)
)
top = mesh.locate_entities_boundary(
    domain, dim-1,
    lambda x: np.isclose(x[2], 1.0, atol=1e-8)
)

bottom_dofs = fem.locate_dofs_topological(V, dim-1, bottom)
top_dofs = fem.locate_dofs_topological(V, dim-1, top)

bc_value = fem.Constant(domain, (0.0, 0.0, 0.0))
bc_bottom = fem.dirichletbc(bc_value, bottom_dofs, V)
# u.x.scatter_forward()
bcs = [bc_bottom]

# Define facet measure with "top" marked
facet_tags = mesh.meshtags(domain, dim-1, np.array(top, dtype=np.int32),
                           np.full_like(top, 1, dtype=np.int32))
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

n = ufl.FacetNormal(domain)



# # Visualize boundary conditions with pyvista
# topology, cell_types, x = vtk_mesh(V)
# grid = pv.UnstructuredGrid(topology, cell_types, x)

# plotter = pv.Plotter()
# plotter.add_mesh(grid, show_edges=True, color="white", opacity=0.3, label="Domain")


# bottom_mesh = grid.extract_points(bottom_dofs)
# bottom_face_inds = bottom_mesh.cell_data['vtkOriginalCellIds']
# grid.cell_data['is_bottom'] = np.zeros(grid.n_cells)
# grid.cell_data['is_bottom'][bottom_face_inds] = 1

# # plotter.add_mesh(grid.extract_surface(), scalars='is_bottom', cmap=['white', 'blue'], show_scalar_bar=False, label="Bottom BC (Fixed)")
# plotter.add_points(bottom_mesh.points, color="blue", point_size=10, render_points_as_spheres=True)

# plotter.add_legend()
# plotter.show()



def residual_stage_I(u, v):
    W = strain_energy_stage_I(u)          # scalar energy density
    psi = W * ufl.dx                      # total potential energy

    # Internal virtual work: δΠ_int = d(∫W dx)/du [v]
    R_int = ufl.derivative(psi, u, v)

    # External work from pressure:
    traction = -P0 * n
    R_ext = ufl.dot(traction, v) * ds(1)

    return R_int - R_ext

R_I = residual_stage_I(u, v)
J_I = ufl.derivative(R_I, u, du)

# petsc_options = {
#     "snes_rtol": 1e-8,
#     "snes_atol": 1e-9,
#     "snes_max_it": 25,
#     "snes_monitor": None,
#     "snes_error_if_not_converged": True,
#     "ksp_type": "gmres",
#     "ksp_error_if_not_converged": True,
#     "pc_type": "hypre",
#     "pc_hypre_type": "boomeramg",
# }

petsc_options = {
    "snes_rtol": 1e-8,
    "snes_atol": 1e-9,
    "snes_max_it": 100,
    "snes_monitor": None,
    "snes_error_if_not_converged": True,
    "snes_linesearch_type": "bt",   # backtracking
    "snes_linesearch_monitor": None,
    "ksp_type": "preonly",
    "pc_type": "lu",
    # Uncomment if you have MUMPS:
    # "pc_factor_mat_solver_type": "mumps",
}


problem_I = NonlinearProblem(R_I, u, bcs=bcs, J=J_I, petsc_options=petsc_options, petsc_options_prefix="stage_I")

if rank == 0:
    print("=== Solving Stage I (hyperelastic pre-stress) ===")

t1 = time()
out = problem_I.solve()
t2 = time()
print(f"Stage I solve time: {t2 - t1:.2f} seconds")

# -----------------------------------------------------------
# Compute displacement components
# -----------------------------------------------------------

# Plotting with pyvista
topology, cell_types, x = vtk_mesh(V)
base_mesh = pv.UnstructuredGrid(topology, cell_types, x)

# Calculate displacement components
V_out = fem.functionspace(domain, ("Lagrange", elem_order, (dim,)))
u_out = fem.Function(V_out, name="u")
u_out.interpolate(u)
u_out_arr = u_out.x.array.reshape((-1,dim)) # reshape to NxD array
ux = u_out_arr[:,0]  # x-displacement
uy = u_out_arr[:,1]  # y-displacement
uz = u_out_arr[:,2]  # z-displacement

# Add displacement to pyvista grid
topology, cell_types, x = vtk_mesh(V_out)
pvmesh = pv.UnstructuredGrid(topology, cell_types, x)
pvmesh.point_data["ux"] = ux  # plot x-displacement as scalar field
pvmesh.point_data["uy"] = uy  # plot y-displacement as scalar field
pvmesh.point_data["uz"] = uz  # plot z-displacement as scalar field
pvmesh.point_data["u_mag"] = np.sqrt(ux**2 + uy**2 + uz**2)  # plot magnitude of displacement
pvmesh.point_data["u"] = u_out_arr  # vector displacement field
warped_mesh = pvmesh.warp_by_vector("u", factor=1.0)

# Plot with pyvista
plotter = pv.Plotter()
plotter.add_mesh(warped_mesh, show_edges=True, show_scalar_bar=True, scalars="u_mag", cmap="viridis")
plotter.add_mesh(base_mesh, style='wireframe', color='black', label="Undeformed Mesh")
plotter.show()


b=0
# # -------------------------------------------------------------------
# # 5. Postprocess Stage I: store σ_v0 (DG0 scalar) for Stage II
# # -------------------------------------------------------------------

# DG0 = fem.FunctionSpace(domain, ("DG", 0))
# sigma_v0 = fem.Function(DG0, name="sigma_v0")
# # We compute σ0 = (1/J) P F^T from the Stage I energy and store its mean trace/3
# F_I, C_I, J_I = kinematics(u)
# F_I = ufl.variable(F_I)
# W_I = strain_energy_stage_I(u)
# psi_I = W_I * ufl.dx
# P_I = ufl.diff(psi_I, u)
# # But psi is integrated over dx; for pointwise P, better recompute P from W directly:
# # Here, we recompute from W_I using pointwise AD:
# W_point = strain_energy_stage_I(u)
# P_point = ufl.diff(W_point, F_I)  # dW/dF ~ P
# P_point = ufl.diff(W_point, ufl.grad(u))  # dW/d(grad u) ~ P

# sigma_I = (1.0/J_I) * P_point * F_I.T
# sigma_v_I = (1.0/3.0) * ufl.tr(sigma_I)

# # Project σ_v_I to DG0
# sigma_v0_proj = fem.petsc.LinearProblem(
#     ufl.inner(ufl.TestFunction(DG0), sigma_v_I) * ufl.dx,
#     ufl.inner(ufl.TestFunction(DG0), ufl.Constant(domain, 0.0)) * ufl.dx
# )
# sigma_v0.x.array[:] = sigma_v0_proj.solve().x.array

# if rank == 0:
#     print("Stored sigma_v0 field for Stage II.")

# # -------------------------------------------------------------------
# # 6. Stage II: Mechanobiologically equilibrated constrained mixture
# #    Key idea:
# #      - Use φ_e(J), φ_m(J), φ_c(J)
# #      - Build constituent stresses σ_e, σ_m, σ_c families
# #      - Compute σ^x, then p_h = tr(σ^x)/3 - σ_v0
# #      - σ = σ^x - p_h I
# # -------------------------------------------------------------------

# # To access sigma_v0 at quadrature points, we'll interpolate DG0 into an expression
# sigma_v0_expr = sigma_v0  # fem.Function used directly in UFL

# def mass_fractions(J):
#     """
#     Mechanobiologically equilibrated mass fractions (eta_ij = 1).
#     φ_e(J) = φ_e0 / J
#     φ_m(J) = φ_m0 * (1 - φ_e0/J) / (1 - φ_e0)
#     φ_c(J) = φ_c0 * (1 - φ_e0/J) / (1 - φ_e0)
#     """
#     phi_e = phi_e0 / J
#     denom = (1.0 - phi_e0)
#     phi_turnover_factor = (1.0 - phi_e0 / J) / denom
#     phi_m = phi_m0 * phi_turnover_factor
#     phi_c = phi_c0 * phi_turnover_factor
#     return phi_e, phi_m, phi_c

# def cmm_extra_stress(F, C, J):
#     """
#     Build the 'extra' mixture Cauchy stress σ^x = sum σ_α (no -p term).
#     This uses a hyperelastic-like construction with current mass fractions.
#     """
#     phi_e, phi_m, phi_c = mass_fractions(J)

#     # Total Cauchy stress from each energy via AD:
#     # For each constituent, build W_alpha and derive P_alpha, then σ_alpha.
#     # NOTE: We approximate SMC and collagen as standard hyperelastic fibers here.
#     W_e = phi_e * W_elastin(C)
#     W_m = phi_m * W_smc(C)
#     W_tot = W_e + W_m + W_c

#     # Derivative wrt F: P = dW_tot/dF
#     F_var = ufl.variable(F)
#     W_tot_var = phi_e * W_elastin(C) + phi_m * W_smc(C) + phi_c * W_collagen(C)
#     P_tot = ufl.diff(W_tot_var, F_var)
#     sigma_x = (1.0/J) * P_tot * F.T
#     P_tot = ufl.diff(W_tot, ufl.grad(u))
#     sigma_x = (1.0/J) * P_tot * F.T

#     return sigma_x

# def cmm_sigma(F, C, J):
#     """
#     Mechanobiologically equilibrated σ:
#       σ = σ^x - p_h I,
#       p_h = tr(σ^x)/3 - σ_v0
#     """
#     sigma_x = cmm_extra_stress(F, C, J)
#     sigma_vx = (1.0/3.0) * ufl.tr(sigma_x)
#     # sigma_v0_expr is field of the original volumetric stress
#     p_h = sigma_vx - sigma_v0_expr
#     sigma = sigma_x - p_h * I
#     return sigma

# def residual_stage_II(u, v):
#     F, C, J = kinematics(u)
#     sigma = cmm_sigma(F, C, J)
#     P = sigma * ufl.inv(F).T  # First Piola
#     R_int = ufl.inner(P, ufl.grad(v)) * ufl.dx

#     # Now apply NEW loading — e.g., increased pressure Ph = 1.6 P0
#     Ph = 1.6 * P0
#     traction = -Ph * n
#     R_ext = ufl.dot(traction, v) * ds(1)
#     return R_int - R_ext

# # Reset displacement for Stage II (or start from Stage I solution)
# u2 = fem.Function(V, name="u_stageII")
# u2.x.array[:] = u.x.array  # start from prestressed state

# R_II = ufl.replace(residual_stage_II(u, v), {u: u2})
# J_II = ufl.derivative(R_II, u2, du)
# problem_II = NonlinearProblem(R_II, u2, bcs, J_II)
# solver_II = NewtonSolver(comm, problem_II)
# solver_II.options_prefix = "stage_II"
# solver_II = NewtonSolver(comm, problem_II)
# solver_II.rtol = 1e-8
# solver_II.report = True

# if rank == 0:
#     print("=== Solving Stage II (mechanobiologically equilibrated CMM) ===")
# solver_II.solve(u2)

# # -------------------------------------------------------------------
# # 7. Output
# # -------------------------------------------------------------------

# if rank == 0:
#     print("Saving results to XDMF...")
# with io.XDMFFile(comm, "cmm_results_stageI_stageII.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)
#     xdmf.write_function(u, 0.0)
#     xdmf.write_function(u2, 1.0)
#     xdmf.write_function(sigma_v0, 0.0)
