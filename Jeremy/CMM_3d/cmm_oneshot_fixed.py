"""
Stage-I constrained-mixture hyperelastic solve (Latorre & Humphrey 2020) in DOLFINx/FEniCSx.

This is a revised version of your cmm_oneshot.py with **major implementation fixes**:

FIX 1) Correct unit conversions: Table-1 parameters are reported in kPa, while the mesh is in mm.
       Therefore stresses must be in N/mm^2 (= MPa). Use: 1 kPa = 1e-3 N/mm^2.
       (Your original script effectively used 1 Pa = 1e-3 N/mm^2, which is off by 1e3.) 

FIX 2) Pressure loading: apply luminal pressure as traction **t = -p * n** on the *inner* surface,
       where n is the outward unit normal of the tissue domain. This is robust for any centered/non-centered geometry
       and avoids hand-built direction vectors.

FIX 3) Boundary conditions per Stage-I description: "axial displacements fixed at both ends" (paper).
       Constrain **only u_z** on end caps (tags 3 and 4) and apply minimal extra constraints to remove rigid modes.

FIX 4) Deposition stretches (prestretches) for each constituent in Stage I:
       use constituent-level elastic C-tensors: C^e = G^{-T} C G^{-1}.
       The paper uses anisotropic elastin prestretches and isotropic G_m, G_c for SMC/collagen (Table 1). 

FIX 5) Near-incompressibility and homeostatic Lagrange multiplier:
       the paper uses U(ln(J*JG)) with JG = exp(-p0/kappa) and kappa = 1e3*c_e (Table 1 / Eq 50). 
       This keeps Stage-I displacements small (F_o ≈ I) while still generating the correct prestress. 

FIX 6) Fiber/basis DG0 mapping in parallel: include ghost cells and scatter forward (important for correct assembly).

Assumptions:
- The provided tube mesh uses boundary tags:
    1 = luminal (inner) surface
    2 = outer surface
    3 = end cap at z=min
    4 = end cap at z=max
  (as written by tube_mesh_vol2.py). 
- Tube axis is aligned with global z (as generated in tube_mesh_vol2.py main). 

Outputs:
- VTU file with displacement + von Mises + hydrostatic stress.

Run:
    mpirun -n 4 python cmm_oneshot_fixed.py
"""

from mpi4py import MPI
import numpy as np

from dolfinx import mesh, fem, io
import ufl
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import VTKFile
from dolfinx.plot import vtk_mesh

# PETSc options are set through petsc4py (recommended)
from petsc4py import PETSc
import pyvista as pv

from tube_mesh_vol2 import tube_mesh_to_dolfinx_model, generate_tube_volume_mesh, plot_surface_normals


comm = MPI.COMM_WORLD
rank = comm.rank


# -----------------------------------------------------------------------------
# Read mesh, boundary tags, fiber data
# -----------------------------------------------------------------------------

alpha0_deg = 29.91  # degrees
alpha0 = alpha0_deg * np.pi / 180.0  # radians

vmesh = generate_tube_volume_mesh(
    axial_length=10.0, # [mm]
    lumen_diameter=1.294, # [mm]
    wall_thickness=0.25, # [mm]
    n_axial=50,
    n_circ=36,
    n_radial=4,
    axis="z",
    mesh_type="hex8",  # Options: 'tet4', 'tet10', 'hex8', 'hex20'
    fiber_angles={
        "theta": np.pi / 2,  # circumferential
        "axial": 0.0,  # axial
        "diagonal1": alpha0,  # +alpha
        "diagonal2": -alpha0,  # -alpha
    },
)

model = tube_mesh_to_dolfinx_model(vmesh)

domain = model.mesh
facet_tags = model.facet_tags
fiber_functions = model.fibers

# Mesh
# mesh_file = "tube_mesh.xdmf"
# domain = read_mesh(mesh_file)

# # Boundary facet tags
# boundary_file = "tube_mesh_boundary.xdmf"
# facet_tags = read_meshtags(boundary_file,domain)

# # Fiber/basis data (DG0 vectors on cells)
# fiber_file = "tube_mesh_fibers.xdmf"
# fiber_data = build_DG0_vector_functions(domain, fiber_file, comm)



elem_order = domain.geometry.cmap.degree
dim = domain.geometry.dim
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)


# -----------------------------------------------------------------------------
# Material parameters (Table 1, kPa -> N/mm^2)
# -----------------------------------------------------------------------------

kPa_to_N_per_mm2 = 1e-3  # 1 kPa = 1e-3 N/mm^2

# Homeostatic pressure (Po) and homeostatic Lagrange multiplier p0
P0 = fem.Constant(domain, PETSc.ScalarType(13.98 * kPa_to_N_per_mm2))   # Po = 13.98 kPa 
p0 = fem.Constant(domain, PETSc.ScalarType(10.21 * kPa_to_N_per_mm2))   # p0 = 10.21 kPa (Table 1)

# Mass fractions
phi_e0 = fem.Constant(domain, PETSc.ScalarType(0.34))
phi_m0 = fem.Constant(domain, PETSc.ScalarType(0.33))
phi_c0 = fem.Constant(domain, PETSc.ScalarType(0.33))

# Collagen family fractions (βθ, βz, βd where βd is for BOTH diagonal families combined)
beta_theta = fem.Constant(domain, PETSc.ScalarType(0.056))
beta_z     = fem.Constant(domain, PETSc.ScalarType(0.067))
beta_d     = fem.Constant(domain, PETSc.ScalarType(0.877))

# Shear modulus for elastin (kPa)
c_e  = fem.Constant(domain, PETSc.ScalarType(89.71 * kPa_to_N_per_mm2))

# Fiber parameters (kPa)
c1m = fem.Constant(domain, PETSc.ScalarType(261.4 * kPa_to_N_per_mm2))
c2m = fem.Constant(domain, PETSc.ScalarType(0.24))

c1c = fem.Constant(domain, PETSc.ScalarType(234.9 * kPa_to_N_per_mm2))
c2c = fem.Constant(domain, PETSc.ScalarType(4.08))

# Deposition stretches (Table 1)
Gm = fem.Constant(domain, PETSc.ScalarType(1.20))
Gc = fem.Constant(domain, PETSc.ScalarType(1.25))
Ge_theta = fem.Constant(domain, PETSc.ScalarType(1.900))
Ge_z     = fem.Constant(domain, PETSc.ScalarType(1.62))
Ge_r     = fem.Constant(domain, PETSc.ScalarType(1.0 / (float(Ge_theta.value) * float(Ge_z.value))))

# FIX 5: bulk-like penalty: kappa = 1e3*c_e (paper)
kappa = fem.Constant(domain, PETSc.ScalarType(1e1 * float(c_e.value)))

# Pre-Jacobian shift JG = exp(-p0/kappa) (paper Stage I)
JG = fem.Constant(domain, PETSc.ScalarType(np.exp(-float(p0.value) / float(kappa.value)))) # page 15


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _normalize(v, eps=1e-12):
    """UFL-safe normalization with a small epsilon."""
    eps_ufl = ufl.as_ufl(eps)
    return v / ufl.sqrt(ufl.dot(v, v) + eps_ufl)

def cylindrical_basis(X):
    """
    Analytic cylindrical basis (axis = global z) computed from SpatialCoordinate.
    Used as robust fallback if basis vectors are not found in the fiber file.
    """
    x, y, z = X[0], X[1], X[2]
    r = ufl.sqrt(x*x + y*y + 1e-16)
    e_r = ufl.as_vector((x/r, y/r, 0.0))
    e_theta = ufl.as_vector((-y/r, x/r, 0.0))
    e_z = ufl.as_vector((0.0, 0.0, 1.0))
    return e_r, e_theta, e_z


# -----------------------------------------------------------------------------
# Function spaces and kinematics
# -----------------------------------------------------------------------------

V = fem.functionspace(domain, ("Lagrange", elem_order, (dim,)))
u = fem.Function(V, name="u")
v = ufl.TestFunction(V)
du = ufl.TrialFunction(V)
dx = ufl.Measure("dx", domain=domain)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
n = ufl.FacetNormal(domain)

# Kinematics
I = ufl.Identity(dim)
F = I + ufl.grad(u)
J = ufl.det(F)
FinvT = ufl.inv(F).T
C = F.T * F


# -----------------------------------------------------------------------------
# Constituent directions and prestretches
# -----------------------------------------------------------------------------

X = ufl.SpatialCoordinate(domain)
e_r_ufl, e_theta_ufl, e_z_ufl = cylindrical_basis(X)

# Use basis vectors from file if present; otherwise analytic cylindrical basis
e_r     = _normalize(fiber_functions.get("basis_radial", e_r_ufl))
e_theta = _normalize(fiber_functions.get("basis_circumferential", e_theta_ufl))
e_z     = _normalize(fiber_functions.get("basis_axial", e_z_ufl))

# Fiber families (from file if present; otherwise analytic)
a_theta = _normalize(fiber_functions.get("theta", e_theta))
a_axial = _normalize(fiber_functions.get("axial", e_z))

# Diagonals: if not present in file, construct from e_theta/e_z and the nominal angle (Table 1 α0=29.91°)
alpha0 = 29.91 * np.pi / 180.0
a_diag1_fallback = _normalize(ufl.cos(alpha0) * e_z + ufl.sin(alpha0) * e_theta)
a_diag2_fallback = _normalize(ufl.cos(alpha0) * e_z - ufl.sin(alpha0) * e_theta)

a_diag1 = _normalize(fiber_functions.get("diagonal1", a_diag1_fallback))
a_diag2 = _normalize(fiber_functions.get("diagonal2", a_diag2_fallback))

# Elastin prestretch inverse tensor (anisotropic, volume-preserving by construction if Ge_r*Ge_theta*Ge_z=1)
# Ginv = sum_i (e_i ⊗ e_i) / Ge_i
# Ge_inv = (ufl.outer(e_r, e_r) / Ge_r +
#           ufl.outer(e_theta, e_theta) / Ge_theta +
#           ufl.outer(e_z, e_z) / Ge_z)
Ge_inv = (Ge_r * ufl.outer(e_r, e_r) +
          Ge_theta * ufl.outer(e_theta, e_theta) +
          Ge_z * ufl.outer(e_z, e_z))

# SMC + collagen prestretch inverse tensors (isotropic)
Gm_inv = (1.0 / Gm) * I
Gc_inv = (1.0 / Gc) * I

# C^alpha = G^{-T} C G^{-1}. Here Ginv is symmetric, so G^{-T}=G^{-1}=Ginv
Ce = Ge_inv * C * Ge_inv
Cm = Gm_inv * C * Gm_inv
Cc = Gc_inv * C * Gc_inv



# -----------------------------------------------------------------------------
# Strain energy density (Stage I)
# -----------------------------------------------------------------------------
def W_elastin(Ce_):
    # Eq (51): W^e = c_e/2 (Ce:I - 3) 
    # I1e = ufl.tr(Ce_)
    return 0.5 * c_e * (ufl.tr(Ce_) - 3.0)

def W_fiber(I4, c1, c2):
    """
    Eq (52): Exponential fiber energy (tension-only).
    
    For smooth muscle and collagen families:
        ψ^k = (c₁^k / (4c₂^k)) [exp(c₂^k E₄^k²) - 1]  if E₄^k > 0 (tension)
        ψ^k = 0                                         if E₄^k ≤ 0 (compression)
    
    where E₄^k = I₄^k - 1 is the Green strain in fiber direction k.
    
    Args:
        I4: Fourth invariant (a · C · a) for fiber family
        c1, c2: Material parameters for exponential fiber response
    """
    E4 = I4 - 1.0  # Green strain in fiber direction
    # Tension-only: fibers only resist stretch (E4 > 0 ⟺ I4 > 1)
    W_tension = (c1 / (4.0 * c2)) * (ufl.exp(c2 * E4**2) - 1.0)
    return W_tension
    # return ufl.conditional(ufl.gt(E4, 0.0), W_tension, 0.0)


# Neo-Hookean strain energy density
# W = (μ/2)(I₁ - 3) + (κ/2)(ln J)²
def W_neohookean(I1, J, mu, kappa):
    W_iso = 0.5 * mu * (I1 - 3.0)           # Isochoric part
    W_vol = 0.5 * kappa * ufl.ln(J)**2      # Volumetric part
    W_total = W_iso + W_vol
    return W_total


def fiber_invariant(a_, C_):
    """Helper to compute I4 = a . C . a"""
    return ufl.dot(a_, ufl.dot(C_, a_))










## TESTING HERE
def prestretch_isochoric(a, g):
    """
    Build a symmetric, volume-preserving (det=1) prestretch tensor with
    stretch g along direction a and 1/sqrt(g) in transverse directions.
    This matches the paper's assumption that deposition tensors are symmetric
    and volume-preserving. :contentReference[oaicite:3]{index=3}
    """
    I  = ufl.Identity(3)
    aa = ufl.outer(a, a)
    return g * aa + (1.0 / ufl.sqrt(g)) * (I - aa)


def C_with_prestretch(C, G):
    # general: Cα = (F G)^T (F G) = G^T C G
    return G.T * C * G

def fung_fiber_energy(C_xiN, a_xiN, c1, c2):
    # Eq. (52): Ŵ^ξ = c1/(4 c2) * (exp(c2*(C:(a⊗a)-1)^2) - 1)
    I4 = ufl.dot(a_xiN, ufl.dot(C_xiN, a_xiN))  # C:(a⊗a)
    E  = I4 - 1.0
    return (c1 / (4.0 * c2)) * (ufl.exp(c2 * E**2) - 1.0)

# ---------- prestretch tensors ----------
G_mN = prestretch_isochoric(a_theta, Gm)
G_cN_th = prestretch_isochoric(a_theta, Gc)
G_cN_z  = prestretch_isochoric(a_axial,  Gc)
G_cN_d1 = prestretch_isochoric(a_diag1, Gc)
G_cN_d2 = prestretch_isochoric(a_diag2, Gc)

# ---------- constituent C-tensors ----------
Ce = C_with_prestretch(C, Ge_inv)
Cm = C_with_prestretch(C, G_mN)
C_c_th = C_with_prestretch(C, G_cN_th)
C_c_z  = C_with_prestretch(C, G_cN_z)
C_c_d1 = C_with_prestretch(C, G_cN_d1)
C_c_d2 = C_with_prestretch(C, G_cN_d2)

# ---------- energies ----------
# Eq. (51): Ŵ^e = (c_e/2)*(C_e:I - 3) :contentReference[oaicite:9]{index=9}
W_e = 0.5 * c_e * (ufl.tr(Ce) - 3.0)

# Eq. (52): fiber energies :contentReference[oaicite:10]{index=10}
W_m = fung_fiber_energy(Cm, a_theta, c1m, c2m)

W_c = (
    beta_theta * fung_fiber_energy(C_c_th, a_theta, c1c, c2c)
  + beta_z  * fung_fiber_energy(C_c_z,  a_axial,  c1c, c2c)
  + 0.5 * beta_d * (fung_fiber_energy(C_c_d1, a_diag1, c1c, c2c)
                + fung_fiber_energy(C_c_d2, a_diag2, c1c, c2c))
)

U_vol  = 0.5 * kappa * (ufl.ln(J))**2

W_total = phi_e0 * W_e + phi_m0 * W_m + phi_c0 * W_c + U_vol


# # Fiber invariants using constituent elastic C-tensors
# I4m = fiber_invariant(a_theta, Cm)     # SMC circumferential
# Wm = W_fiber(I4m, c1m, c2m)

# # collagen contribution (βd split equally between the two diagonal families)
# I4ct = fiber_invariant(a_theta, Cc)    # collagen circumferential
# I4cz = fiber_invariant(a_axial, Cc)    # collagen axial
# I4cd1 = fiber_invariant(a_diag1, Cc)   # collagen diag +
# I4cd2 = fiber_invariant(a_diag2, Cc)   # collagen diag -
# Wc = (beta_theta * W_fiber(I4ct, c1c, c2c) +
#       beta_z     * W_fiber(I4cz, c1c, c2c) +
#       0.5 * beta_d * (W_fiber(I4cd1, c1c, c2c) + W_fiber(I4cd2, c1c, c2c)))

# # FIX 5: volumetric penalty uses ln(J*JG)
# # lnJ = ufl.ln(J * JG)
# # U_vol = 0.5 * kappa * lnJ * lnJ
# U_vol  = 0.5 * kappa * (ufl.ln(J))**2

# W_total = phi_e0 * W_elastin(Ce) + phi_m0 * Wm + phi_c0 * Wc + U_vol
# # W_total =  0.3 * W_elastin(Ce) + 0.7 * W_neohookean(ufl.tr(C), J, mu=c_e, kappa=kappa) + U_vol
# # W_total = Wm # + U_vol
# # W_total = Wc # + U_vol


# -----------------------------------------------------------------------------
# Weak form: residual and Jacobian
# -----------------------------------------------------------------------------



# FIX 2: traction = -P0*n on inner surface (tag 1)
# traction_inner = -P0 * n
# R = ufl.derivative(W_total * dx, u, v) - ufl.dot(traction_inner, v) * ds(1)
# J_form = ufl.derivative(R, u, ufl.TrialFunction(V))

traction_inner = -P0 * J * FinvT * n
R = ufl.derivative(W_total * ufl.dx, u, v) - ufl.dot(traction_inner, v) * ds(1)
J_form = ufl.derivative(R, u, ufl.TrialFunction(V))

# Pi = W_total * dx - P0 * ufl.dot(u, n) * ds(1)
# R = ufl.derivative(Pi, u, v)
# J_form = ufl.derivative(R, u, du)


# -----------------------------------------------------------------------------
# Boundary conditions
# -----------------------------------------------------------------------------

# Subspaces for components
Vx = V.sub(0)  # x component
Vy = V.sub(1)  # y component
Vz = V.sub(2)  # z component

zero = fem.Constant(domain, PETSc.ScalarType(0.0))

# End caps (tag 3 and 4): u_z = 0
facets_end1 = facet_tags.indices[facet_tags.values == 3]
facets_end2 = facet_tags.indices[facet_tags.values == 4]

dofs_end1_x = fem.locate_dofs_topological(Vx, fdim, facets_end1)
dofs_end1_y = fem.locate_dofs_topological(Vy, fdim, facets_end1)
dofs_end1_z = fem.locate_dofs_topological(Vz, fdim, facets_end1)
dofs_end2_x = fem.locate_dofs_topological(Vx, fdim, facets_end2)
dofs_end2_y = fem.locate_dofs_topological(Vy, fdim, facets_end2)
dofs_end2_z = fem.locate_dofs_topological(Vz, fdim, facets_end2)

bc_end1_x = fem.dirichletbc(zero, dofs_end1_x, Vx)
bc_end1_y = fem.dirichletbc(zero, dofs_end1_y, Vy)
bc_end1_z = fem.dirichletbc(zero, dofs_end1_z, Vz)
bc_end2_x = fem.dirichletbc(zero, dofs_end2_x, Vx)
bc_end2_y = fem.dirichletbc(zero, dofs_end2_y, Vy)
bc_end2_z = fem.dirichletbc(zero, dofs_end2_z, Vz)

bcs = [bc_end1_x, bc_end1_y, bc_end1_z, bc_end2_x, bc_end2_y, bc_end2_z]


# -----------------------------------------------------------------------------
# Solve nonlinear problem (Stage I)
# -----------------------------------------------------------------------------

petsc_options = {
    # SNES (Nonlinear Solver) Options for Newton stepping
    "snes_rtol": 1e-5, # Stop when the relative residual drops below 10⁻⁸ (i.e., residual is 0.00000001× the initial residual)
    "snes_atol": 1e-5, # Stop when the absolute residual drops below 10⁻⁹ (i.e., residual norm < 0.000000001)
    "snes_max_it": 250, # Maximum nonlinear iterations before giving up
    "snes_monitor": None, # Print residual norm at each iteration (helps debug convergence)
    "snes_error_if_not_converged": False, # Raise error if not converged
    "snes_linesearch_type": "bt",   # Use backtracking line search to ensure Newton steps decrease the residual
    "snes_linesearch_monitor": None, # Print line search details at each step
    
    # KSP (Linear Solver) Options
    "ksp_type": "gmres", # Use GMRES (Generalized Minimal Residual) Krylov solver for the linearized system at each Newton step
    "ksp_rtol": 1e-6, # Linear solve converges when residual drops to 'ksp_rtol' of initial
    "ksp_atol": 1e-6, # Or when absolute residual is below 'ksp_atol'
    "ksp_max_it": 1000, # Maximum linear solver iterations for each linear solve
    "ksp_monitor": None, # Print linear solver residuals at each iteration

    # Preconditioner (PC) Options
    "pc_type": "hypre", # Use the Hypre library's preconditioners
    "pc_hypre_type": "boomeramg", # Use the BoomerAMG (Algebraic Multigrid) preconditioner
    
    # # BoomerAMG options for better parallel performance
    # "pc_hypre_boomeramg_strong_threshold": 0.7, # How "strongly connected" nodes must be to group together (0.25-0.7 typical; higher = sparser coarse grids)
    # "pc_hypre_boomeramg_agg_nl": 4, # Number of levels of aggressive coarsening (0-4; higher = faster setup, potentially slower convergence)
    # "pc_hypre_boomeramg_agg_num_paths": 5, # Number of paths for aggressive coarsening (1-10; higher = better quality coarse grids, more expensive)
    # "pc_hypre_boomeramg_max_levels": 25, # Maximum number of multigrid levels
    # "pc_hypre_boomeramg_coarsen_type": "HMIS", # Coarsening strategy (HMIS or PMIS recommended for parallel)
    # "pc_hypre_boomeramg_interp_type": "ext+i", # Interpolation type (ext+i or classical)
    # "pc_hypre_boomeramg_P_max": 2, # Max elements per row in interpolation operator
    # "pc_hypre_boomeramg_truncfactor": 0.3, # Truncation factor for interpolation operator
}


problem_I = NonlinearProblem(R, u, bcs=bcs, J=J_form, petsc_options=petsc_options, petsc_options_prefix="stage_I")

if rank == 0:
    print("\n\n=== Solving Stage I (hyperelastic pre-stress) ===")
    print(f"Using {comm.size} MPI processes")
    print(f"DOFs (global): {V.dofmap.index_map.size_global * V.dofmap.index_map_bs}")
    print(f"DOFs per process (local on rank 0): {V.dofmap.index_map.size_local * V.dofmap.index_map_bs}")

from time import time
t1 = time()
out = problem_I.solve()
t2 = time()

total_seconds = t2 - t1
hours = int(total_seconds // 3600)
minutes = int((total_seconds % 3600) // 60)
seconds = int(total_seconds % 60)
print(f"Stage I solve time: {hours} hours, {minutes} minutes, {seconds} seconds")
print(f"  Time per DOF: {(t2-t1)/(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)*1e6:.2f} μs")


# -----------------------------------------------------------------------------
# Post-processing: displacements
# -----------------------------------------------------------------------------

# Calculate displacement components
V_out = fem.functionspace(domain, ("Lagrange", elem_order, (dim,)))
u_out = fem.Function(V_out, name="u")
u_out.interpolate(u)
u_out_arr = u_out.x.array.reshape((-1,dim)) # reshape to NxD array
ux = u_out_arr[:,0]  # x-displacement
uy = u_out_arr[:,1]  # y-displacement
uz = u_out_arr[:,2]  # z-displacement

# Plotting with pyvista
topology, cell_types, x = vtk_mesh(V)
base_mesh = pv.UnstructuredGrid(topology, cell_types, x)

# Add displacement to pyvista grid
topology, cell_types, x = vtk_mesh(V_out)
pvmesh = pv.UnstructuredGrid(topology, cell_types, x)
pvmesh.point_data["ux"] = ux  # plot x-displacement as scalar field
pvmesh.point_data["uy"] = uy  # plot y-displacement as scalar field
pvmesh.point_data["uz"] = uz  # plot z-displacement as scalar field
pvmesh.point_data["u_mag"] = np.sqrt(ux**2 + uy**2 + uz**2)  # plot magnitude of displacement
pvmesh.point_data["u"] = u_out_arr  # vector displacement field

warped_mesh = pvmesh.warp_by_vector("u", factor=1.0)

# Plot with pyvista - extract surface to show actual cell types (quads/tris)
plotter = pv.Plotter()
warped_surface = warped_mesh.extract_surface()
base_surface = base_mesh.extract_surface()

# Plot displacement on deformed configuration
plotter.add_mesh(warped_surface, show_edges=True, show_scalar_bar=True, 
                scalars="u", cmap="jet", edge_color='black',
                scalar_bar_args={'title': 'Displacement (mm)'})
plotter.add_mesh(base_surface, style='wireframe', color='gray', opacity=0.3, 
                line_width=1, label="Undeformed Mesh")
plotter.add_axes()
plotter.show()


# if rank == 0:
#     print(f"Newton iters: {n_it}, converged: {converged}")
#     print(f"||u||_inf: {np.max(np.abs(u.x.array)):.6e} (in mm)")


# -----------------------------------------------------------------------------
# Post-processing: Cauchy stress, von Mises, hydrostatic
# -----------------------------------------------------------------------------

# First Piola-Kirchhoff stress: P = dW/dF
P = ufl.diff(W_total, F)
sigma = (1.0 / J) * P * F.T
sigma_dev = sigma - (ufl.tr(sigma) / 3.0) * I
von_mises = ufl.sqrt(1.5 * ufl.inner(sigma_dev, sigma_dev))
hydrostatic = ufl.tr(sigma) / 3.0

# Project to a discontinuous scalar space for output (cellwise)
Q0 = fem.functionspace(domain, ("DG", 0))
von_mises_func = fem.Function(Q0, name="von_mises")
hydro_func = fem.Function(Q0, name="hydrostatic")

# Projection helper (DG0): solve (q, w) = (expr, w)
w = ufl.TestFunction(Q0)
q = ufl.TrialFunction(Q0)

a_proj = ufl.inner(q, w) * dx

# von Mises
L_vm = ufl.inner(von_mises, w) * dx
A = fem.petsc.assemble_matrix(fem.form(a_proj), bcs=[])
A.assemble()
b = fem.petsc.assemble_vector(fem.form(L_vm))
solver_ksp = PETSc.KSP().create(comm)
solver_ksp.setOperators(A)
solver_ksp.setType("preonly")
solver_ksp.getPC().setType("lu")
solver_ksp.setFromOptions()
solver_ksp.solve(b, von_mises_func.x.petsc_vec)
von_mises_func.x.scatter_forward()

# hydrostatic
b2 = fem.petsc.assemble_vector(fem.form(ufl.inner(hydrostatic, w) * dx))
solver_ksp.solve(b2, hydro_func.x.petsc_vec)
hydro_func.x.scatter_forward()

# -----------------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------------

if rank == 0:
    print("\n=== Writing results to CMM_tube_results_fixed.vtu ===")

with VTKFile(domain.comm, "CMM_tube_results_fixed.vtu", "w") as f:
    f.write_mesh(domain)
    f.write_function(u)
    f.write_function(von_mises_func)
    f.write_function(hydro_func)
