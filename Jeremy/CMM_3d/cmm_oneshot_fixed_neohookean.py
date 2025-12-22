"""
Simple neo-Hookean hyperelastic solve in DOLFINx/FEniCSx.

This script demonstrates a basic neo-Hookean material model applied to a cylindrical tube
under internal pressure loading.

Neo-Hookean strain energy density:
    W = (μ/2)(I₁ - 3) + (κ/2)(ln J)²

where:
    - μ is the shear modulus
    - κ is the bulk modulus (for near-incompressibility)
    - I₁ = tr(C) is the first invariant of the right Cauchy-Green tensor
    - J = det(F) is the Jacobian

Boundary conditions:
- Internal pressure applied to inner surface (tag 1)
- End caps (tags 3 and 4) fixed in all directions

Assumptions:
- The tube mesh uses boundary tags:
    1 = luminal (inner) surface
    2 = outer surface
    3 = end cap at z=min
    4 = end cap at z=max
- Tube axis is aligned with global z

Outputs:
- VTU file with displacement + von Mises + hydrostatic stress

Run:
    mpirun -n 4 python cmm_oneshot_fixed_neohookean.py
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
# Read mesh and boundary tags (no fiber data needed for neo-Hookean)
# -----------------------------------------------------------------------------

vmesh = generate_tube_volume_mesh(
    axial_length=10.0,  # [mm]
    lumen_diameter=1.294,  # [mm]
    wall_thickness=0.25,  # [mm]
    n_axial=50,
    n_circ=36,
    n_radial=4,
    axis="z",
    mesh_type="hex8",  # Options: 'tet4', 'tet10', 'hex8', 'hex20'
    fiber_angles=None,  # No fibers needed for neo-Hookean
)

model = tube_mesh_to_dolfinx_model(vmesh)

domain = model.mesh
facet_tags = model.facet_tags



elem_order = domain.geometry.cmap.degree
dim = domain.geometry.dim
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)


# -----------------------------------------------------------------------------
# Material parameters (neo-Hookean)
# -----------------------------------------------------------------------------

kPa_to_N_per_mm2 = 1e-3  # 1 kPa = 1e-3 N/mm^2

# Applied internal pressure
P0 = fem.Constant(domain, PETSc.ScalarType(13.98 * kPa_to_N_per_mm2))  # 13.98 kPa

# Neo-Hookean material parameters
mu = fem.Constant(domain, PETSc.ScalarType(50.0 * kPa_to_N_per_mm2))   # Shear modulus (50 kPa)
kappa = fem.Constant(domain, PETSc.ScalarType(1000.0 * kPa_to_N_per_mm2))  # Bulk modulus (1000 kPa for near-incompressibility)





# -----------------------------------------------------------------------------
# Function spaces and kinematics
# -----------------------------------------------------------------------------

V = fem.functionspace(domain, ("Lagrange", elem_order, (dim,)))
u = fem.Function(V, name="u")
v = ufl.TestFunction(V)
dx = ufl.Measure("dx", domain=domain)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
n = ufl.FacetNormal(domain)

# Kinematics
I = ufl.Identity(dim)
F = I + ufl.grad(u)
C = F.T * F
J = ufl.det(F)





# -----------------------------------------------------------------------------
# Strain energy density (neo-Hookean)
# -----------------------------------------------------------------------------

# First invariant of right Cauchy-Green tensor
I1 = ufl.tr(C)

# Neo-Hookean strain energy density
# W = (μ/2)(I₁ - 3) + (κ/2)(ln J)²
W_iso = 0.5 * mu * (I1 - 3.0)           # Isochoric part
W_vol = 0.5 * kappa * ufl.ln(J)**2      # Volumetric part
W_total = W_iso + W_vol


# -----------------------------------------------------------------------------
# Weak form: residual and Jacobian
# -----------------------------------------------------------------------------

# Traction = -P0*n on inner surface (tag 1)
traction_inner = -P0 * n
R = ufl.derivative(W_total * dx, u, v) - ufl.dot(traction_inner, v) * ds(1)
J_form = ufl.derivative(R, u, ufl.TrialFunction(V))


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
# Solve nonlinear problem
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


problem = NonlinearProblem(R, u, bcs=bcs, J=J_form, petsc_options=petsc_options, petsc_options_prefix="neohookean")

if rank == 0:
    print("\n\n=== Solving neo-Hookean hyperelastic problem ===")
    print(f"Using {comm.size} MPI processes")
    print(f"DOFs (global): {V.dofmap.index_map.size_global * V.dofmap.index_map_bs}")
    print(f"DOFs per process (local on rank 0): {V.dofmap.index_map.size_local * V.dofmap.index_map_bs}")

from time import time
t1 = time()
out = problem.solve()
t2 = time()

total_seconds = t2 - t1
hours = int(total_seconds // 3600)
minutes = int((total_seconds % 3600) // 60)
seconds = int(total_seconds % 60)
print(f"Solve time: {hours} hours, {minutes} minutes, {seconds} seconds")
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


with VTKFile(domain.comm, "neohookean_tube_results.vtu", "w") as f:
    f.write_mesh(domain)
    f.write_function(u)
    