# mooney_tube_demo.py
#
# Simple 2D "tube wall" patch with a compressible Mooney–Rivlin material.
# Geometry: [Ri, Ro] x [0, Lz] in (r,z)-like coordinates.
# BCs:
#   - Outer radius (x = Ro): clamped (u = 0)
#   - Inner radius (x = Ri): internal pressure p_in
# Output: displacement field u -> mooney_tube_results.xdmf

from mpi4py import MPI
import numpy as np
import ufl

from petsc4py import PETSc
from dolfinx import mesh, fem, io, log
from dolfinx.fem.petsc import NewtonSolverNonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.plot import vtk_mesh

import pyvista as pv

Scalar = PETSc.ScalarType

# -----------------------------------------------------------
# Geometry and mesh
# -----------------------------------------------------------
Ri = 8.0e-3   # inner radius [m]
Ro = 10.0e-3  # outer radius [m]  -> 2 mm thick wall
Lz = 20.0e-3  # "length" of tube [m]

nr, nz = 40, 80  # mesh resolution (radial x axial)

domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([Ri, 0.0]), np.array([Ro, Lz])],
    [nr, nz],
    cell_type=mesh.CellType.triangle,
)

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

# -----------------------------------------------------------
# Function space (vector displacement)
# -----------------------------------------------------------
dim = domain.geometry.dim
V = fem.functionspace(domain, ("Lagrange", 2, (dim,)))  # P2 vector space

u = fem.Function(V, name="u")          # unknown displacement
v = ufl.TestFunction(V)                # test function
du = ufl.TrialFunction(V)              # for Jacobian

I = ufl.Identity(dim)

# -----------------------------------------------------------
# Material: compressible Mooney–Rivlin
# -----------------------------------------------------------
c1    = fem.Constant(domain, Scalar(80e3))     # [Pa]
c2    = fem.Constant(domain, Scalar(20e3))     # [Pa]
kappa = fem.Constant(domain, Scalar(1.0e7))    # bulk modulus [Pa]

def deformation_gradient(u):
    return I + ufl.grad(u)

# F = deformation_gradient(u)
F = ufl.variable(I + ufl.grad(u)) # Use this instead to enable differentiation later on
C = F.T * F
Ic = ufl.tr(C)
IIc = 0.5 * (Ic**2 - ufl.tr(C * C))
J = ufl.det(F)

psi_iso = c1 * (Ic - 3.0) + c2 * (IIc - 3.0)
psi_vol = 0.5 * kappa * (J - 1.0)**2
psi = psi_iso + psi_vol

dx = ufl.Measure("dx", domain=domain)

# -----------------------------------------------------------
# Boundary conditions and loading
# -----------------------------------------------------------
def inner_boundary(x):
    return np.isclose(x[0], Ri)

def outer_boundary(x):
    return np.isclose(x[0], Ro)

inner_facets = mesh.locate_entities_boundary(domain, fdim, inner_boundary)
outer_facets = mesh.locate_entities_boundary(domain, fdim, outer_boundary)

facet_indices = np.concatenate([inner_facets, outer_facets])
facet_markers = np.concatenate([
    np.full(inner_facets.shape, 1, dtype=np.int32),  # id=1 -> inner
    np.full(outer_facets.shape, 2, dtype=np.int32),  # id=2 -> outer
])

mt = mesh.meshtags(domain, fdim, facet_indices, facet_markers)

ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)
n = ufl.FacetNormal(domain)

# Displacement BC: clamp u = 0 on outer boundary (x = Ro)
zero_vec = fem.Function(V)  # defaults to 0
dofs_outer = fem.locate_dofs_topological(V, fdim, outer_facets)
bc_outer = fem.dirichletbc(zero_vec, dofs_outer)
bcs = [bc_outer]

# Pressure load on inner boundary (x = Ri)
p_in_val = 25.0e3  # 25 kPa
p_in = fem.Constant(domain, Scalar(p_in_val))

# Traction work term: ∫_{inner} p * n · v ds
# n is outward normal from the solid. On the inner boundary it points into the lumen,
# so -p*n gives pressure pushing outward on the wall.
external_work = - p_in * ufl.inner(n, v) * ds(1)

# -----------------------------------------------------------
# Weak form: total potential = strain energy - external work
# -----------------------------------------------------------
Pi = psi * dx + external_work
F_res = ufl.derivative(Pi, u, v)
J_form = ufl.derivative(F_res, u, du)

# -----------------------------------------------------------
# Nonlinear solve (same pattern as your poro script)
# -----------------------------------------------------------
problem = NewtonSolverNonlinearProblem(F_res, u, bcs=bcs, J=J_form)
solver = NewtonSolver(MPI.COMM_WORLD, problem)

solver.rtol = 1e-8
solver.atol = 1e-9
solver.max_it = 25
solver.convergence_criterion = "residual"

ksp = solver.krylov_solver
ksp.setType("gmres")
pc = ksp.getPC()
pc.setType("lu")

num_its, converged = solver.solve(u)
if domain.comm.rank == 0:
    print(f"Newton iterations: {num_its}, converged = {converged}")


# -----------------------------------------------------------
# Compute displacement components
# -----------------------------------------------------------
V_out = fem.functionspace(domain, ("Lagrange", 1, (dim,)))
u_out = fem.Function(V_out, name="u")
u_out.interpolate(u)
u_out_arr = u_out.x.array.reshape((-1,dim)) # reshape to NxD array
ux = u_out_arr[:,0]  # x-displacement
uy = u_out_arr[:,1]  # y-displacement


# -----------------------------------------------------------
# Compute and output stress components and von Mises stress
# -----------------------------------------------------------
P = ufl.diff(psi, F)  # First Piola-Kirchhoff stress
sigma = (1/J) * P * F.T  # Cauchy stress

# Project stress components to a suitable function space
# Use DG-0 (piecewise constant) for stress which is naturally discontinuous
S_space = fem.functionspace(domain, ("DG", 0))
sigma_xx = fem.Function(S_space, name="sigma_xx")
sigma_yy = fem.Function(S_space, name="sigma_yy")
sigma_xy = fem.Function(S_space, name="sigma_xy")

# Create expressions for stress components
sigma_xx_expr = fem.Expression(sigma[0,0], S_space.element.interpolation_points)
sigma_yy_expr = fem.Expression(sigma[1,1], S_space.element.interpolation_points)
sigma_xy_expr = fem.Expression(sigma[0,1], S_space.element.interpolation_points)

# Interpolate stress components
sigma_xx.interpolate(sigma_xx_expr)
sigma_yy.interpolate(sigma_yy_expr)
sigma_xy.interpolate(sigma_xy_expr)

# Compute von Mises stress
von_mises = ufl.sqrt(sigma_xx**2 + sigma_yy**2 - sigma_xx*sigma_yy + 3*sigma_xy**2)
sigma_vm = fem.Function(S_space, name="von_Mises")
vm_expr = fem.Expression(von_mises, S_space.element.interpolation_points)
sigma_vm.interpolate(vm_expr)


# Plotting with pyvista
topology, cell_types, x = vtk_mesh(V_out)
grid = pv.UnstructuredGrid(topology, cell_types, x)
grid.point_data["ux"] = ux  # plot x-displacement as scalar field
grid.point_data["uy"] = uy  # plot y-displacement as scalar field
grid.point_data["u"] = np.sqrt(ux**2 + uy**2)  # plot magnitude of displacement
grid.cell_data["sigma_xx"] = sigma_xx.x.array
grid.cell_data["sigma_yy"] = sigma_yy.x.array
grid.cell_data["sigma_xy"] = sigma_xy.x.array
grid.cell_data["von_Mises"] = sigma_vm.x.array


plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True, scalars="von_Mises", cmap="viridis")
plotter.show()

b=0
# nodes = domain.geometry.x
# cells = domain.geometry.dofmap

# plot_scalar(V_out, u_out)

# import pyvista as pv
# mesh_data = pv.UnstructuredGrid(cells, np.full(cells.shape[0], pv.CellType.TRIANGLE), nodes)
# mesh_data.cell_data["values"] = np.arange(cells.shape[0])
# plotter = pv.Plotter()
# plotter.add_mesh(mesh_data, show_edges=True, color="lightgrey", opacity=0.5)
# plotter.add_mesh(pv.PolyData(nodes + u_out.x.array.reshape((-1, dim))), color="red", opacity=0.7)
# plotter.show()

# go.Figure(data=[go.Mesh3d(
#     x=nodes[:, 0],
#     y=nodes[:, 1],
#     z=nodes[:, 2],
#     i=cells[:, 0],
#     j=cells[:, 1],
#     k=cells[:, 2],
#     opacity=1.0,
# )]).show()

with io.XDMFFile(domain.comm, "mooney_tube_results.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_out)

if domain.comm.rank == 0:
    print("Done. Results written to mooney_tube_results.xdmf")

from dolfinx.io import VTKFile

dim = domain.geometry.dim   # should be 2 for your Mooney–Rivlin demo

# Create a proper vector-valued function space (P1 vector)
V_out = fem.functionspace(domain, ("Lagrange", 1, (dim,)))
u_out = fem.Function(V_out, name="u")

# Interpolate the P2 displacement u to P1
u_out.interpolate(u)

# Write as vector point data
with VTKFile(domain.comm, "mooney_tube_results.vtu", "w") as f:
    f.write_function(u_out)
