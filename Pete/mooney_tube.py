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

F = deformation_gradient(u)
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
    log.log(log.LogLevel.INFO,
            f"Newton iterations: {num_its}, converged = {converged}")

# -----------------------------------------------------------
# Output to XDMF (interpolate to P1 to match mesh degree)
# -----------------------------------------------------------
V_out = fem.functionspace(domain, ("Lagrange", 1, (dim,)))
u_out = fem.Function(V_out, name="u")
u_out.interpolate(u)

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
