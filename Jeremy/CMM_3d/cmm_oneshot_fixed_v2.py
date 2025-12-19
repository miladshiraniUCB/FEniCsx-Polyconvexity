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

# PETSc options are set through petsc4py (recommended)
from petsc4py import PETSc
import pyvista as pv
from dolfinx.plot import vtk_mesh


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _normalize(v, eps=1e-12):
    """UFL-safe normalization with a small epsilon."""
    return v / ufl.sqrt(ufl.dot(v, v) + eps)


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


def read_cell_vector_fields_from_xdmf_h5(fiber_xdmf, comm):
    """
    Read meshio-written cell-data vectors (shape [num_cells_global, 3]) from the .h5 referenced by the XDMF.

    This mimics your original parsing approach, but we keep it here as a compact utility.
    """
    import re
    import h5py
    fiber_data = {}

    h5_path = fiber_xdmf.replace(".xdmf", ".h5")
    with open(fiber_xdmf, "r") as f:
        xdmf_txt = f.read()

    # match Attribute Name="fiber_<something>"
    fiber_names = re.findall(r'<Attribute Name="(fiber_\w+)"', xdmf_txt)
    if comm.rank == 0:
        print(f"Found fiber/basis attributes in {fiber_xdmf}: {fiber_names}")

    with h5py.File(h5_path, "r") as h5:
        for name in fiber_names:
            # find the DataItem that points to the dataset
            # example: tube_mesh_fibers.h5:/data0
            pattern = rf'<Attribute Name="{name}".*?>(.*?)</Attribute>'
            attr_match = re.search(pattern, xdmf_txt, re.DOTALL)
            if attr_match is None:
                continue
            data_match = re.search(r'>([^<]+\.h5:[^<]+)</DataItem>', attr_match.group(1))
            if data_match is None:
                continue
            dataset_path = data_match.group(1).split(":")[1]
            arr = np.array(h5[dataset_path])
            fiber_data[name] = arr
            if comm.rank == 0:
                print(f"  Loaded {name}: {arr.shape}")
    return fiber_data


def build_DG0_vector_functions(domain, fiber_data):
    """
    Map global cell-data arrays -> DG0 vector Functions (one vector per cell).
    Includes ghost cells and scatter_forward() for parallel correctness.  (FIX 6)
    """
    comm = domain.comm
    rank = comm.rank
    dim = domain.geometry.dim

    DG0v = fem.functionspace(domain, ("DG", 0, (dim,)))
    cell_imap = domain.topology.index_map(domain.topology.dim)

    n_local = cell_imap.size_local
    n_ghost = cell_imap.num_ghosts

    # IMPORTANT: include ghosts for correct evaluation/assembly on shared facets
    local_cells = np.arange(n_local + n_ghost, dtype=np.int32)
    global_cells = cell_imap.local_to_global(local_cells)

    fiber_functions = {}
    for full_name, global_arr in fiber_data.items():
        short = full_name.replace("fiber_", "")
        f = fem.Function(DG0v, name=full_name)

        # Subset the global array by the (owned+ghost) global cell ids
        local_arr = global_arr[global_cells]  # shape (n_local+n_ghost, 3)

        # DG0 vector dof layout is flat [x0,y0,z0, x1,y1,z1, ...]
        f.x.array[: 3 * (n_local + n_ghost)] = local_arr.astype(np.float64).reshape(-1)
        f.x.scatter_forward()

        fiber_functions[short] = f
        if rank == 0:
            print(f"Created DG0 vector field: '{short}'")
    return fiber_functions


def constituent_C_elastic(C, Ginv):
    """C^alpha = G^{-T} C G^{-1}. Here Ginv is symmetric, so G^{-T}=G^{-1}=Ginv."""
    return Ginv * C * Ginv


# -----------------------------------------------------------------------------
# Load mesh + tags + fibers
# -----------------------------------------------------------------------------

comm = MPI.COMM_WORLD
rank = comm.rank

mesh_file = "tube_mesh.xdmf"
with io.XDMFFile(comm, mesh_file, "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid", ghost_mode=mesh.GhostMode.shared_facet)

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

# Boundary facet tags
facet_tags = None
boundary_file = "tube_mesh_boundary.xdmf"
try:
    with io.XDMFFile(comm, boundary_file, "r") as xdmf:
        facet_tags = xdmf.read_meshtags(domain, name="Grid")
    if rank == 0:
        print(f"Loaded facet tags from {boundary_file}; unique tags: {np.unique(facet_tags.values)}")
except Exception as e:
    if rank == 0:
        print(f"WARNING: Could not read boundary tags: {e}")
        print("         Proceeding without tags will fail unless you add geometric markers.")

# Fiber/basis data (DG0 vectors on cells)
fiber_functions = {}
fiber_file = "tube_mesh_fibers.xdmf"
try:
    fiber_data = read_cell_vector_fields_from_xdmf_h5(fiber_file, comm)
    fiber_functions = build_DG0_vector_functions(domain, fiber_data)
except Exception as e:
    if rank == 0:
        print(f"WARNING: Could not read fiber data: {e}")
        print("         Will use analytic cylindrical fallback directions.")


# -----------------------------------------------------------------------------
# Function spaces and kinematics
# -----------------------------------------------------------------------------

dim = domain.geometry.dim
elem_order = domain.geometry.cmap.degree

V = fem.functionspace(domain, ("Lagrange", elem_order, (dim,)))
u = fem.Function(V, name="u")
v = ufl.TestFunction(V)

I = ufl.Identity(dim)
F = I + ufl.grad(u)
C = F.T * F
J = ufl.det(F)


# -----------------------------------------------------------------------------
# Material parameters (Table 1, kPa -> N/mm^2)
# -----------------------------------------------------------------------------
# FIX 1: convert from kPa directly (do NOT multiply by 1e3 then 1e-3)

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
c1m = fem.Constant(domain, PETSc.ScalarType(6.939 * kPa_to_N_per_mm2))
c2m = fem.Constant(domain, PETSc.ScalarType(4.442))

c1c = fem.Constant(domain, PETSc.ScalarType(14.04 * kPa_to_N_per_mm2))
c2c = fem.Constant(domain, PETSc.ScalarType(0.6530))

# Deposition stretches (Table 1)
Gm = fem.Constant(domain, PETSc.ScalarType(1.080))
Gc = fem.Constant(domain, PETSc.ScalarType(1.130))
Ge_theta = fem.Constant(domain, PETSc.ScalarType(1.100))
Ge_z     = fem.Constant(domain, PETSc.ScalarType(1.720))
Ge_r     = fem.Constant(domain, PETSc.ScalarType(1.0 / (float(Ge_theta.value) * float(Ge_z.value))))
# NOTE: For volumetric-preserving elastin prestretch you want Ge_r*Ge_theta*Ge_z = 1.
# If you want exactly that, keep Ge_r = 1/(Ge_theta*Ge_z). If you prefer the paper's implied form,
# you can set Ge_r = 1.0 and accept a small volume change in G (not recommended).

# FIX 5: bulk-like penalty: kappa = 1e3*c_e (paper)
kappa = fem.Constant(domain, PETSc.ScalarType(1e3 * float(c_e.value)))

# Pre-Jacobian shift JG = exp(-p0/kappa) (paper Stage I)
JG = fem.Constant(domain, PETSc.ScalarType(np.exp(-float(p0.value) / float(kappa.value))))


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
Ge_inv = (ufl.outer(e_r, e_r) / Ge_r +
          ufl.outer(e_theta, e_theta) / Ge_theta +
          ufl.outer(e_z, e_z) / Ge_z)

# SMC + collagen prestretch inverse tensors (isotropic)
Gm_inv = (1.0 / Gm) * I
Gc_inv = (1.0 / Gc) * I

Ce = constituent_C_elastic(C, Ge_inv)
Cm = constituent_C_elastic(C, Gm_inv)
Cc = constituent_C_elastic(C, Gc_inv)


# -----------------------------------------------------------------------------
# Strain energy density (Stage I)
# -----------------------------------------------------------------------------

def W_elastin(Ce_):
    # Eq (51): W^e = c_e/2 (Ce:I - 3) 
    I1e = ufl.tr(Ce_)
    return 0.5 * c_e * (I1e - 3.0)


def W_fiber(I4, c1, c2):
    """
    Eq (52)-type exponential fiber energy (tension-only).
    The paper defines smooth muscle + collagen with an exponential on (I4-1)^2; we enforce tension-only.
    """
    E = I4 - 1.0
    # tension-only: no contribution in compression
    return ufl.conditional(ufl.gt(I4, 1.0), (c1 / (4.0 * c2)) * (ufl.exp(c2 * E * E) - 1.0), 0.0)


# Fiber invariants using constituent elastic C-tensors
I4m = ufl.dot(a_theta, Cm * a_theta)     # SMC circumferential
I4ct = ufl.dot(a_theta, Cc * a_theta)    # collagen circumferential
I4cz = ufl.dot(a_axial, Cc * a_axial)    # collagen axial
I4cd1 = ufl.dot(a_diag1, Cc * a_diag1)   # collagen diag +
I4cd2 = ufl.dot(a_diag2, Cc * a_diag2)   # collagen diag -

Wm = W_fiber(I4m, c1m, c2m)

# collagen contribution (βd split equally between the two diagonal families)
Wc = (beta_theta * W_fiber(I4ct, c1c, c2c) +
      beta_z     * W_fiber(I4cz, c1c, c2c) +
      0.5 * beta_d * (W_fiber(I4cd1, c1c, c2c) + W_fiber(I4cd2, c1c, c2c)))

# FIX 5: volumetric penalty uses ln(J*JG)
lnJ = ufl.ln(J * JG)
U_vol = 0.5 * kappa * lnJ * lnJ

W_total = phi_e0 * W_elastin(Ce) + phi_m0 * Wm + phi_c0 * Wc + U_vol


# -----------------------------------------------------------------------------
# Weak form: residual and Jacobian
# -----------------------------------------------------------------------------

dx = ufl.Measure("dx", domain=domain)

if facet_tags is None:
    raise RuntimeError("Facet tags are required (tube_mesh_boundary.xdmf).")

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)


# -----------------------------------------------------------------------------
# Surface tag diagnostics + auto-detect inner/outer cylindrical wall tags
# -----------------------------------------------------------------------------
# Your tube generator intends: inner=1, outer=2, end1=3, end2=4.
# In practice, XDMF facet tags can get permuted depending on how facet data
# is written/read. A swapped/misaligned tag is the #1 cause of the "tube bends like
# a cantilever" artifact, because pressure ends up acting on a *non-axisymmetric*
# subset of facets.
#
# We therefore compute, for every boundary tag:
#   - surface area A(tag)
#   - mean radius <r> (r = sqrt(x^2+y^2))
#   - mean |n·e_z|  (≈1 for end caps, ≈0 for cylindrical walls)
#
# Then we classify "cylindrical" tags as those with mean|n·e_z| < 0.3 and pick:
#   inner_tag = min <r> among cylindrical tags
#   outer_tag = max <r> among cylindrical tags
#
# This makes the code robust to tag permutations and will immediately reveal
# whether pressure is being applied to the correct surface.

tags = np.unique(facet_tags.values)
areas = {}
mean_r = {}
mean_abs_nz = {}

n = ufl.FacetNormal(domain)

X = ufl.SpatialCoordinate(domain)
r = ufl.sqrt(X[0] * X[0] + X[1] * X[1] + 1e-16)
e_z = ufl.as_vector((PETSc.ScalarType(0.0), PETSc.ScalarType(0.0), PETSc.ScalarType(1.0)))
abs_nz = ufl.sqrt(ufl.dot(n, e_z) ** 2)  # = |n·e_z|

for tag in tags:
    tag_i = int(tag)
    A = fem.assemble_scalar(fem.form(PETSc.ScalarType(1.0) * ds(tag_i)))
    A = comm.allreduce(A, op=MPI.SUM)
    areas[tag_i] = float(A)

    if A > 1e-14:
        mr = fem.assemble_scalar(fem.form(r * ds(tag_i)))
        mr = comm.allreduce(mr, op=MPI.SUM) / A
        mean_r[tag_i] = float(mr)

        mnz = fem.assemble_scalar(fem.form(abs_nz * ds(tag_i)))
        mnz = comm.allreduce(mnz, op=MPI.SUM) / A
        mean_abs_nz[tag_i] = float(mnz)

if rank == 0:
    print("--- Boundary tag diagnostics (area, mean radius, mean|n·e_z|) ---")
    for tag_i in sorted(areas.keys()):
        print(
            f" tag {tag_i:>2d}:  A={areas[tag_i]:.6e},  "
            f"<r>={mean_r.get(tag_i, float('nan')):.6e},  "
            f"<|n·e_z|>={mean_abs_nz.get(tag_i, float('nan')):.6e}"
        )

# Identify cylindrical wall tags (radial normals => |n·e_z| ~ 0)
cyl_tags = [t for t in sorted(areas.keys()) if mean_abs_nz.get(t, 1.0) < 0.3]

if len(cyl_tags) >= 2:
    inner_tag = min(cyl_tags, key=lambda t: mean_r[t])
    outer_tag = max(cyl_tags, key=lambda t: mean_r[t])
else:
    # Fallback to your intended tags (inner=1, outer=2)
    inner_tag, outer_tag = 1, 2

if rank == 0:
    print(f"Detected inner_tag={inner_tag} and outer_tag={outer_tag} (cyl_tags={cyl_tags})")


# FIX 2: traction = -P0*n on the detected inner cylindrical surface (inner_tag)
traction_inner = -P0 * n
R = ufl.derivative(W_total * dx, u, v) - ufl.dot(traction_inner, v) * ds(inner_tag)


# Diagnostic: net pressure force (should be ~0 for a straight, fully-tagged inner cylinder)
# If this is NOT ~0, it strongly indicates:
#   (a) the wrong surface tag is being loaded, OR
#   (b) only part of the inner surface is tagged with inner_tag.
Fnet = np.zeros(3)
for i in range(3):
    e_i = ufl.as_vector([1.0 if j == i else 0.0 for j in range(3)])
    F_i = fem.assemble_scalar(fem.form(ufl.dot(traction_inner, e_i) * ds(inner_tag)))
    Fnet[i] = comm.allreduce(F_i, op=MPI.SUM)
if rank == 0:
    print(f"Net pressure force on inner_tag {inner_tag}: [{Fnet[0]:.6e}, {Fnet[1]:.6e}, {Fnet[2]:.6e}] (N)")

J_form = ufl.derivative(R, u, ufl.TrialFunction(V))


# -----------------------------------------------------------------------------
# Boundary conditions
# -----------------------------------------------------------------------------

# FIX 3: only constrain axial displacement (u_z) on end caps (tags 3 and 4), not full vector.
# Also add minimal constraints on one end to prevent rigid-body translation in x/y.

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
    
}


# -----------------------------------------------------------------------------
# Solve nonlinear problem (Stage I)
# -----------------------------------------------------------------------------
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