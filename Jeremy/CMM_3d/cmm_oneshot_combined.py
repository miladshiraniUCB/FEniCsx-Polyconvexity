from mpi4py import MPI
import numpy as np
from dolfinx import mesh, fem, io
import ufl
from dolfinx.fem.petsc import NonlinearProblem, NewtonSolverNonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.plot import vtk_mesh
import pyvista as pv
from petsc4py import PETSc

from time import time

# -------------------------------------------------------------------
# 0. Basic setup
# -------------------------------------------------------------------

comm = MPI.COMM_WORLD
rank = comm.rank

# Load mesh from XDMF file
mesh_file = "tube_mesh.xdmf"
with io.XDMFFile(comm, mesh_file, "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid", ghost_mode=mesh.GhostMode.shared_facet)

    # Create facet connectivity
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

    # Get number of cells (elements) in the mesh
    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    num_cells_global = domain.topology.index_map(domain.topology.dim).size_global
    
    # Gather cell counts from all ranks for load balancing check
    all_cell_counts = comm.gather(num_cells, root=0)
    
    if rank == 0:
        print(f"Mesh loaded and partitioned across {comm.size} processes")
        print(f"Number of elements (global): {num_cells_global}")
        
        # Determine mesh type
        cell_type = domain.topology.cell_type
        cell_name = domain.topology.cell_name()
        print(f"Mesh cell type: {cell_name}")
        
        # Check MPI usage
        if comm.size == 1:
            print(f"WARNING: Running in serial mode (only 1 MPI process)")
            print(f"  To use MPI parallelization, run with: mpirun -n <num_processes> python cmm_oneshot.py")

# Determine element order from the mesh geometry
# The mesh geometry degree indicates whether the mesh uses linear or higher-order elements
elem_order = domain.geometry.cmap.degree

if rank == 0:
    element_type = "quadratic" if elem_order == 2 else ("linear" if elem_order == 1 else f"order-{elem_order}")
    print(f"\nMesh geometry information:")
    print(f"  Rank {rank}: Number of elements (local): {num_cells}")
    print(f"  Element interpolation: {element_type} (degree {elem_order})")
    print(f"  Nodes per cell: {domain.geometry.cmap.dim}")
    print(f"  Total nodes (global): {domain.geometry.x.shape[0]}")
    
    if comm.size > 1:
        print(f"\nLoad balancing:")
        print(f"  Elements per rank: {all_cell_counts}")
        print(f"  Min: {min(all_cell_counts)}, Max: {max(all_cell_counts)}, " 
          f"Avg: {sum(all_cell_counts)/len(all_cell_counts):.1f}")
        imbalance = (max(all_cell_counts) - min(all_cell_counts)) / (sum(all_cell_counts)/len(all_cell_counts))
        print(f"  Load imbalance: {imbalance*100:.1f}%")

# Read fiber families from separate fiber file
# Fiber families are stored as cell data (DG0) in the XDMF file
# For parallel execution, we only load the fiber data for local cells
fiber_file = "tube_mesh_fibers.xdmf"
fiber_data = {}
try:
    import h5py
    import re
    
    # Read fiber data from HDF5 file
    # Note: In parallel, each rank reads the full array and will extract its local portion
    with h5py.File(fiber_file.replace('.xdmf', '.h5'), 'r') as h5:
        # Scan XDMF to find fiber attributes
        with open(fiber_file, 'r') as xdmf_text:
            xdmf_content = xdmf_text.read()
            # Look for fiber attributes
            fiber_matches = re.findall(r'<Attribute Name="(fiber_\w+)"', xdmf_content)
            
            if rank == 0:
                print(f"Found fiber families in {fiber_file}: {fiber_matches}")
            
            # Read each fiber family
            # In parallel, we read the full dataset and will subset it based on cell ownership
            for fiber_name in fiber_matches:
                # Find the data path for this fiber
                pattern = f'<Attribute Name="{fiber_name}".*?>(.*?)</Attribute>'
                attr_match = re.search(pattern, xdmf_content, re.DOTALL)
                if attr_match:
                    data_match = re.search(r'>([^<]+\.h5:[^<]+)</DataItem>', attr_match.group(1))
                    if data_match:
                        data_path = data_match.group(1).split(':')[1]
                        # Read full fiber array (all processes read the same data)
                        full_fiber_data = np.array(h5[data_path])
                        
                        # Store for now - will subset to local cells when creating functions
                        fiber_data[fiber_name] = full_fiber_data
                        
                        if rank == 0:
                            print(f"  Loaded {fiber_name}: shape {full_fiber_data.shape}")
except Exception as e:
    if rank == 0:
        print(f"Warning: Could not read fiber data from {fiber_file}: {e}")
        print("Using constant fiber directions as fallback.")

# Read boundary facets from separate boundary file
boundary_file = "tube_mesh_boundary.xdmf"
try:
    with io.XDMFFile(comm, boundary_file, "r") as xdmf:
        # Read the facet tags (boundary markers)
        facet_tags = xdmf.read_meshtags(domain, name="Grid")
        if rank == 0:
            print(f"Loaded boundary facets from {boundary_file}")
            print(f"  Facet tags shape (local): {facet_tags.indices.shape}")
            print(f"  Unique tags: {np.unique(facet_tags.values)}")
except Exception as e:
    if rank == 0:
        print(f"Warning: Could not read boundary data from {boundary_file}: {e}")
    facet_tags = None

dim = domain.topology.dim

# -------------------------------------------------------------------
# Compute explicit normal vectors for pressure boundary (tag 1)
# -------------------------------------------------------------------
if facet_tags is not None and rank == 0:
    print("\n=== Computing explicit normals for pressure boundary ===")

# For cylindrical tube, inner surface normals point radially inward
# We'll compute them explicitly based on geometry
# Normal at point (x, y, z) on inner surface: n = -[x, y, 0] / sqrt(x² + y²)

# Create a DG function space on facets to store normals
# We'll use a different approach: define pressure direction analytically
# For a tube along z-axis, pressure direction at (x,y,z) is radially outward: [x, y, 0]/r

# Define the pressure traction direction as a UFL expression
x_ufl = ufl.SpatialCoordinate(domain)
r_xy = ufl.sqrt(x_ufl[0]**2 + x_ufl[1]**2)
# Outward radial direction (for internal pressure to expand the tube)
pressure_direction = ufl.as_vector([x_ufl[0]/r_xy, x_ufl[1]/r_xy, 0.0])

if rank == 0:
    print("Pressure direction defined as radially outward: [x/r, y/r, 0]")
    print("This will cause the tube to expand under internal pressure")

n = ufl.FacetNormal(domain)

# -------------------------------------------------------------------
# Visualize UFL facet normals
# -------------------------------------------------------------------
if facet_tags is not None and rank == 0:
    print("\n=== Visualizing UFL facet normals ===")
    
    # Get mesh topology
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Create connectivity
    domain.topology.create_connectivity(fdim, tdim)
    
    # Get geometry
    geom = domain.geometry
    x = geom.x
    
    # Get facets with tag 1 (inner surface)
    inner_facets = facet_tags.indices[facet_tags.values == 1]
    
    if len(inner_facets) > 0:
        # Compute normals manually from geometry instead of using UFL FacetNormal
        # which is not supported in cell-based Expression interpolation
        
        # Get facet-to-vertex connectivity for visualization
        facet_to_vertex = domain.topology.connectivity(fdim, 0)
        
        # Compute facet centers
        facet_centers = []
        for facet_idx in inner_facets[:50]:  # Visualize first 50 facets
            vertices = facet_to_vertex.links(facet_idx)
        # Get facet-to-vertex connectivity for visualization
        facet_to_vertex = domain.topology.connectivity(fdim, 0)
        
        # Compute facet centers and normals geometrically
        facet_centers = []
        facet_normals_ufl = []
        for facet_idx in inner_facets[:50]:  # Visualize first 50 facets
            vertices = facet_to_vertex.links(facet_idx)
            coords = x[vertices]
            center = np.mean(coords, axis=0)
            facet_centers.append(center)
            
            # Compute normal geometrically
            if len(vertices) >= 4:
                v1 = coords[1] - coords[0]
                v2 = coords[2] - coords[0]
                normal = np.cross(v1, v2)
                normal = normal / np.linalg.norm(normal)
                facet_normals_ufl.append(normal)
            elif len(vertices) == 3:
                v1 = coords[1] - coords[0]
                v2 = coords[2] - coords[0]
                normal = np.cross(v1, v2)
                normal = normal / np.linalg.norm(normal)
                facet_normals_ufl.append(normal)
        
        facet_centers = np.array(facet_centers)
        facet_normals_ufl = np.array(facet_normals_ufl)
        
        # Create visualization
        plotter = pv.Plotter()
        
        # Plot mesh
        topology_viz, cell_types_viz, x_viz = vtk_mesh(domain)
        grid = pv.UnstructuredGrid(topology_viz, cell_types_viz, x_viz)
        plotter.add_mesh(grid.extract_surface(), show_edges=True, opacity=0.3, color='lightgray')
        
        # Plot UFL normal vectors as arrows
        if len(facet_centers) > 0:
            scale_factor = 0.2
            arrows = pv.PolyData(facet_centers)
            arrows['vectors'] = facet_normals_ufl * scale_factor
            
            plotter.add_mesh(arrows.glyph(orient='vectors', scale=False, factor=1.0), 
                           color='blue', label='UFL Facet Normals (Tag 1: Inner Surface)')
            
            print(f"Plotted {len(facet_centers)} UFL facet normals for inner surface (tag 1)")
            print(f"Example UFL normals:\n{facet_normals_ufl[:3]}")
        
        plotter.add_legend()
        # plotter.show()



# Create function space with element order matching the mesh geometry
# For hex8/tet4: elem_order = 1 (linear)
# For hex20/tet10: elem_order = 2 (quadratic)
V = fem.functionspace(domain, ("Lagrange", elem_order, (dim,)))

if rank == 0:
    print(f"\nFunction space created:")
    print(f"  Element family: Lagrange")
    print(f"  Polynomial degree: {elem_order}")
    print(f"  Vector dimension: {dim}")
u = fem.Function(V, name="u")
v = ufl.TestFunction(V)
du = ufl.TrialFunction(V)

I = ufl.Identity(dim)

# Create DG0 vector function space for fiber directions (piecewise constant per element)
DG0_vector = fem.functionspace(domain, ("DG", 0, (dim,)))

# Create fiber function objects from loaded data
# In parallel, we need to map the fiber data to local cells
fiber_functions = {}

# Get the cell map to understand the global-to-local cell indexing
cell_imap = domain.topology.index_map(domain.topology.dim)
num_cells_local = cell_imap.size_local
cells_global = cell_imap.local_to_global(np.arange(num_cells_local, dtype=np.int32))

for fiber_name, fiber_array in fiber_data.items():
    # Remove 'fiber_' prefix for cleaner names
    short_name = fiber_name.replace('fiber_', '')
    fiber_func = fem.Function(DG0_vector, name=fiber_name)
    
    # Extract fiber data for local cells only
    # fiber_array has shape (num_elements_global, 3) from mesh file
    # We need to extract only the rows corresponding to local cells
    local_fiber_data = fiber_array[cells_global]
    
    # Assign fiber data to function
    # DG0 vector function expects data in flattened form: [x0, y0, z0, x1, y1, z1, ...]
    # which is C-order (row-major) flattening
    fiber_func.x.array[:] = local_fiber_data.flatten()
    fiber_functions[short_name] = fiber_func
    
    if rank == 0:
        print(f"Created fiber function '{short_name}'")
        print(f"  Global elements: {len(fiber_array)}, Local to rank 0: {len(local_fiber_data)}")
        # # Diagnostic: check first few fiber vectors
        # print(f"  First 3 fiber vectors from file (global indices {cells_global[:3]}):")
        # for i in range(min(3, len(local_fiber_data))):
        #     print(f"    Element {cells_global[i]}: {local_fiber_data[i]}")
        # # Verify what was assigned
        # print(f"  First 9 values in DOF array: {fiber_func.x.array[:9]}")
        # # Reshape back to verify
        # test_reshape = fiber_func.x.array.reshape(-1, 3)
        # print(f"  First 3 vectors after reshape:")
        # for i in range(min(3, len(test_reshape))):
        #     print(f"    Element {i} (local): {test_reshape[i]}")


# # Plotting with pyvista
# topology, cell_types, x = vtk_mesh(V)
# grid = pv.UnstructuredGrid(topology, cell_types, x)

# plotter = pv.Plotter()
# plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
# plotter.show()


# -------------------------------------------------------------------
# 1. Material parameters (from Table 1 in paper, scaled to your units)
#    You will probably read these from JSON or a config in practice.
#    NOTE: Mesh is in [mm], so all stress units must be in [N/mm²] = [MPa]
#          1 Pa = 0.001 N/mm², 1 kPa = 0.001 MPa
# -------------------------------------------------------------------

kpa_to_N_per_mm2 = 1e-3  # Conversion factor from kPa to N/mm²

# Pressure on "top" face
# Replace with pressure consistent with your geometry / Po in paper
# Original: 14 kPa = 14,000 Pa → Convert to N/mm²
P0 = 14 * kpa_to_N_per_mm2  # [N/mm²] = 14 Pa (converted from 14 kPa)
p0 = 10.21 * kpa_to_N_per_mm2  # [N/mm²] = 10.21 Pa (converted from 10.21 kPa)

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
# Original: 89.71 kPa → Convert to N/mm²
c_e = 89.71 * kpa_to_N_per_mm2  # [N/mm²] = 89.71 N/mm² (converted from 89.71 kPa)

# Smooth muscle parameters
# Original: 261.4 kPa → Convert to N/mm²
c_m1 = 261.4 * kpa_to_N_per_mm2  # [N/mm²] = 261.4 N/mm² (converted from 261.4 kPa)
c_m2 = 0.24  # dimensionless

# Collagen parameters
# Original: 234.9 kPa → Convert to N/mm²
c_c1 = 234.9 * kpa_to_N_per_mm2  # [N/mm²] = 234.9 N/mm² (converted from 234.9 kPa)
c_c2 = 4.08  # dimensionless

# Deposition stretches
G_e_theta = 1.9 # Circumferential deposition stretch
G_e_z = 1.62 # Axial deposition stretch
G_e_r = 1.0 / (G_e_theta * G_e_z)

G_m = 1.20 # Smooth muscle deposition stretch (isotropic)
G_c = 1.25 # Collagen deposition stretch (isotropic)

# Penalty bulk modulus for Stage I
# kappa = 1e3 * c_e # Might be too stiff
# kappa = 1e2 * c_e
kappa = 1e1 * c_e
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
    I4 = a0 · C a0 for a unit vector a0.
    
    Parameters
    ----------
    C : ufl.Expr
        Right Cauchy-Green deformation tensor
    a0 : numpy.ndarray or fem.Function
        Fiber direction (3D vector or DG0 vector function)
    """
    # Handle both constant arrays and spatially varying functions
    if isinstance(a0, np.ndarray):
        a = ufl.as_vector(a0)
    else:
        # a0 is a fem.Function (DG0 vector field)
        a = a0
    return ufl.dot(a, C * a)

# -------------------------------------------------------------------
# 3. Four-fiber strain energy pieces (per reference volume)
#    NOTE: We implement them as hyperelastic parts suitable for UFL AD.
# -------------------------------------------------------------------

def W_elastin(C, e_r=None, e_theta=None, e_z=None):
    """
    Elastin strain energy with anisotropic deposition stretches.
    
    Parameters
    ----------
    C : ufl.Expr
        Right Cauchy-Green deformation tensor
    e_r : fem.Function, optional
        Radial basis vector field
    e_theta : fem.Function, optional
        Circumferential basis vector field
    e_z : fem.Function, optional
        Axial basis vector field
    """
    if e_r is None or e_theta is None or e_z is None:
        # Fallback to isotropic deposition stretch
        G_e_iso = (G_e_r * G_e_theta * G_e_z)**(1.0/3.0)
        G = ufl.as_matrix(np.diag([G_e_iso, G_e_iso, G_e_iso]))
        C_e = G * C * G
        I1_e = ufl.tr(C_e)
        return 0.5 * c_e * (I1_e - 3.0)
    else:
        # Construct deposition stretch tensor in local cylindrical coordinates
        # According to eq. (68) and the multiplicative decomposition F = F^e · G_e
        # We have: F^e = F · G_e^(-1), therefore C^e = G_e^(-T) · C · G_e^(-1)
        # Where G_e is diagonal in principal directions: G_e = diag(G_e_r, G_e_theta, G_e_z)
        
        # Build the INVERSE deposition stretch tensor G_e^(-1)
        # Since G_e is diagonal in the basis {e_r, e_θ, e_z}, its inverse is:
        # G_e^(-1) = (1/G_e_r) e_r⊗e_r + (1/G_e_θ) e_θ⊗e_θ + (1/G_e_z) e_z⊗e_z
        
        # Old version
        # G_e_inv = ((1.0/G_e_r) * ufl.outer(e_r, e_r) + 
        #            (1.0/G_e_theta) * ufl.outer(e_theta, e_theta) + 
        #            (1.0/G_e_z) * ufl.outer(e_z, e_z))
        # # C^e = G_e^(-T) · C · G_e^(-1) (eq. 68)
        # # Since G_e is symmetric (diagonal), G_e^(-T) = G_e^(-1)
        # C_e = G_e_inv * C * G_e_inv
        # I1_e = ufl.tr(C_e)
        # return 0.5 * c_e * (I1_e - 3.0) # eq. (51)

        # New version
        G_e_inv = (ufl.outer(e_r, e_r) / G_e_r +
                   ufl.outer(e_theta, e_theta) / G_e_theta +
                   ufl.outer(e_z, e_z) / G_e_z)
        C_e = G_e_inv * C * G_e_inv
        I1_e = ufl.tr(C_e)
        return 0.5 * c_e * (I1_e - 3.0)
        
        

def W_smc(C, a_fiber=None):
    """
    Smooth muscle cell strain energy.
    
    Parameters
    ----------
    C : ufl.Expr
        Right Cauchy-Green deformation tensor
    a_fiber : fem.Function or numpy.ndarray, optional
        Fiber direction field. If None, uses constant circumferential direction.
    """
    if a_fiber is None:
        # Fallback to constant circumferential fiber
        a_fiber = np.array([0.0, 1.0, 0.0])
    
    I4m = fiber_invariant(C, a_fiber)
    
    # SMC only contributes in tension (I4 > 1)
    return ufl.conditional(
        ufl.gt(I4m, 1.0),
        (c_m1 / (4.0 * c_m2)) * (ufl.exp(c_m2 * (I4m - 1.0)**2) - 1.0),
        0.0
    )

def W_collagen(C, a_theta=None, a_axial=None, a_diag1=None, a_diag2=None):
    """
    Sum of θ, z, d collagen families, each weighted by β_j.
    Each uses the same (c_c1, c_c2) but different directions.
    
    Parameters
    ----------
    C : ufl.Expr
        Right Cauchy-Green deformation tensor
    a_theta : fem.Function or numpy.ndarray, optional
        Circumferential fiber direction field
    a_axial : fem.Function or numpy.ndarray, optional
        Axial fiber direction field
    a_diag1 : fem.Function or numpy.ndarray, optional
        First diagonal fiber direction field
    a_diag2 : fem.Function or numpy.ndarray, optional
        Second diagonal fiber direction field
    """
    # Fallback to constant directions if not provided
    if a_theta is None:
        a_theta = np.array([0.0, 1.0, 0.0])
    if a_axial is None:
        a_axial = np.array([0.0, 0.0, 1.0])
    if a_diag1 is None:
        a_diag1 = np.array([0.0, np.sin(alpha0), np.cos(alpha0)])
    if a_diag2 is None:
        a_diag2 = np.array([0.0, -np.sin(alpha0), np.cos(alpha0)])
    
    # Compute fiber invariants
    I4_theta = fiber_invariant(C, a_theta)
    I4_z = fiber_invariant(C, a_axial)
    I4_d1 = fiber_invariant(C, a_diag1)
    I4_d2 = fiber_invariant(C, a_diag2)

    # Common exponential function for collagen
    # NOTE: Collagen only contributes in tension (I4 > 1)
    # Using conditional to prevent contribution in compression
    def W_f(I4):
        # Heaviside-like activation: only active when stretched (I4 > 1)
        return ufl.conditional(
            ufl.gt(I4, 1.0),
            (c_c1 / (4.0 * c_c2)) * (ufl.exp(c_c2 * (I4 - 1.0)**2) - 1.0),
            0.0
        )  # eq. (52)

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
    # # U(ln J) = 0.5 * kappa * (ln J)^2
    # lnJ = ufl.ln(J)
    # return 0.5 * kappa * lnJ**2 # Page 15

    # Pre-Jacobian shift JG = exp(-p0/kappa) (paper Stage I)
    JG = ufl.exp(-P0 / kappa)  # Using P0 as p0 for Stage I pre-stress
    lnJ = ufl.ln(J * JG)
    return 0.5 * kappa * lnJ**2 # Page 15

def strain_energy_stage_I(u):
    F, C, J = kinematics(u)
    
    # Get fiber functions if available, otherwise use None (fallback to constants)
    a_smc = fiber_functions.get('theta', None)
    a_theta = fiber_functions.get('theta', None)
    a_axial = fiber_functions.get('axial', None)
    a_diag1 = fiber_functions.get('diagonal1', None)
    a_diag2 = fiber_functions.get('diagonal2', None)
    
    # Get cylindrical basis vectors for elastin deposition stretch
    e_r = fiber_functions.get('basis_radial', None)
    e_theta = fiber_functions.get('basis_circumferential', None)
    e_z = fiber_functions.get('basis_axial', None)
    
    W = (phi_e0 * W_elastin(C, e_r=e_r, e_theta=e_theta, e_z=e_z) +
         phi_m0 * W_smc(C, a_fiber=a_smc) +
         phi_c0 * W_collagen(C, a_theta=a_theta, a_axial=a_axial, 
                            a_diag1=a_diag1, a_diag2=a_diag2) +
         volumetric_energy(J)) # eq. (50)
    return W

# Boundary conditions & loading using loaded facet tags
# Tag 1: Inner surface (lumen) - will be pressurized
# Tag 2: Outer surface - free boundary
# Tag 3: End cap 1 (inlet) - fixed in all directions
# Tag 4: End cap 2 (outlet) - fixed in all directions

if facet_tags is not None:
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
    
    # Define facet measure with loaded tags
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    
else:
    # Fallback to geometric detection if boundary file not loaded
    if rank == 0:
        print("Warning: Using geometric boundary detection (fallback)")
    
    bottom = mesh.locate_entities_boundary(
        domain, dim-1,
        lambda x: np.isclose(x[2], 0.0, atol=1e-8)
    )
    top = mesh.locate_entities_boundary(
        domain, dim-1,
        lambda x: np.isclose(x[2], 1.0, atol=1e-8)
    )
    
    fixed_facets = np.concatenate([bottom, top])
    fixed_dofs = fem.locate_dofs_topological(V, dim-1, fixed_facets)
    
    bc_value = fem.Constant(domain, (0.0, 0.0, 0.0))
    bc_fixed = fem.dirichletbc(bc_value, fixed_dofs, V)
    bcs = [bc_fixed]
    
    # Create facet tags manually for pressure boundary
    # Assuming inner surface for pressure
    inner = mesh.locate_entities_boundary(
        domain, dim-1,
        lambda x: np.sqrt(x[0]**2 + x[1]**2) < 1.1  # Inner radius ~1.0
    )
    facet_tags = mesh.meshtags(domain, dim-1, np.array(inner, dtype=np.int32),
                               np.full_like(inner, 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)



def residual_stage_I(u, v):
    W = strain_energy_stage_I(u)          # scalar energy density
    psi = W * ufl.dx                      # total potential energy

    # Internal virtual work: δΠ_int = d(∫W dx)/du [v]
    R_int = ufl.derivative(psi, u, v)

    # External work from pressure applied to inner lumen surface only (tag 1)
    # ds(1) restricts the pressure to facets with boundary tag 1 (inner wall)
    # Use explicit radially outward pressure direction instead of facet normal
    # This ensures pressure always expands the tube radially
    traction = fem.Constant(domain, P0) * pressure_direction
    R_ext = ufl.inner(traction, v) * ds(1)

    return R_int - R_ext

R_I = residual_stage_I(u, v)
J_I = ufl.derivative(R_I, u, du)



petsc_options = {
    # SNES (Nonlinear Solver) Options for Newton stepping
    "snes_rtol": 1e-5, # Stop when the relative residual drops below 10⁻⁸ (i.e., residual is 0.00000001× the initial residual)
    "snes_atol": 1e-5, # Stop when the absolute residual drops below 10⁻⁹ (i.e., residual norm < 0.000000001)
    "snes_max_it": 5, # Maximum nonlinear iterations before giving up
    "snes_monitor": None, # Print residual norm at each iteration (helps debug convergence)
    "snes_error_if_not_converged": False, # Raise error if not converged
    "snes_linesearch_type": "bt",   # Use backtracking line search to ensure Newton steps decrease the residual
    "snes_linesearch_monitor": None, # Print line search details at each step
    
    # KSP (Linear Solver) Options
    "ksp_type": "gmres", # Use GMRES (Generalized Minimal Residual) Krylov solver for the linearized system at each Newton step
    "ksp_rtol": 1e-6, # Linear solve converges when residual drops to 'ksp_rtol' of initial
    "ksp_atol": 1e-6, # Or when absolute residual is below 'ksp_atol'
    "ksp_max_it": 5000, # Maximum linear solver iterations for each linear solve
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


problem_I = NonlinearProblem(R_I, u, bcs=bcs, J=J_I, petsc_options=petsc_options, petsc_options_prefix="stage_I")

if rank == 0:
    print("\n\n=== Solving Stage I (hyperelastic pre-stress) ===")
    print(f"Using {comm.size} MPI processes")
    print(f"DOFs (global): {V.dofmap.index_map.size_global * V.dofmap.index_map_bs}")
    print(f"DOFs per process (local on rank 0): {V.dofmap.index_map.size_local * V.dofmap.index_map_bs}")

t1 = time()
out = problem_I.solve()
t2 = time()

if rank == 0:
    total_seconds = t2 - t1
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    print(f"Stage I solve time: {hours} hours, {minutes} minutes, {seconds} seconds")
    print(f"  Time per DOF: {(t2-t1)/(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)*1e6:.2f} μs")
else:
    print(f"Rank {rank} solve time: {t2 - t1:.2f} seconds")

# -----------------------------------------------------------
# Compute displacement components and stress
# -----------------------------------------------------------

if rank == 0:
    print("\n=== Computing post-processing fields ===")

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

# Compute pressure field (hydrostatic pressure = -mean normal stress)
# p = -tr(σ)/3
F, C, J = kinematics(u)
W_expr = strain_energy_stage_I(u)
F = ufl.variable(F)
P = ufl.diff(W_expr, F)
sigma = (1.0 / J) * P * F.T
pressure_ufl = -(1.0/3.0) * ufl.tr(sigma)

# Create DG0 scalar function for pressure
DG0_scalar_p = fem.functionspace(domain, ("DG", 0))
pressure_func = fem.Function(DG0_scalar_p, name="pressure")
pressure_expr = fem.Expression(pressure_ufl, DG0_scalar_p.element.interpolation_points)
pressure_func.interpolate(pressure_expr)

# Add pressure to cell data
pvmesh.cell_data["pressure"] = pressure_func.x.array

warped_mesh = pvmesh.warp_by_vector("u", factor=1.0)

# Plot with pyvista - extract surface to show actual cell types (quads/tris)
plotter = pv.Plotter()
warped_surface = warped_mesh.extract_surface()
base_surface = base_mesh.extract_surface()

# Plot displacement on deformed configuration
plotter.add_mesh(warped_surface, show_edges=True, show_scalar_bar=True, 
                scalars="u", cmap="jet", edge_color='black',
                scalar_bar_args={'title': 'Pressure (Pa)'})
plotter.add_mesh(base_surface, style='wireframe', color='gray', opacity=0.3, 
                line_width=1, label="Undeformed Mesh")
plotter.add_axes()
plotter.show()








# Compute Cauchy stress tensor
# σ = (1/J) P F^T where P = ∂W/∂F
F, C, J = kinematics(u)
F = ufl.variable(F)
C = ufl.variable(C)
W_expr = strain_energy_stage_I(u)

# Compute first Piola-Kirchhoff stress P = ∂W/∂F
P = ufl.diff(W_expr, F)

# Compute Cauchy stress: σ = (1/J) P F^T
sigma = (1.0 / J) * P * F.T

# Create DG0 tensor function space for stress (piecewise constant per element)
DG0_tensor = fem.functionspace(domain, ("DG", 0, (dim, dim)))
sigma_dg0 = fem.Function(DG0_tensor, name="sigma")

# Project stress to DG0 space
sigma_expr = fem.Expression(sigma, DG0_tensor.element.interpolation_points())
sigma_dg0.interpolate(sigma_expr)

# Extract stress components for visualization
sigma_arr = sigma_dg0.x.array.reshape((-1, dim, dim))

# Compute von Mises stress
# For 3D: σ_vm = sqrt(0.5 * [(σ11-σ22)² + (σ22-σ33)² + (σ33-σ11)² + 6(σ12² + σ23² + σ13²)])
s11 = sigma_arr[:, 0, 0]
s22 = sigma_arr[:, 1, 1]
s33 = sigma_arr[:, 2, 2]
s12 = sigma_arr[:, 0, 1]
s23 = sigma_arr[:, 1, 2]
s13 = sigma_arr[:, 0, 2]

von_mises = np.sqrt(0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2 + 
                           6 * (s12**2 + s23**2 + s13**2)))

# Compute hydrostatic stress (mean normal stress)
hydrostatic = (s11 + s22 + s33) / 3.0

# Create DG0 scalar functions for visualization
DG0_scalar = fem.functionspace(domain, ("DG", 0))
von_mises_func = fem.Function(DG0_scalar, name="von_mises_stress")
hydrostatic_func = fem.Function(DG0_scalar, name="hydrostatic_stress")

von_mises_func.x.array[:] = von_mises
hydrostatic_func.x.array[:] = hydrostatic

if rank == 0:
    print(f"  von Mises stress: min={von_mises.min():.2e}, max={von_mises.max():.2e}")
    print(f"  Hydrostatic stress: min={hydrostatic.min():.2e}, max={hydrostatic.max():.2e}")

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

# Add stress to cell data (DG0 is piecewise constant per cell)
pvmesh.cell_data["von_mises"] = von_mises
pvmesh.cell_data["hydrostatic"] = hydrostatic
pvmesh.cell_data["sigma_xx"] = s11
pvmesh.cell_data["sigma_yy"] = s22
pvmesh.cell_data["sigma_zz"] = s33

warped_mesh = pvmesh.warp_by_vector("u", factor=1.0)

# Plot with pyvista - extract surface to show actual cell types (quads/tris)
plotter = pv.Plotter()
warped_surface = warped_mesh.extract_surface()
base_surface = base_mesh.extract_surface()

# Plot von Mises stress on deformed configuration
plotter.add_mesh(warped_surface, show_edges=True, show_scalar_bar=True, 
                scalars="von_mises", cmap="jet", edge_color='black',
                scalar_bar_args={'title': 'von Mises Stress (Pa)'})
plotter.add_mesh(base_surface, style='wireframe', color='gray', opacity=0.3, 
                line_width=1, label="Undeformed Mesh")
plotter.add_axes()
plotter.show()


from dolfinx.io import VTKFile

# Save results to VTU file
if rank == 0:
    print("\n=== Saving results ===")

with VTKFile(domain.comm, "CMM_tube_results.vtu", "w") as f:
    f.write_mesh(domain)
    f.write_function(u_out)
    f.write_function(von_mises_func)
    f.write_function(hydrostatic_func)

if rank == 0:
    print("  Results saved to CMM_tube_results.vtu")

b=0
