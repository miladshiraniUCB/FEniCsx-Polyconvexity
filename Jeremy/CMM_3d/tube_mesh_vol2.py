from dataclasses import dataclass
import numpy as np
from typing import Literal, Dict, Any, Optional
import pyvista as pv
import meshio

MeshType = Literal["tet4", "tet10", "hex8", "hex20"]
AxisType = Literal["x", "y", "z"]

def generate_tube_volume_mesh(
    axial_length: float,
    lumen_diameter: float,
    wall_thickness: float,
    n_axial: int,
    n_circ: int,
    n_radial: int,
    axis: AxisType = "z",
    mesh_type: MeshType = "hex8",
    fiber_angles: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Generate a volumetric mesh of a cylindrical tube (pipe).

    Parameters
    ----------
    axial_length : float
        Tube length along the chosen axis.
    lumen_diameter : float
        Inner lumen diameter.
    wall_thickness : float
        Tube wall thickness.
    n_axial : int
        Axial subdivisions along the tube.
    n_circ : int
        Circumferential subdivisions.
    n_radial : int
        Radial subdivisions through the wall thickness (>=1).
    axis : {'x','y','z'}
        Which axis the tube is aligned with.
    mesh_type : {'tet4', 'tet10', 'hex8', 'hex20'}
        Type of volumetric elements:
            'tet4'  : linear tetrahedral elements
            'tet10' : quadratic tetrahedral elements
            'hex8'  : linear hexahedral elements
            'hex20' : quadratic hexahedral elements
    fiber_angles : dict[str, float], optional
        Dictionary of fiber family names to angles (in radians).
        Each angle defines the fiber orientation with respect to the axial direction.
        Example: {'axial': 0.0, 'helical': np.pi/4}
        If None, a single axial fiber family is created with angle 0.0.

    Returns
    -------
    mesh : dict
        {
          'nodes': (N, 3) float array of nodal coordinates,
          'elements': (M, nnpe) int array of connectivity (0-based indices),
          'element_type': str,   # 'hex8','hex20','tet4','tet10'
          'element_order': int,  # 1 or 2
          'fiber_families': dict[str, (M, 3)] of fiber orientation vectors per element,
          'boundary_faces': (K, npf) int array of boundary face connectivity,
          'boundary_tags': (K,) int array of boundary tags
        }

    Notes
    -----
    * Geometry is a hollow tube (inner radius to outer radius).
    * Mesh is structured in (r, theta, axial) before conversion to Cartesian.
    * Circumferential direction is periodic with a "seam" at theta=0/2π.
    """

    if n_axial < 1 or n_circ < 3 or n_radial < 1:
        raise ValueError("n_axial>=1, n_circ>=3, n_radial>=1 required.")
    if mesh_type not in ("tet4", "tet10", "hex8", "hex20"):
        raise ValueError("mesh_type must be 'tet4', 'tet10', 'hex8', or 'hex20'.")
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be 'x', 'y', or 'z'.")
    
    # Default to single axial fiber family if none specified
    if fiber_angles is None:
        fiber_angles = {"axial": 0.0}
    
    # Derive facet_type and element_order from mesh_type
    facet_type = "tri" if mesh_type.startswith("tet") else "quad"
    element_order = 2 if mesh_type.endswith("10") or mesh_type.endswith("20") else 1

    # Radii and parametric grids
    r_inner = lumen_diameter / 2.0
    r_outer = r_inner + wall_thickness

    radial_vals = np.linspace(r_inner, r_outer, n_radial + 1)     # 0..n_radial
    axial_vals = np.linspace(0.0, axial_length, n_axial + 1)      # 0..n_axial
    theta_vals = np.linspace(0.0, 2.0 * np.pi, n_circ, endpoint=False)  # 0..n_circ-1

    # --- 1) Build base grid of linear nodes (corner nodes only) ---

    def cyl_to_cart(r, a, theta):
        """Map cylindrical (r, axial, theta) to Cartesian (x,y,z) given axis."""
        x_c = r * np.cos(theta)
        y_c = r * np.sin(theta)
        if axis == "z":
            return np.array([x_c, y_c, a])
        elif axis == "x":
            # Tube along X axis: axial -> x, circle in YZ
            return np.array([a, x_c, y_c])
        else:  # axis == "y"
            # Tube along Y axis: axial -> y, circle in XZ
            return np.array([x_c, a, y_c])

    # Node indexing: (ir, ia, it) -> node_id
    # ir: 0..n_radial, ia: 0..n_axial, it: 0..n_circ-1
    def node_id(ir, ia, it):
        return (ir * (n_axial + 1) + ia) * n_circ + it

    num_linear_nodes = (n_radial + 1) * (n_axial + 1) * n_circ
    nodes = np.zeros((num_linear_nodes, 3), dtype=float)

    for ir, r in enumerate(radial_vals):
        for ia, a in enumerate(axial_vals):
            for it, theta in enumerate(theta_vals):
                nid = node_id(ir, ia, it)
                nodes[nid, :] = cyl_to_cart(r, a, theta)

    # --- 2) Build hexahedral cells in terms of linear nodes ---

    # Hex indexing:
    # cell indices: ir=0..n_radial-1, ia=0..n_axial-1, it=0..n_circ-1
    # Local vertex order (v0..v7) chosen as:
    #   v0 = (ir,   ia,   it)
    #   v1 = (ir+1, ia,   it)
    #   v2 = (ir+1, ia+1, it)
    #   v3 = (ir,   ia+1, it)
    #   v4 = (ir,   ia,   it+1)
    #   v5 = (ir+1, ia,   it+1)
    #   v6 = (ir+1, ia+1, it+1)
    #   v7 = (ir,   ia+1, it+1)
    # with circumferential wrap: it+1 -> (it + 1) % n_circ

    hex_elements = []
    for ir in range(n_radial):
        for ia in range(n_axial):
            for it in range(n_circ):
                itn = (it + 1) % n_circ
                v0 = node_id(ir,     ia,     it)
                v1 = node_id(ir + 1, ia,     it)
                v2 = node_id(ir + 1, ia + 1, it)
                v3 = node_id(ir,     ia + 1, it)
                v4 = node_id(ir,     ia,     itn)
                v5 = node_id(ir + 1, ia,     itn)
                v6 = node_id(ir + 1, ia + 1, itn)
                v7 = node_id(ir,     ia + 1, itn)
                hex_elements.append([v0, v1, v2, v3, v4, v5, v6, v7])

    hex_elements = np.asarray(hex_elements, dtype=int)

    # --- 3) Boundary quad faces (linear) ---
    boundary_quads = []
    boundary_tags  = []

    # Inner wall (ir=0)
    # Correct ordering for outward normal (toward axis): [v0, v3, v7, v4]
    ir = 0
    for ia in range(n_axial):
        for it in range(n_circ):
            itn = (it + 1) % n_circ
            q = [
                node_id(ir, ia, it),      # v0
                node_id(ir, ia+1, it),    # v3
                node_id(ir, ia+1, itn),   # v7
                node_id(ir, ia, itn)      # v4
            ]
            boundary_quads.append(q)
            boundary_tags.append(1)

    # Outer wall (ir=n_radial)
    # Correct ordering for outward normal (away from axis): [v1, v5, v6, v2]
    ir = n_radial
    for ia in range(n_axial):
        for it in range(n_circ):
            itn = (it + 1) % n_circ
            q = [
                node_id(ir, ia, it),      # v1
                node_id(ir, ia, itn),     # v5
                node_id(ir, ia+1, itn),   # v6
                node_id(ir, ia+1, it)     # v2
            ]
            boundary_quads.append(q)
            boundary_tags.append(2)

    # End cap 1 (inlet) ia = 0
    # Correct ordering for outward normal (-z direction): [v0, v4, v5, v1]
    ia = 0
    for ir in range(n_radial):
        for it in range(n_circ):
            itn = (it + 1) % n_circ
            q = [
                node_id(ir,     ia, it),   # v0
                node_id(ir,     ia, itn),  # v4
                node_id(ir+1,   ia, itn),  # v5
                node_id(ir+1,   ia, it)    # v1
            ]
            boundary_quads.append(q)
            boundary_tags.append(3)

    # End cap 2 (outlet) ia = n_axial
    # Correct ordering for outward normal (+z direction): [v3, v2, v6, v7]
    ia = n_axial
    for ir in range(n_radial):
        for it in range(n_circ):
            itn = (it + 1) % n_circ
            q = [
                node_id(ir,     ia, it),   # v3
                node_id(ir+1,   ia, it),   # v2
                node_id(ir+1,   ia, itn),  # v6
                node_id(ir,     ia, itn)   # v7
            ]
            boundary_quads.append(q)
            boundary_tags.append(4)

    boundary_quads = np.asarray(boundary_quads, int)
    boundary_tags  = np.asarray(boundary_tags, int)


    # --- 4) Convert hex cells to desired element type/order ---

    # Helper: global edge -> mid-node index, shared across elements
    # key = (min_id, max_id) -> mid_node_id
    edge_to_mid = {}

    def get_midpoint_node(i, j):
        """Return index of mid-edge node between linear nodes i and j."""
        key = (i, j) if i < j else (j, i)
        if key in edge_to_mid:
            return edge_to_mid[key]
        # Create new mid-node
        p = 0.5 * (nodes[key[0]] + nodes[key[1]])
        edge_to_mid[key] = len(nodes_list)
        nodes_list.append(p)
        return edge_to_mid[key]

    # For quadratic elements, we'll extend the nodes array
    nodes_list = list(nodes)

    if facet_type == "quad":
        # --- Hexahedral elements: hex8 or hex20 ---

        if element_order == 1:
            # Linear hex8
            elements = hex_elements
            element_type = "hex8"

        else:
            # Quadratic hex20: 8 corners + 12 mid-edge nodes
            # Edge list (by local index v0..v7) in a consistent order:
            # bottom face: 0-1, 1-2, 2-3, 3-0
            # top face:    4-5, 5-6, 6-7, 7-4
            # verticals:   0-4, 1-5, 2-6, 3-7
            quad_elements = []
            for cell in hex_elements:
                v0, v1, v2, v3, v4, v5, v6, v7 = cell

                # Corner nodes (first 8)
                corner = [v0, v1, v2, v3, v4, v5, v6, v7]

                # Edges: 12 of them in chosen order
                edge_pairs = [
                    (v0, v1), (v1, v2), (v2, v3), (v3, v0),
                    (v4, v5), (v5, v6), (v6, v7), (v7, v4),
                    (v0, v4), (v1, v5), (v2, v6), (v3, v7),
                ]

                mids = [get_midpoint_node(i, j) for (i, j) in edge_pairs]

                quad_elements.append(corner + mids)

            nodes = np.asarray(nodes_list, dtype=float)
            elements = np.asarray(quad_elements, dtype=int)
            element_type = "hex20"

    else:
        # --- Tetrahedral elements: tet4 or tet10 (via hex subdivision) ---

        # First, define tet4 elements by subdividing each hex8
        # We'll use a 5-tet decomposition of the cube with local v0..v7:
        #
        # Example set of 5 tets:
        #   [v0, v1, v3, v4]
        #   [v1, v2, v3, v6]
        #   [v1, v3, v4, v6]
        #   [v3, v4, v6, v7]
        #   [v1, v4, v5, v6]
        #
        # (All in terms of the local hex8 vertices.)

        tet4_list = []
        for cell in hex_elements:
            v0, v1, v2, v3, v4, v5, v6, v7 = cell
            tet4_list.extend([
                [v0, v1, v3, v4],
                [v1, v2, v3, v6],
                [v1, v3, v4, v6],
                [v3, v4, v6, v7],
                [v1, v4, v5, v6],
            ])

        tet4_array = np.asarray(tet4_list, dtype=int)

        if element_order == 1:
            # Linear tet4
            elements = tet4_array
            element_type = "tet4"
        else:
            # Quadratic tet10: 4 corners + 6 mid-edge nodes
            tet10_list = []
            for tet in tet4_array:
                a, b, c, d = tet
                corners = [a, b, c, d]
                # Edges: 6 unique edges of the tetrahedron
                edge_pairs = [
                    (a, b), (a, c), (a, d),
                    (b, c), (b, d),
                    (c, d),
                ]
                mids = [get_midpoint_node(i, j) for (i, j) in edge_pairs]
                tet10_list.append(corners + mids)

            nodes = np.asarray(nodes_list, dtype=float)
            elements = np.asarray(tet10_list, dtype=int)
            element_type = "tet10"

    if facet_type == "tri":
        tri_faces = []
        tri_tags  = []
        for face, tag in zip(boundary_quads, boundary_tags):
            v0, v1, v2, v3 = face
            tri_faces.append([v0, v1, v2])
            tri_tags.append(tag)
            tri_faces.append([v0, v2, v3])
            tri_tags.append(tag)
        boundary_faces = np.asarray(tri_faces, int)
        boundary_tags  = np.asarray(tri_tags, int)
    else:
        boundary_faces = boundary_quads
    
    # --- 5) Compute fiber orientation vectors and local basis for each element ---
    
    # Get element centroids to determine local coordinate system
    num_elements = elements.shape[0]
    
    # Use final nodes array (which may include midpoint nodes for quadratic elements)
    if element_order == 2:
        nodes_final = np.asarray(nodes_list, dtype=float)
    else:
        nodes_final = nodes
    
    # Store fiber vectors for each family
    fiber_families = {}
    
    # Also store local cylindrical basis vectors (for elastin deposition stretch)
    # These will be stored as separate "fiber families" for reading in FEniCS
    radial_vectors = np.zeros((num_elements, 3), dtype=float)
    circumferential_vectors = np.zeros((num_elements, 3), dtype=float)
    axial_vectors = np.zeros((num_elements, 3), dtype=float)
    
    for family_name, fiber_angle in fiber_angles.items():
        fiber_vectors = np.zeros((num_elements, 3), dtype=float)
        
        for i, elem in enumerate(elements):
            # Compute element centroid using corner nodes only
            if element_type in ("hex8", "hex20"):
                # First 8 nodes are corners for hex
                corner_nodes = elem[:8]
            else:  # tet4 or tet10
                # First 4 nodes are corners for tet
                corner_nodes = elem[:4]
            
            centroid = np.mean(nodes_final[corner_nodes], axis=0)
            
            # Determine local cylindrical coordinates at centroid
            if axis == "z":
                x_c, y_c, z_c = centroid
                r = np.sqrt(x_c**2 + y_c**2)
                theta = np.arctan2(y_c, x_c)
                # Radial direction (outward)
                e_r = np.array([np.cos(theta), np.sin(theta), 0.0])
                # Circumferential (tangential) direction
                e_theta = np.array([-np.sin(theta), np.cos(theta), 0.0])
                # Axial direction
                e_axial = np.array([0.0, 0.0, 1.0])
            elif axis == "x":
                x_c, y_c, z_c = centroid
                r = np.sqrt(y_c**2 + z_c**2)
                theta = np.arctan2(z_c, y_c)
                # Radial direction (outward)
                e_r = np.array([0.0, np.cos(theta), np.sin(theta)])
                # Circumferential direction
                e_theta = np.array([0.0, -np.sin(theta), np.cos(theta)])
                # Axial direction
                e_axial = np.array([1.0, 0.0, 0.0])
            else:  # axis == "y"
                x_c, y_c, z_c = centroid
                r = np.sqrt(x_c**2 + z_c**2)
                theta = np.arctan2(z_c, x_c)
                # Radial direction (outward)
                e_r = np.array([np.cos(theta), 0.0, np.sin(theta)])
                # Circumferential direction
                e_theta = np.array([-np.sin(theta), 0.0, np.cos(theta)])
                # Axial direction
                e_axial = np.array([0.0, 1.0, 0.0])
            
            # Store cylindrical basis vectors (only need to do this once, not per family)
            if family_name == list(fiber_angles.keys())[0]:  # First family only
                radial_vectors[i] = e_r
                circumferential_vectors[i] = e_theta
                axial_vectors[i] = e_axial
            
            # Fiber direction: combination of circumferential and axial directions
            # fiber = cos(fiber_angle) * e_axial + sin(fiber_angle) * e_theta
            fiber = np.cos(fiber_angle) * e_axial + np.sin(fiber_angle) * e_theta
            
            # Normalize (should already be normalized, but ensure it)
            fiber = fiber / np.linalg.norm(fiber)
            
            fiber_vectors[i] = fiber
        
        fiber_families[family_name] = fiber_vectors
    
    # Add cylindrical basis vectors as special fiber families
    fiber_families["basis_radial"] = radial_vectors
    fiber_families["basis_circumferential"] = circumferential_vectors
    fiber_families["basis_axial"] = axial_vectors
        
    return {
        "nodes": nodes,
        "elements": elements,
        "element_type": element_type,
        "element_order": element_order,
        "fiber_families": fiber_families,
        "boundary_faces": boundary_faces,
        "boundary_tags": boundary_tags,
    }

@dataclass
class DolfinxTubeModel:
    mesh: "object"
    cell_tags: "object"
    facet_tags: "object"
    fibers: Dict[str, "object"]  # name -> dolfinx.fem.Function


def tube_mesh_to_dolfinx_model(
    mesh_dict: dict,
    comm=None,
    rank: int = 0,
) -> DolfinxTubeModel:
    """
    Convert the returned mesh dict into a dolfinx mesh + MeshTags, and attach fiber vectors
    as DG0 vector Functions (one per fiber family).

    Implementation note:
      We rebuild a discrete gmsh model in-memory and use gmshio.model_to_mesh, which
      avoids having to guess dolfinx/basix node ordering for higher-order cells.

    Returns:
      DolfinxTubeModel(mesh, cell_tags, facet_tags, fibers)
    """
    from mpi4py import MPI
    import gmsh
    from dolfinx.io import gmsh as gmshio
    from dolfinx import fem

    if comm is None:
        comm = MPI.COMM_WORLD

    # --- Broadcast mesh_dict (lightweight approach) ---
    # If you already have mesh_dict on all ranks, this is still OK.
    mesh_dict = comm.bcast(mesh_dict if comm.rank == rank else None, root=rank)

    nodes = mesh_dict["nodes"]
    elements = mesh_dict["elements"]
    element_type = mesh_dict["element_type"]
    order = int(mesh_dict["element_order"])
    bfaces = mesh_dict["boundary_faces"]
    btags = mesh_dict["boundary_tags"]
    fiber_families = mesh_dict["fiber_families"]

    gmsh_vol_type = {
        "tet4": 4,
        "tet10": 11,
        "hex8": 5,
        "hex20": 17,
    }[element_type]
    cell_kind = "tet" if element_type.startswith("tet") else "hex"
    gmsh_facet_type = (2 if order == 1 else 9) if cell_kind == "tet" else (3 if order == 1 else 16)

    faces_by_tag: Dict[int, np.ndarray] = {}
    for tag in (1, 2, 3, 4):
        faces_by_tag[tag] = bfaces[btags == tag]

    # Build gmsh model on root only
    if comm.rank == rank:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("walled_tube")

        for s in (1, 2, 3, 4):
            gmsh.model.addDiscreteEntity(2, s)
        gmsh.model.addDiscreteEntity(3, 1, [1, 2, 3, 4])

        gmsh.model.addPhysicalGroup(3, [1], 1)
        for s in (1, 2, 3, 4):
            gmsh.model.addPhysicalGroup(2, [s], s)

        N = nodes.shape[0]
        node_tags = np.arange(1, N + 1, dtype=np.int64)
        gmsh.model.mesh.addNodes(3, 1, node_tags.tolist(), nodes.reshape(-1).tolist())

        M = elements.shape[0]
        elem_tags = np.arange(1, M + 1, dtype=np.int64)
        gmsh.model.mesh.addElements(
            3, 1,
            [gmsh_vol_type],
            [elem_tags.tolist()],
            [(elements + 1).astype(np.int64).reshape(-1).tolist()]
        )

        next_tag = int(elem_tags[-1]) + 1
        for s in (1, 2, 3, 4):
            F = faces_by_tag[s]
            if F.size == 0:
                continue
            K = F.shape[0]
            face_tags = np.arange(next_tag, next_tag + K, dtype=np.int64)
            next_tag += K
            gmsh.model.mesh.addElements(
                2, s,
                [gmsh_facet_type],
                [face_tags.tolist()],
                [(F + 1).astype(np.int64).reshape(-1).tolist()]
            )

    # Convert to dolfinx mesh (+ tags)
    mesh_data = gmshio.model_to_mesh(gmsh.model, comm, rank, gdim=3)
    domain = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags

    if comm.rank == rank:
        gmsh.finalize()

    # Attach fiber families as DG0 vector functions
    V = fem.functionspace(domain, ("DG", 0, (domain.geometry.dim,)))
    fibers_out: Dict[str, fem.Function] = {}

    # We assume the cell ordering matches the order we inserted elements (tags 1..M),
    # and that gmshio distributes cells in contiguous ranges of the global ordering.
    # This is typically true for model_to_mesh in dolfinx.
    imap = domain.topology.index_map(domain.topology.dim)
    lo, hi = imap.local_range  # global cell index range owned by this rank

    for name, fvec_global in fiber_families.items():
        fvec_global = np.asarray(fvec_global, dtype=float)
        f_local = fvec_global[lo:hi]  # (n_local_cells, 3)

        fn = fem.Function(V)
        # DG0 vector: one vector value per cell (owned cells are first in local numbering)
        # fn.x.array is flat, size = (num_local_cells + num_ghosts) * gdim
        # Fill owned part; leave ghosts as 0 (can be updated by scatter_forward if needed).
        gdim = domain.geometry.dim
        owned = hi - lo
        fn.x.array[: owned * gdim] = f_local.reshape(-1)
        fn.x.scatter_forward()
        fibers_out[name] = fn

    return DolfinxTubeModel(mesh=domain, cell_tags=cell_tags, facet_tags=facet_tags, fibers=fibers_out)






if __name__ == "__main__":
    # Fiber families for constrained mixture model
    # theta: circumferential (90 degrees from axial)
    # axial: aligned with tube axis (0 degrees)
    # diagonal1, diagonal2: symmetric diagonal families (±29.91 degrees)
    alpha0_deg = 29.91  # degrees
    alpha0 = alpha0_deg * np.pi / 180.0  # radians
    
    mesh = generate_tube_volume_mesh(
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

    # Build an in-memory dolfinx model
    model = tube_mesh_to_dolfinx_model(mesh)
    print(model.mesh, model.facet_tags, list(model.fibers.keys()))

    nodes = mesh["nodes"]
    cells = mesh["elements"]
    mesh_order = mesh["element_order"]
    mesh_type = mesh["element_type"]

    if mesh_type == "hex8":
        # For hex8, VTK expects [8, n0, n1, ..., n7] per cell
        num_cells = cells.shape[0]
        vtk_cells = np.hstack([np.full((num_cells, 1), 8, dtype=int), cells]).ravel()
        grid = pv.UnstructuredGrid({pv.CellType.HEXAHEDRON: cells}, nodes)
    elif mesh_type == "tet4":
        # For tet4, VTK expects [4, n0, n1, n2, n3] per cell
        num_cells = cells.shape[0]
        vtk_cells = np.hstack([np.full((num_cells, 1), 4, dtype=int), cells]).ravel()
        grid = pv.UnstructuredGrid({pv.CellType.TETRA: cells}, nodes)
    elif mesh_type == "hex20":
        # For hex20, VTK expects [20, n0, n1, ..., n19] per cell
        num_cells = cells.shape[0]
        vtk_cells = np.hstack([np.full((num_cells, 1), 20, dtype=int), cells]).ravel()
        grid = pv.UnstructuredGrid({pv.CellType.QUADRATIC_HEXAHEDRON: cells}, nodes)
    elif mesh_type == "tet10":
        # For tet10, VTK expects [10, n0, n1, ..., n9] per cell
        num_cells = cells.shape[0]
        vtk_cells = np.hstack([np.full((num_cells, 1), 10, dtype=int), cells]).ravel()
        grid = pv.UnstructuredGrid({pv.CellType.QUADRATIC_TETRA: cells}, nodes)

    # Add fiber families to grid as cell data
    for family_name, fiber_vectors in mesh["fiber_families"].items():
        grid.cell_data[f"fiber_{family_name}"] = fiber_vectors
    
    # Compute element centroids for glyph positioning
    element_centroids = grid.cell_centers().points
    
    # Create visualization
    plotter = pv.Plotter()
    
    # Add mesh with transparency
    plotter.add_mesh(grid, show_edges=True, opacity=0.3, color='lightgray', label='Mesh')
    
    # Add fiber glyphs for each family with different colors
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    for idx, (family_name, fiber_vectors) in enumerate(mesh["fiber_families"].items()):
        # Create a point cloud at element centroids
        fiber_cloud = pv.PolyData(element_centroids)
        fiber_cloud[f"fiber_{family_name}"] = fiber_vectors
        
        # Create arrow glyphs
        color = colors[idx % len(colors)]
        arrows = fiber_cloud.glyph(
            orient=f"fiber_{family_name}",
            scale=False,
            factor=0.3,  # Adjust arrow size
            geom=pv.Arrow()
        )
        
        # Cast to PolyData to satisfy type checker
        if isinstance(arrows, pv.PolyData):
            plotter.add_mesh(arrows, color=color, label=f"Fiber: {family_name}")
    
    # plotter.add_legend()  # Legend is automatically added when labels are provided
    # plotter.add_axes()
    plotter.show()

    # Write mesh to XDMF format - separate files for mesh, boundaries, and fibers
    meshio_cell_type = {
        "hex8": "hexahedron",
        "hex20": "hexahedron20",
        "tet4": "tetra",
        "tet10": "tetra10",
    }[mesh_type]
    
    meshio_boundary_type = "triangle" if mesh["boundary_faces"].shape[1] == 3 else "quad"

    # 1. Write volume mesh only
    mesh_volume = meshio.Mesh(
        points=nodes,
        cells=[(meshio_cell_type, cells)]
    )
    mesh_volume.write("tube_mesh.msh", file_format="gmsh")
    mesh_volume.write("tube_mesh.xdmf")
    print("\nVolume mesh written to tube_mesh.xdmf")
    print(f"  Volume elements: {cells.shape[0]} {mesh_type}")
    
    # 2. Write boundary mesh with tags
    mesh_boundary = meshio.Mesh(
        points=nodes,
        cells=[(meshio_boundary_type, mesh["boundary_faces"])],
        cell_data={"boundary_tags": [mesh["boundary_tags"]]}
    )
    mesh_boundary.write("tube_mesh_boundary.msh", file_format="gmsh")
    mesh_boundary.write("tube_mesh_boundary.xdmf")
    print(f"Boundary mesh written to tube_mesh_boundary.xdmf")
    print(f"  Boundary faces: {mesh['boundary_faces'].shape[0]} {meshio_boundary_type}")
    print(f"  Boundary tags: {np.unique(mesh['boundary_tags'])}")
    
    # 3. Write fiber families as cell data on volume mesh
    fiber_cell_data = {}
    for family_name, fiber_vectors in mesh["fiber_families"].items():
        fiber_cell_data[f"fiber_{family_name}"] = [fiber_vectors]
    
    mesh_fibers = meshio.Mesh(
        points=nodes,
        cells=[(meshio_cell_type, cells)],
        cell_data=fiber_cell_data
    )
    mesh_fibers.write("tube_mesh_fibers.msh", file_format="gmsh")
    mesh_fibers.write("tube_mesh_fibers.xdmf")
    print(f"Fiber fields written to tube_mesh_fibers.xdmf")
    print(f"  Fiber families: {len(mesh['fiber_families'])}")
    # for family_name, fiber_vectors in mesh["fiber_families"].items():
    #     # Compute average fiber direction and some statistics
    #     avg_fiber = np.mean(fiber_vectors, axis=0)
    #     norm_avg = np.linalg.norm(avg_fiber)
    #     print(f"    - {family_name}: {fiber_vectors.shape[0]} vectors")
    #     print(f"      Average direction: [{avg_fiber[0]:.4f}, {avg_fiber[1]:.4f}, {avg_fiber[2]:.4f}], norm={norm_avg:.4f}")
    #     print(f"      First 3 vectors:")
    #     for i in range(min(3, len(fiber_vectors))):
    #         fv = fiber_vectors[i]
    #         print(f"        Element {i}: [{fv[0]:.6f}, {fv[1]:.6f}, {fv[2]:.6f}]")
