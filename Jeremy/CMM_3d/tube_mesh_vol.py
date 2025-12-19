import numpy as np
from typing import Literal, Tuple, Dict
import pyvista as pv
import meshio


FacetType = Literal["quad", "tri"]
AxisType = Literal["x", "y", "z"]

def generate_tube_mesh(
    axial_length: float,
    lumen_diameter: float,
    wall_thickness: float,
    n_axial: int,
    n_circ: int,
    n_radial: int,
    axis: AxisType = "z",
    facet_type: FacetType = "quad",
    element_order: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Generate a hollow tube volumetric mesh with watertight end caps.

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
        Number of radial subdivisions through the wall thickness (>=1).
    axis : {'x','y','z'}
        Which axis the tube is aligned with.
    facet_type : {'quad','tri'}
        Type of mesh elements: 'quad' -> hexahedral, 'tri' -> tetrahedral.
    element_order : int
        Element order: 1 for linear (tet4/hex8), 2 for quadratic (tet10/hex20).

    Returns
    -------
    dict with keys:
        'vertices' : (N,3)
        'cells' : (M, nodes_per_element) element connectivity
        'cell_type' : 'tetra' or 'hexahedron'
        'element_order' : int
        'boundary_faces' : boundary face connectivity
        'boundary_tags' : tag for each boundary face
    """

    if n_axial < 1 or n_circ < 3 or n_radial < 1:
        raise ValueError("n_axial>=1, n_circ>=3, n_radial>=1 required.")

    if facet_type not in ("quad", "tri"):
        raise ValueError("facet_type must be 'quad' or 'tri'.")
    
    if element_order not in (1, 2):
        raise ValueError("element_order must be 1 or 2.")

    # Radii
    r_inner = lumen_diameter / 2.0
    r_outer = r_inner + wall_thickness
    axial_vals = np.linspace(0, axial_length, n_axial + 1)
    theta_vals = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
    radial_vals = np.linspace(r_inner, r_outer, n_radial + 1)

    def cyl_vertex(r, a, theta):
        """Generate vertex in cylindrical coordinates."""
        x_c = r * np.cos(theta)
        y_c = r * np.sin(theta)
        if axis == "z":
            return np.array([x_c, y_c, a])
        elif axis == "x":
            return np.array([a, x_c, y_c])
        else:  # axis == "y"
            return np.array([x_c, a, y_c])

    # =================================================================
    # Build 3D structured grid of vertices: [axial, radial, circumferential]
    # =================================================================
    vertex_ids = np.zeros((n_axial + 1, n_radial + 1, n_circ), dtype=int)
    vertices = []
    
    for ia, a in enumerate(axial_vals):
        for ir, r in enumerate(radial_vals):
            for it, theta in enumerate(theta_vals):
                vertex_ids[ia, ir, it] = len(vertices)
                vertices.append(cyl_vertex(r, a, theta))
    
    vertices = np.array(vertices, float)

    # =================================================================
    # Generate volumetric cells (hexahedra or tetrahedra)
    # =================================================================
    cells = []
    
    if facet_type == "quad":
        # Generate hexahedral elements
        for ia in range(n_axial):
            for ir in range(n_radial):
                for it in range(n_circ):
                    itn = (it + 1) % n_circ
                    
                    # Hex vertices (following VTK convention)
                    v0 = vertex_ids[ia, ir, it]
                    v1 = vertex_ids[ia, ir, itn]
                    v2 = vertex_ids[ia, ir + 1, itn]
                    v3 = vertex_ids[ia, ir + 1, it]
                    v4 = vertex_ids[ia + 1, ir, it]
                    v5 = vertex_ids[ia + 1, ir, itn]
                    v6 = vertex_ids[ia + 1, ir + 1, itn]
                    v7 = vertex_ids[ia + 1, ir + 1, it]
                    
                    cells.append([v0, v1, v2, v3, v4, v5, v6, v7])
        
        cell_type = "hexahedron"
    
    else:  # facet_type == "tri"
        # Generate tetrahedral elements by splitting each hex into 6 tets
        for ia in range(n_axial):
            for ir in range(n_radial):
                for it in range(n_circ):
                    itn = (it + 1) % n_circ
                    
                    # Hex vertices
                    v0 = vertex_ids[ia, ir, it]
                    v1 = vertex_ids[ia, ir, itn]
                    v2 = vertex_ids[ia, ir + 1, itn]
                    v3 = vertex_ids[ia, ir + 1, it]
                    v4 = vertex_ids[ia + 1, ir, it]
                    v5 = vertex_ids[ia + 1, ir, itn]
                    v6 = vertex_ids[ia + 1, ir + 1, itn]
                    v7 = vertex_ids[ia + 1, ir + 1, it]
                    
                    # Split hex into 6 tets (standard decomposition)
                    cells.append([v0, v1, v3, v4])
                    cells.append([v1, v2, v3, v6])
                    cells.append([v1, v5, v4, v6])
                    cells.append([v3, v6, v4, v7])
                    cells.append([v1, v3, v4, v6])
                    cells.append([v1, v6, v4, v5])
        
        cell_type = "tetra"
    
    cells = np.array(cells, dtype=int)

    # =================================================================
    # Generate boundary faces and tags
    # =================================================================
    boundary_faces = []
    boundary_tags = []
    
    # Inner surface (tag=1)
    for ia in range(n_axial):
        for it in range(n_circ):
            itn = (it + 1) % n_circ
            v0 = vertex_ids[ia, 0, it]
            v1 = vertex_ids[ia + 1, 0, it]
            v2 = vertex_ids[ia + 1, 0, itn]
            v3 = vertex_ids[ia, 0, itn]
            
            if facet_type == "quad":
                boundary_faces.append([v0, v1, v2, v3])
                boundary_tags.append(1)
            else:
                boundary_faces.append([v0, v1, v2])
                boundary_faces.append([v0, v2, v3])
                boundary_tags.extend([1, 1])
    
    # Outer surface (tag=2)
    for ia in range(n_axial):
        for it in range(n_circ):
            itn = (it + 1) % n_circ
            v0 = vertex_ids[ia, n_radial, it]
            v1 = vertex_ids[ia, n_radial, itn]
            v2 = vertex_ids[ia + 1, n_radial, itn]
            v3 = vertex_ids[ia + 1, n_radial, it]
            
            if facet_type == "quad":
                boundary_faces.append([v0, v1, v2, v3])
                boundary_tags.append(2)
            else:
                boundary_faces.append([v0, v1, v2])
                boundary_faces.append([v0, v2, v3])
                boundary_tags.extend([2, 2])
    
    # End cap 1 at ia=0 (tag=3)
    for ir in range(n_radial):
        for it in range(n_circ):
            itn = (it + 1) % n_circ
            v0 = vertex_ids[0, ir, it]
            v1 = vertex_ids[0, ir, itn]
            v2 = vertex_ids[0, ir + 1, itn]
            v3 = vertex_ids[0, ir + 1, it]
            
            if facet_type == "quad":
                boundary_faces.append([v0, v1, v2, v3])
                boundary_tags.append(3)
            else:
                boundary_faces.append([v0, v1, v2])
                boundary_faces.append([v0, v2, v3])
                boundary_tags.extend([3, 3])
    
    # End cap 2 at ia=n_axial (tag=4)
    for ir in range(n_radial):
        for it in range(n_circ):
            itn = (it + 1) % n_circ
            v0 = vertex_ids[n_axial, ir, it]
            v1 = vertex_ids[n_axial, ir + 1, it]
            v2 = vertex_ids[n_axial, ir + 1, itn]
            v3 = vertex_ids[n_axial, ir, itn]
            
            if facet_type == "quad":
                boundary_faces.append([v0, v1, v2, v3])
                boundary_tags.append(4)
            else:
                boundary_faces.append([v0, v1, v2])
                boundary_faces.append([v0, v2, v3])
                boundary_tags.extend([4, 4])
    
    boundary_faces = np.array(boundary_faces, dtype=int)
    boundary_tags = np.array(boundary_tags, dtype=int)

    # =================================================================
    # Handle quadratic elements (element_order == 2)
    # =================================================================
    if element_order == 2:
        vertices, cells = _add_midpoint_nodes(vertices, cells, cell_type)
    
    return {
        "vertices": vertices,
        "cells": cells,
        "cell_type": cell_type,
        "element_order": element_order,
        "boundary_faces": boundary_faces,
        "boundary_tags": boundary_tags,
    }


def _add_midpoint_nodes(vertices, cells, cell_type):
    """
    Convert linear elements to quadratic by adding midpoint nodes.
    
    Parameters
    ----------
    vertices : ndarray (N, 3)
        Vertex coordinates
    cells : ndarray (M, nodes_per_linear_element)
        Linear element connectivity
    cell_type : str
        'tetra' or 'hexahedron'
    
    Returns
    -------
    new_vertices : ndarray
        Extended vertex array with midpoint nodes
    new_cells : ndarray
        Quadratic element connectivity (tet10 or hex20)
    """
    new_vertices = list(vertices)
    edge_to_midpoint = {}
    
    def get_midpoint(v1, v2):
        """Get or create midpoint node between two vertices."""
        edge = tuple(sorted([v1, v2]))
        if edge not in edge_to_midpoint:
            midpoint = (vertices[v1] + vertices[v2]) / 2.0
            edge_to_midpoint[edge] = len(new_vertices)
            new_vertices.append(midpoint)
        return edge_to_midpoint[edge]
    
    new_cells = []
    
    if cell_type == "tetra":
        # Tet10: 4 corner nodes + 6 edge midpoints
        for tet in cells:
            v0, v1, v2, v3 = tet
            # Edge midpoints
            m01 = get_midpoint(v0, v1)
            m02 = get_midpoint(v0, v2)
            m03 = get_midpoint(v0, v3)
            m12 = get_midpoint(v1, v2)
            m13 = get_midpoint(v1, v3)
            m23 = get_midpoint(v2, v3)
            
            new_cells.append([v0, v1, v2, v3, m01, m12, m02, m03, m13, m23])
    
    elif cell_type == "hexahedron":
        # Hex20: 8 corner nodes + 12 edge midpoints
        for hex_elem in cells:
            v0, v1, v2, v3, v4, v5, v6, v7 = hex_elem
            # Bottom face edges
            m01 = get_midpoint(v0, v1)
            m12 = get_midpoint(v1, v2)
            m23 = get_midpoint(v2, v3)
            m30 = get_midpoint(v3, v0)
            # Top face edges
            m45 = get_midpoint(v4, v5)
            m56 = get_midpoint(v5, v6)
            m67 = get_midpoint(v6, v7)
            m74 = get_midpoint(v7, v4)
            # Vertical edges
            m04 = get_midpoint(v0, v4)
            m15 = get_midpoint(v1, v5)
            m26 = get_midpoint(v2, v6)
            m37 = get_midpoint(v3, v7)
            
            new_cells.append([v0, v1, v2, v3, v4, v5, v6, v7,
                            m01, m12, m23, m30, m45, m56, m67, m74,
                            m04, m15, m26, m37])
    
    return np.array(new_vertices, dtype=float), np.array(new_cells, dtype=int)



if __name__ == "__main__":
    # Generate volumetric mesh
    mesh = generate_tube_mesh(
        axial_length=10.0,
        lumen_diameter=2.0,
        wall_thickness=0.5,
        n_axial=20,
        n_circ=16,
        n_radial=4,
        axis="z",
        facet_type="quad",  # 'quad' -> hex, 'tri' -> tet
        element_order=2,    # 1 -> linear, 2 -> quadratic
    )

    verts = mesh["vertices"]
    cells = mesh["cells"]
    cell_type = mesh["cell_type"]
    element_order = mesh["element_order"]
    boundary_faces = mesh["boundary_faces"]
    boundary_tags = mesh["boundary_tags"]

    print(f"Generated {cell_type} mesh with element_order={element_order}")
    print(f"Vertices: {verts.shape[0]}")
    print(f"Cells: {cells.shape[0]}")
    print(f"Nodes per cell: {cells.shape[1]}")
    print(f"Boundary faces: {boundary_faces.shape[0]}")

    # Create PyVista mesh for visualization
    if cell_type == "tetra":
        if element_order == 1:
            pv_mesh = pv.UnstructuredGrid({pv.CellType.TETRA: cells}, verts)
        else:  # element_order == 2
            pv_mesh = pv.UnstructuredGrid({pv.CellType.QUADRATIC_TETRA: cells}, verts)
    else:  # hexahedron
        if element_order == 1:
            pv_mesh = pv.UnstructuredGrid({pv.CellType.HEXAHEDRON: cells}, verts)
        else:  # element_order == 2
            pv_mesh = pv.UnstructuredGrid({pv.CellType.QUADRATIC_HEXAHEDRON: cells}, verts)
    
    # Create boundary mesh for visualization
    if boundary_faces.shape[1] == 3:  # triangular faces
        boundary_faces_with_count = np.column_stack([np.full(len(boundary_faces), 3), boundary_faces]).ravel()
    else:  # quad faces
        boundary_faces_with_count = np.column_stack([np.full(len(boundary_faces), 4), boundary_faces]).ravel()
    
    boundary_mesh = pv.PolyData(verts, boundary_faces_with_count)
    boundary_mesh.cell_data["boundary_tags"] = boundary_tags
    
    # Visualize
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, show_edges=True, opacity=0.3, color='lightgray', label='Volume')
    plotter.add_mesh(boundary_mesh, scalars="boundary_tags", show_edges=True,
                     cmap=['blue', 'red', 'green', 'magenta'],
                     categories=True, clim=[1, 4], opacity=0.8)
    plotter.add_legend([['Inner Surface (1)', 'blue'],
                        ['Outer Surface (2)', 'red'],
                        ['End Cap 1 (3)', 'green'],
                        ['End Cap 2 (4)', 'magenta']])
    plotter.add_axes()
    plotter.show_grid(xtitle='X', ytitle='Y', ztitle='Z')
    plotter.show()

    # Write mesh to XDMF format
    meshio_cell_type = {
        ("tetra", 1): "tetra",
        ("tetra", 2): "tetra10",
        ("hexahedron", 1): "hexahedron",
        ("hexahedron", 2): "hexahedron20",
    }[(cell_type, element_order)]
    
    meshio_boundary_type = "triangle" if boundary_faces.shape[1] == 3 else "quad"

    # Create meshio mesh object
    meshio_mesh = meshio.Mesh(
        points=verts,
        cells=[
            (meshio_cell_type, cells),
        ],
    )
    
    # Create boundary mesh for FEniCS
    boundary_meshio = meshio.Mesh(
        points=verts,
        cells=[(meshio_boundary_type, boundary_faces)],
        cell_data={"boundary_tags": [boundary_tags]}
    )

    # Write to XDMF files
    meshio_mesh.write("tube_mesh.xdmf")
    boundary_meshio.write("tube_mesh_boundary.xdmf")
    
    print("\nMesh written to tube_mesh.xdmf")
    print("Boundary mesh written to tube_mesh_boundary.xdmf")
