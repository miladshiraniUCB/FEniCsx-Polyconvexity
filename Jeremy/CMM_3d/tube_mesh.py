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
    n_radial_cap: int,
    axis: AxisType = "z",
    facet_type: FacetType = "quad",
) -> Dict[str, np.ndarray]:
    """
    Generate a hollow tube surface mesh with watertight end caps.

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
    n_radial_cap : int
        Number of radial subdivisions on each end cap annulus (>=1).
    axis : {'x','y','z'}
        Which axis the tube is aligned with.
    facet_type : {'quad','tri'}
        Type of mesh faces.

    Returns
    -------
    dict with keys:
        'vertices' : (N,3)
        'faces' : (M,4) or (M,3)
        'facet_type'
    """

    if n_axial < 1 or n_circ < 3 or n_radial_cap < 1:
        raise ValueError("n_axial>=1, n_circ>=3, n_radial_cap>=1 required.")

    if facet_type not in ("quad", "tri"):
        raise ValueError("facet_type must be 'quad' or 'tri'.")

    # Radii
    r_inner = lumen_diameter / 2.0
    r_outer = r_inner + wall_thickness
    axial_vals = np.linspace(0, axial_length, n_axial + 1)
    theta_vals = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)

    # Existing vertices: inner + outer cylindrical surfaces
    # -----------------------------------------------------
    def cyl_vertex(r, a, theta):
        x_c = r * np.cos(theta)
        y_c = r * np.sin(theta)
        if axis == "z":
            return np.array([x_c, y_c, a])
        elif axis == "x":
            return np.array([a, x_c, y_c])
        else:  # axis == "y"
            return np.array([x_c, a, y_c])

    vertices = []

    # Maps to indices after filling
    def add_vert(v):
        vertices.append(v)
        return len(vertices) - 1

    # Build cylindrical surface vertices
    inner_ids = np.zeros((n_axial + 1, n_circ), dtype=int)
    outer_ids = np.zeros((n_axial + 1, n_circ), dtype=int)

    for ia, a in enumerate(axial_vals):
        for it, theta in enumerate(theta_vals):
            inner_ids[ia, it] = add_vert(cyl_vertex(r_inner, a, theta))
            outer_ids[ia, it] = add_vert(cyl_vertex(r_outer, a, theta))

    faces = []

    # Helper for quads
    def add_quad(v0, v1, v2, v3):
        faces.append([v0, v1, v2, v3])

    # 1) Cylindrical surfaces (inner + outer)
    # --------------------------------------
    for ring in (inner_ids, outer_ids):
        for ia in range(n_axial):
            for it in range(n_circ):
                itn = (it + 1) % n_circ
                add_quad(
                    ring[ia, it],
                    ring[ia + 1, it],
                    ring[ia + 1, itn],
                    ring[ia, itn],
                )

    # 2) End caps (with radial subdivisions)
    # --------------------------------------
    # Radial interpolation radii
    radial_r = np.linspace(r_inner, r_outer, n_radial_cap + 1)

    def build_cap(ia):
        """
        Build one end cap surface at axial index ia (0 or n_axial).
        """
        # Generate annular grid points:
        # radial layer: 0 .. n_radial_cap (0=inner, end=n_radial_cap)
        cap_ids = np.zeros((n_radial_cap + 1, n_circ), dtype=int)

        for ir, r in enumerate(radial_r):
            for it, theta in enumerate(theta_vals):
                # If ir==0 and ir==n_radial_cap, reuse existing vertices
                if ir == 0:
                    cap_ids[ir, it] = inner_ids[ia, it]
                elif ir == n_radial_cap:
                    cap_ids[ir, it] = outer_ids[ia, it]
                else:
                    cap_ids[ir, it] = add_vert(cyl_vertex(r, axial_vals[ia], theta))

        # Build quads between radial layers
        for ir in range(n_radial_cap):
            for it in range(n_circ):
                itn = (it + 1) % n_circ
                v0 = cap_ids[ir, it]
                v1 = cap_ids[ir, itn]
                v2 = cap_ids[ir + 1, itn]
                v3 = cap_ids[ir + 1, it]
                add_quad(v0, v1, v2, v3)

    # Add front and back caps
    build_cap(0)
    build_cap(n_axial)

    vertices = np.array(vertices, float)
    faces = np.array(faces, int)

    # Store surface definitions
    surface_tags = np.zeros(faces.shape[0], dtype=int)
    
    # Track face indices
    face_idx = 0
    
    # Inner surface
    n_inner_faces = n_axial * n_circ
    surface_tags[face_idx:face_idx + n_inner_faces] = 1
    face_idx += n_inner_faces
    
    # Outer surface
    n_outer_faces = n_axial * n_circ
    surface_tags[face_idx:face_idx + n_outer_faces] = 2
    face_idx += n_outer_faces
    
    # End cap 1 (at ia=0)
    n_cap_faces = n_radial_cap * n_circ
    surface_tags[face_idx:face_idx + n_cap_faces] = 3
    face_idx += n_cap_faces
    
    # End cap 2 (at ia=n_axial)
    surface_tags[face_idx:face_idx + n_cap_faces] = 4

    # 3) Optional triangulation
    if facet_type == "tri":
        tris = np.zeros((2 * faces.shape[0], 3), dtype=int)
        tris[0::2] = faces[:, [0, 1, 2]]
        tris[1::2] = faces[:, [0, 2, 3]]
        faces = tris
        # Duplicate surface tags for triangulated faces (each quad becomes 2 triangles)
        surface_tags = np.repeat(surface_tags, 2)

    return {
        "vertices": vertices,
        "faces": faces,
        "facet_type": facet_type,
        "surface_tags": surface_tags,
    }



if __name__ == "__main__":
    mesh = generate_tube_mesh(
        axial_length=10.0,
        lumen_diameter=2.0,
        wall_thickness=0.5,
        n_radial_cap=4,
        n_axial=20,
        n_circ=32,
        axis="z",
        facet_type="quad",
    )

    faces = mesh["faces"]
    verts = mesh["vertices"]
    surface_tags = mesh["surface_tags"]

    # Create PyVista mesh
    if mesh["facet_type"] == "tri":
        # For triangular faces, prepend '3' to each face to indicate triangle
        faces_with_count = np.column_stack([np.full(len(faces), 3), faces]).ravel()
    else:  # quad
        # For quad faces, prepend '4' to each face to indicate quad
        faces_with_count = np.column_stack([np.full(len(faces), 4), faces]).ravel()
    
    pv_mesh = pv.PolyData(verts, faces_with_count)
    pv_mesh.cell_data["surface_tags"] = surface_tags
    
    # Define color map: tag 1=blue, 2=red, 3=green, 4=magenta
    color_map = {1: 'blue', 2: 'red', 3: 'green', 4: 'magenta'}
    colors = np.array([color_map[tag] for tag in surface_tags])
    
    # Visualize
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, scalars="surface_tags", show_edges=True, 
                     cmap=['blue', 'red', 'green', 'magenta'], 
                     categories=True, clim=[1, 4], opacity=0.8, scalar_bar_args={'title': 'Surface Tags'})
    plotter.add_legend([['Inner Surface (1)', 'blue'],
                        ['Outer Surface (2)', 'red'],
                        ['End Cap 1 (3)', 'green'],
                        ['End Cap 2 (4)', 'magenta']])
    plotter.add_axes()
    plotter.remove_scalar_bar()
    plotter.show_grid(xlabel='X', ylabel='Y', zlabel='Z')
    plotter.show()

    # Write mesh to XDMF format

    # Determine cell type based on facet_type
    if mesh["facet_type"] == "tri":
        cell_type = "triangle"
    else:
        cell_type = "quad"

    # Create meshio mesh object
    meshio_mesh = meshio.Mesh(
        points=verts,
        cells=[(cell_type, faces)],
        cell_data={"surface_tags": [surface_tags]}
    )

    # Write to XDMF file
    meshio_mesh.write("tube_mesh.xdmf")
    print("Mesh written to tube_mesh.xdmf")
    print("Vertices:", verts.shape)
    print("Faces:", faces.shape)
