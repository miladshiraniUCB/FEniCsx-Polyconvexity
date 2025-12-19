import numpy as np
from typing import Union
from dolfinx.plot import vtk_mesh
import pyvista

def mesh3d_dict(
    xyz: np.ndarray,
    ijk: np.ndarray,
    facecolor: Union[list[str], str] = "black",
    opacity: float = 1.0,
    flatshade: bool = True,
):
    """Create a dict for a Mesh3D object that can be plotted with plotly.

    Parameters
    ----------
    xyz : np.ndarray
        Vertices of a mesh.
    ijk : np.ndarray
        Triangle faces of a mesh.
    facecolor : {list[str], str}, default="black"
        List of strings or a single string specifying the color of a face
    opacity : float, default=1.0
        Opacity of the mesh.
    flatshade : bool, default=True
        Flat shading if True.

    Returns
    -------
    dict
        Can be passed to a plotly figure for plotting
    """
    return dict(
        type="mesh3d",
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        i=ijk[:, 0],
        j=ijk[:, 1],
        k=ijk[:, 2],
        opacity=opacity,
        facecolor=facecolor,
        flatshading=flatshade,
    )


def plot_scalar(msh, u):
    # We start by creating a unit square mesh and interpolating a
    # function into a degree 1 Lagrange space
    # msh = create_unit_square(
    #     MPI.COMM_WORLD, 12, 12, cell_type=CellType.quadrilateral, dtype=np.float64
    # )
    # V = functionspace(msh, ("Lagrange", 1))
    # u = Function(V, dtype=np.float64)

    # u.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(2 * x[1] * np.pi))

    # # To visualize the function u, we create a VTK-compatible grid to
    # # values of u to
    # cells, types, x = pyvista.plot.vtk_mesh(V)
    # grid = pyvista.UnstructuredGrid(cells, types, x)
    # grid.point_data["u"] = u.x.array
    
    topology, cell_types, x = vtk_mesh(msh)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)
    grid.point_data["u"] = u.x.array


    # # Create PyVista unstructured grid from input cells and nodes
    # cell_types = np.full(len(cells), pyvista.CellType.TRIANGLE, dtype=np.uint8)
    # cells_pyvista = np.column_stack([np.full(len(cells), 3), cells]).ravel()
    # grid = pyvista.UnstructuredGrid(cells_pyvista, cell_types, nodes)
    
    # # Set scalar values from u
    # grid.point_data["u"] = u



    # The function "u" is set as the active scalar for the mesh, and
    # warp in z-direction is set
    grid.set_active_scalars("u")
    warped = grid.warp_by_scalar()

    # A plotting window is created with two sub-plots, one of the scalar
    # values and the other of the mesh is warped by the scalar values in
    # z-direction
    subplotter = pyvista.Plotter(shape=(1, 2))
    subplotter.subplot(0, 0)
    subplotter.add_text("Scalar contour field", font_size=14, color="black", position="upper_edge")
    subplotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
    subplotter.view_xy()

    subplotter.subplot(0, 1)
    subplotter.add_text("Warped function", position="upper_edge", font_size=14, color="black")
    sargs = dict(
        height=0.8,
        width=0.1,
        vertical=True,
        position_x=0.05,
        position_y=0.05,
        fmt="%1.2e",
        title_font_size=40,
        color="black",
        label_font_size=25,
    )
    subplotter.set_position([-3, 2.6, 0.3])
    subplotter.set_focus([3, -1, -0.15])
    subplotter.set_viewup([0, 0, 1])
    subplotter.add_mesh(warped, show_edges=True, scalar_bar_args=sargs)
    # if pyvista.OFF_SCREEN:
    #     subplotter.screenshot(
    #         out_folder / "2D_function_warp.png",
    #         transparent_background=transparent,
    #         window_size=[figsize, figsize],
    #     )
    # else:
    subplotter.show()