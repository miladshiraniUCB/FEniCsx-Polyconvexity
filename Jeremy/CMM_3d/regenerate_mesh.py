#!/usr/bin/env python3
"""
Regenerate the tube mesh with corrected boundary face orientations.
"""
import numpy as np
import meshio
import sys
from tube_mesh_vol2 import generate_tube_volume_mesh

# # Import the function from tube_mesh_vol2.py
# exec(open('tube_mesh_vol2.py').read().split('if __name__')[0])

# Fiber families for constrained mixture model
alpha0_deg = 29.91  # degrees
alpha0 = alpha0_deg * np.pi / 180.0  # radians

print("Generating mesh with corrected boundary face orientations...")

mesh = generate_tube_volume_mesh(
    axial_length=10.0,
    lumen_diameter=2.0,
    wall_thickness=0.5,
    n_axial=10,
    n_circ=24,
    n_radial=3,
    axis='z',
    mesh_type='hex8',
    fiber_angles={
        'theta': np.pi / 2,
        'axial': 0.0,
        'diagonal1': alpha0,
        'diagonal2': -alpha0,
    },
)

nodes = mesh['nodes']
cells = mesh['elements']
mesh_type = mesh['element_type']

meshio_cell_type = {
    'hex8': 'hexahedron',
    'hex20': 'hexahedron20',
    'tet4': 'tetra',
    'tet10': 'tetra10',
}[mesh_type]

meshio_boundary_type = 'triangle' if mesh['boundary_faces'].shape[1] == 3 else 'quad'

# Create cell data dictionary
cell_data_dict = {
    'boundary_tags': [
        np.zeros(cells.shape[0], dtype=int),
        mesh['boundary_tags']
    ]
}

for family_name, fiber_vectors in mesh['fiber_families'].items():
    cell_data_dict[f'fiber_{family_name}'] = [
        fiber_vectors,
        np.zeros((mesh['boundary_faces'].shape[0], 3), dtype=float)
    ]

meshio_mesh = meshio.Mesh(
    points=nodes,
    cells=[
        (meshio_cell_type, cells),
        (meshio_boundary_type, mesh['boundary_faces']),
    ],
    cell_data=cell_data_dict
)

# Write mesh
print("Writing tube_mesh.xdmf...")
meshio_mesh.write('tube_mesh.xdmf')

# Write separate boundary file
boundary_mesh = meshio.Mesh(
    points=nodes,
    cells=[(meshio_boundary_type, mesh['boundary_faces'])],
    cell_data={'boundary_tags': [mesh['boundary_tags']]}
)

print("Writing tube_mesh_boundary.xdmf...")
boundary_mesh.write('tube_mesh_boundary.xdmf')

# Write separate fiber file
fiber_mesh = meshio.Mesh(
    points=nodes,
    cells=[(meshio_cell_type, cells)],
    cell_data={f'fiber_{name}': [vectors] for name, vectors in mesh['fiber_families'].items()}
)

print("Writing tube_mesh_fibers.xdmf...")
fiber_mesh.write('tube_mesh_fibers.xdmf')

print(f'\nMesh generation complete!')
print(f'  Volume elements: {cells.shape[0]} {mesh_type}')
print(f'  Boundary faces: {mesh["boundary_faces"].shape[0]} {meshio_boundary_type}')
print(f'  Fiber families: {list(mesh["fiber_families"].keys())}')
print(f'  Boundary tags: {np.unique(mesh["boundary_tags"])}')
