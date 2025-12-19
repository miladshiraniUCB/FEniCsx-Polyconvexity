#!/usr/bin/env python3
"""
Check the actual boundary face definitions in the mesh file.
"""
import h5py
import numpy as np

# Read the mesh HDF5 file
print("=== MESH FILE ===")
with h5py.File('tube_mesh.h5', 'r') as f:
    datasets = list(f.keys())
    print(f"Datasets: {datasets}")
    
    geom = f['data0'][:]  # Geometry (nodes)
    cells = f['data1'][:]  # Hex elements
    
    print(f"Geometry shape: {geom.shape}")
    print(f"Cells shape: {cells.shape}")
    print(f"First 3 nodes:\n{geom[:3]}")
    print(f"First 2 hex elements:\n{cells[:2]}")

# Read boundary file
print("\n=== BOUNDARY FILE ===")
with h5py.File('tube_mesh_boundary.h5', 'r') as f:
    datasets = list(f.keys())
    print(f"Datasets: {datasets}")
    
    # data0 = geometry (same as mesh)
    # data1 = boundary face connectivity (quads)
    # data2 = boundary tags
    
    boundary_faces = f['data1'][:]
    boundary_tags = f['data2'][:]
    
    print(f"Boundary faces shape: {boundary_faces.shape}")
    print(f"Boundary tags shape: {boundary_tags.shape}")
    print(f"Unique tags: {np.unique(boundary_tags)}")
    
    # Find inner surface faces (tag 1)
    inner_indices = np.where(boundary_tags == 1)[0]
    print(f"\nInner surface (tag 1): {len(inner_indices)} faces")
    print(f"First 5 inner face node indices:\n{boundary_faces[inner_indices[:5]]}")
    
    # Check first inner face
    face_idx = inner_indices[0]
    face_nodes = boundary_faces[face_idx]
    print(f"\n=== Analyzing first inner surface face ===")
    print(f"Node indices: {face_nodes}")
    
    # Get node coordinates from mesh file
    with h5py.File('tube_mesh.h5', 'r') as f_mesh:
        geom = f_mesh['data0'][:]
        coords = geom[face_nodes]
        
        print(f"Node coordinates:")
        for i, (node_idx, coord) in enumerate(zip(face_nodes, coords)):
            r = np.sqrt(coord[0]**2 + coord[1]**2)
            print(f"  v{i} (global {node_idx}): [{coord[0]:7.4f}, {coord[1]:7.4f}, {coord[2]:7.4f}]  r={r:.4f}")
        
        # Compute normal from vertex ordering
        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[0]
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        
        center = np.mean(coords, axis=0)
        r_center = np.sqrt(center[0]**2 + center[1]**2)
        inward_dir = -center / r_center  # Points toward axis
        inward_dir = np.append(inward_dir[:2], 0)  # Make it 3D
        
        print(f"\nFace center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
        print(f"Computed normal (v0→v1 × v0→v2): [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]")
        print(f"Inward direction (toward axis): [{inward_dir[0]:.4f}, {inward_dir[1]:.4f}, {inward_dir[2]:.4f}]")
        print(f"Dot product (normal · inward): {np.dot(normal, inward_dir):.4f}")
        
        if np.dot(normal, inward_dir) > 0:
            print("  ✓ Normal points INWARD (toward axis) - CORRECT for inner surface")
        else:
            print("  ✗ Normal points OUTWARD (away from axis) - WRONG for inner surface!")
