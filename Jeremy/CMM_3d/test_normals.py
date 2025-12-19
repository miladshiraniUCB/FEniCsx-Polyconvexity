#!/usr/bin/env python3
"""
Test script to verify boundary face normal directions.
Creates a simple hex element and checks face normals.
"""
import numpy as np

# Create a simple hex element for testing
# Positioned like one element from the tube mesh
r_inner = 1.0
r_outer = 1.5
z0 = 0.0
z1 = 1.0
theta0 = 0.0
theta1 = np.pi / 4

# Convert cylindrical to Cartesian
def cyl_to_cart(r, theta, z):
    return np.array([r * np.cos(theta), r * np.sin(theta), z])

# Create 8 vertices of hex element
v0 = cyl_to_cart(r_inner, theta0, z0)  # inner, lower z, theta0
v1 = cyl_to_cart(r_outer, theta0, z0)  # outer, lower z, theta0
v2 = cyl_to_cart(r_outer, theta0, z1)  # outer, upper z, theta0
v3 = cyl_to_cart(r_inner, theta0, z1)  # inner, upper z, theta0
v4 = cyl_to_cart(r_inner, theta1, z0)  # inner, lower z, theta1
v5 = cyl_to_cart(r_outer, theta1, z0)  # outer, lower z, theta1
v6 = cyl_to_cart(r_outer, theta1, z1)  # outer, upper z, theta1
v7 = cyl_to_cart(r_inner, theta1, z1)  # inner, upper z, theta1

vertices = np.array([v0, v1, v2, v3, v4, v5, v6, v7])

print("Hex element vertices:")
for i, v in enumerate(vertices):
    print(f"  v{i}: {v}")

# Element center
center = np.mean(vertices, axis=0)
print(f"\nElement center: {center}")

def compute_face_normal(face_vertices):
    """Compute normal using cross product of two edge vectors."""
    # face_vertices should be ordered for right-hand rule
    v1 = face_vertices[1] - face_vertices[0]
    v2 = face_vertices[2] - face_vertices[0]
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)

def check_face(name, vertex_indices, expected_direction):
    """Check if face normal points in expected direction."""
    face_verts = vertices[vertex_indices]
    face_center = np.mean(face_verts, axis=0)
    normal = compute_face_normal(face_verts)
    
    # Vector from element center to face center
    outward = face_center - center
    outward = outward / np.linalg.norm(outward)
    
    # Check if normal aligns with outward direction
    dot_product = np.dot(normal, outward)
    
    print(f"\n{name}:")
    print(f"  Vertex indices: {vertex_indices}")
    print(f"  Face center: {face_center}")
    print(f"  Normal: {normal}")
    print(f"  Outward direction: {outward}")
    print(f"  Dot product: {dot_product:.3f}")
    print(f"  Expected: {expected_direction}")
    
    if dot_product > 0.9:
        print(f"  ✓ Normal points OUTWARD from element")
    elif dot_product < -0.9:
        print(f"  ✗ Normal points INWARD to element (WRONG!)")
    else:
        print(f"  ? Normal is not aligned with radial direction")
    
    return dot_product > 0

# Test different face orderings for inner surface
print("\n" + "="*60)
print("TESTING INNER SURFACE (should point toward axis, i.e., inward)")
print("="*60)

# Current code uses: [v0, v4, v7, v3]
check_face("Current: [v0, v4, v7, v3]", [0, 4, 7, 3], "toward axis (inward)")

# Try alternative: [v0, v3, v7, v4]
check_face("Alternative: [v0, v3, v7, v4]", [0, 3, 7, 4], "toward axis (inward)")

# Try: [v3, v0, v4, v7]
check_face("Alternative: [v3, v0, v4, v7]", [3, 0, 4, 7], "toward axis (inward)")

print("\n" + "="*60)
print("TESTING OUTER SURFACE (should point away from axis, i.e., outward)")
print("="*60)

# Current code uses: [v1, v5, v6, v2]
check_face("Current: [v1, v5, v6, v2]", [1, 5, 6, 2], "away from axis (outward)")

# Try alternative: [v1, v2, v6, v5]
check_face("Alternative: [v1, v2, v6, v5]", [1, 2, 6, 5], "away from axis (outward)")

print("\n" + "="*60)
print("TESTING INLET CAP (should point in -z direction)")
print("="*60)

# Current code uses: [v0, v4, v5, v1]
check_face("Current: [v0, v4, v5, v1]", [0, 4, 5, 1], "-z direction")

# Try alternative: [v0, v1, v5, v4]
check_face("Alternative: [v0, v1, v5, v4]", [0, 1, 5, 4], "-z direction")

print("\n" + "="*60)
print("TESTING OUTLET CAP (should point in +z direction)")
print("="*60)

# Current code uses: [v3, v7, v6, v2]
check_face("Current: [v3, v7, v6, v2]", [3, 7, 6, 2], "+z direction")

# Try alternative: [v3, v2, v6, v7]
check_face("Alternative: [v3, v2, v6, v7]", [3, 2, 6, 7], "+z direction")
