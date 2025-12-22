from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np


# -----------------------------
# Helpers (geometry + indexing)
# -----------------------------

def _assert_axis(axis: str) -> str:
    assert axis in ("x", "y", "z"), "axis must be one of {'x','y','z'}"
    return axis


def _assert_mesh_type(mesh_type: str) -> Tuple[str, int, str]:
    """
    Returns (element_type, order, cell_kind)
      element_type: 'hex8','hex20','tet4','tet10'
      order: 1 or 2
      cell_kind: 'hex' or 'tet'
    """
    assert mesh_type in ("tet4", "tet10", "hex8", "hex20"), \
        "mesh_type must be one of {'tet4','tet10','hex8','hex20'}"
    order = 2 if mesh_type in ("tet10", "hex20") else 1
    cell_kind = "tet" if mesh_type.startswith("tet") else "hex"
    return mesh_type, order, cell_kind


def _map_local_to_global(u: float, v: float, a: float, axis: str) -> Tuple[float, float, float]:
    """
    Local coordinates:
      - cross-section plane is (u,v)
      - axial coordinate is a
    Map to global so tube is aligned with chosen axis.
    """
    if axis == "z":
        return (u, v, a)
    if axis == "x":
        return (a, u, v)  # axial -> x, cross-section -> (y,z)
    # axis == "y"
    return (u, a, v)      # axial -> y, cross-section -> (x,z)


def _theta_from_xyz(xyz: np.ndarray, axis: str) -> np.ndarray:
    """
    Compute polar angle theta at points xyz in the tube cross-section plane.
    """
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    if axis == "z":
        return np.arctan2(y, x)
    if axis == "x":
        # cross-section plane is (y,z)
        return np.arctan2(z, y)
    # axis == "y": cross-section plane is (x,z)
    return np.arctan2(z, x)


def _unit_vectors_theta_axial(theta: np.ndarray, axis: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (e_theta, e_axial), each (M,3).
    """
    M = theta.shape[0]
    e_theta = np.zeros((M, 3), dtype=float)
    e_axial = np.zeros((M, 3), dtype=float)

    if axis == "z":
        # e_theta = (-sin, cos, 0), e_axial = (0,0,1)
        e_theta[:, 0] = -np.sin(theta)
        e_theta[:, 1] = np.cos(theta)
        e_axial[:, 2] = 1.0
    elif axis == "x":
        # cross-section in (y,z): e_theta=(0,-sin,cos), e_axial=(1,0,0)
        e_theta[:, 1] = -np.sin(theta)
        e_theta[:, 2] = np.cos(theta)
        e_axial[:, 0] = 1.0
    else:  # axis == "y"
        # cross-section in (x,z): e_theta=(-sin,0,cos), e_axial=(0,1,0)
        e_theta[:, 0] = -np.sin(theta)
        e_theta[:, 2] = np.cos(theta)
        e_axial[:, 1] = 1.0

    return e_theta, e_axial


def _idx(ir: int, ith: int, iz: int, n_radial: int, n_circ: int, n_axial: int) -> int:
    """
    Vertex node indexing for structured grid:
      ir in [0..n_radial]
      ith in [0..n_circ-1] (periodic seam)
      iz in [0..n_axial]
    Layout: ((iz * n_circ + ith) * (n_radial+1) + ir)
    """
    return ((iz * n_circ + ith) * (n_radial + 1) + ir)


def _hex8_edges(v: np.ndarray) -> List[Tuple[int, int]]:
    """
    v is length-8 vertex list in gmsh hex8 ordering:
      1:(0,0,0) 2:(1,0,0) 3:(1,1,0) 4:(0,1,0)
      5:(0,0,1) 6:(1,0,1) 7:(1,1,1) 8:(0,1,1)
    Returns the 12 edges as (i,j) pairs of vertex indices (global node ids).
    """
    # edges in terms of positions 0..7
    E = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical
    ]
    return [(int(v[i]), int(v[j])) for (i, j) in E]


def _tet4_edges(v: np.ndarray) -> List[Tuple[int, int]]:
    """
    v is length-4 vertex list.
    Returns 6 edges.
    """
    E = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    return [(int(v[i]), int(v[j])) for (i, j) in E]


def _make_midpoint_nodes(nodes: np.ndarray, elements_lin: np.ndarray, cell_kind: str) -> Tuple[np.ndarray, Dict[Tuple[int, int], int]]:
    """
    Given linear elements, create unique midpoint nodes on all edges.
    Returns (new_nodes, edge_mid_map).
    edge_mid_map[(min_i,max_i)] -> new node index (0-based in new_nodes).
    """
    edge_mid: Dict[Tuple[int, int], int] = {}
    new_nodes = nodes.tolist()  # list of [x,y,z]

    for elem in elements_lin:
        if cell_kind == "hex":
            edges = _hex8_edges(elem)
        else:
            edges = _tet4_edges(elem)

        for a, b in edges:
            key = (a, b) if a < b else (b, a)
            if key in edge_mid:
                continue
            pa = nodes[key[0]]
            pb = nodes[key[1]]
            pm = 0.5 * (pa + pb)
            edge_mid[key] = len(new_nodes)
            new_nodes.append(pm.tolist())

    return np.asarray(new_nodes, dtype=float), edge_mid


def _ensure_positive_tet_orientation(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """
    Ensure each tet has positive signed volume; if negative, swap two vertices.
    """
    tets = tets.copy()
    X = nodes
    for i in range(tets.shape[0]):
        a, b, c, d = tets[i]
        pa, pb, pc, pd = X[a], X[b], X[c], X[d]
        # signed volume ~ det([pb-pa, pc-pa, pd-pa])
        J = np.stack([pb - pa, pc - pa, pd - pa], axis=1)
        vol6 = np.linalg.det(J)
        if vol6 < 0:
            # swap c and d
            tets[i, 2], tets[i, 3] = tets[i, 3], tets[i, 2]
    return tets


# -----------------------------
# 1) Mesh generator
# -----------------------------

def generate_walled_tube_mesh_with_fibers(
    axial_length: float,
    lumen_diameter: float,
    wall_thickness: float,
    n_axial: int,
    n_circ: int,
    n_radial: int,
    axis: str = "z",
    mesh_type: str = "hex8",
    fiber_angles: Optional[Dict[str, float]] = None,
) -> dict:
    """
    Generate a structured volumetric mesh of a hollow tube (inner radius -> outer radius),
    optionally quadratic (tet10/hex20), and compute per-element fiber vectors for one or
    more fiber families.

    Notes:
      * Structured in (r, theta, axial) then mapped to Cartesian.
      * Circumference is periodic with a seam at theta=0/2π.
      * Boundary tags:
          1: inner surface
          2: outer surface
          3: end cap at axial=0
          4: end cap at axial=axial_length

    Returns:
      mesh_dict = {
        'nodes': (N,3) float,
        'elements': (M,nnpe) int (0-based),
        'element_type': str,
        'element_order': int,
        'fiber_families': dict[str,(M,3)] float,
        'boundary_faces': (K,npf) int (0-based),
        'boundary_tags': (K,) int
      }
    """
    axis = _assert_axis(axis)
    element_type, element_order, cell_kind = _assert_mesh_type(mesh_type)

    assert axial_length > 0
    assert lumen_diameter > 0
    assert wall_thickness > 0
    assert n_axial >= 1
    assert n_circ >= 3
    assert n_radial >= 1

    if fiber_angles is None:
        fiber_angles = {"axial": 0.0}

    ri = 0.5 * lumen_diameter
    ro = ri + wall_thickness

    # --- Vertex nodes (linear grid nodes) ---
    # Keep theta nodes = n_circ (no duplicate at 2π).
    rs = np.linspace(ri, ro, n_radial + 1)
    thetas = 2.0 * np.pi * np.arange(n_circ) / n_circ
    ax = np.linspace(0.0, axial_length, n_axial + 1)

    nodes = np.zeros(((n_radial + 1) * n_circ * (n_axial + 1), 3), dtype=float)

    for iz, a in enumerate(ax):
        for ith, th in enumerate(thetas):
            ct, st = np.cos(th), np.sin(th)
            for ir, r in enumerate(rs):
                u = r * ct
                v = r * st
                x, y, z = _map_local_to_global(u, v, a, axis)
                nodes[_idx(ir, ith, iz, n_radial, n_circ, n_axial)] = (x, y, z)

    # --- Linear elements ---
    if cell_kind == "hex":
        # hex8 per (ir, ith, iz) cell
        hexes = []
        for iz in range(n_axial):
            for ith in range(n_circ):
                ith1 = (ith + 1) % n_circ
                for ir in range(n_radial):
                    v1 = _idx(ir,   ith,  iz,   n_radial, n_circ, n_axial)
                    v2 = _idx(ir+1, ith,  iz,   n_radial, n_circ, n_axial)
                    v3 = _idx(ir+1, ith1, iz,   n_radial, n_circ, n_axial)
                    v4 = _idx(ir,   ith1, iz,   n_radial, n_circ, n_axial)
                    v5 = _idx(ir,   ith,  iz+1, n_radial, n_circ, n_axial)
                    v6 = _idx(ir+1, ith,  iz+1, n_radial, n_circ, n_axial)
                    v7 = _idx(ir+1, ith1, iz+1, n_radial, n_circ, n_axial)
                    v8 = _idx(ir,   ith1, iz+1, n_radial, n_circ, n_axial)
                    hexes.append([v1, v2, v3, v4, v5, v6, v7, v8])
        elements_lin = np.asarray(hexes, dtype=np.int64)
    else:
        # Build tets by subdividing each logical hex into 5 tets along diagonal (v1->v7)
        tets = []
        for iz in range(n_axial):
            for ith in range(n_circ):
                ith1 = (ith + 1) % n_circ
                for ir in range(n_radial):
                    v1 = _idx(ir,   ith,  iz,   n_radial, n_circ, n_axial)
                    v2 = _idx(ir+1, ith,  iz,   n_radial, n_circ, n_axial)
                    v3 = _idx(ir+1, ith1, iz,   n_radial, n_circ, n_axial)
                    v4 = _idx(ir,   ith1, iz,   n_radial, n_circ, n_axial)
                    v5 = _idx(ir,   ith,  iz+1, n_radial, n_circ, n_axial)
                    v6 = _idx(ir+1, ith,  iz+1, n_radial, n_circ, n_axial)
                    v7 = _idx(ir+1, ith1, iz+1, n_radial, n_circ, n_axial)
                    v8 = _idx(ir,   ith1, iz+1, n_radial, n_circ, n_axial)

                    # 5-tet split
                    tets.extend([
                        [v1, v2, v3, v7],
                        [v1, v3, v4, v7],
                        [v1, v4, v8, v7],
                        [v1, v8, v5, v7],
                        [v1, v5, v6, v7],
                    ])
        elements_lin = np.asarray(tets, dtype=np.int64)
        elements_lin = _ensure_positive_tet_orientation(nodes, elements_lin)

    # --- Upgrade to quadratic if requested ---
    if element_order == 1:
        elements = elements_lin
        nodes_q = nodes
        edge_mid = None
    else:
        nodes_q, edge_mid = _make_midpoint_nodes(nodes, elements_lin, cell_kind=cell_kind)

        if cell_kind == "hex":
            # gmsh hex20 ordering:
            #  1..8 vertices, then edges:
            #  (1-2,2-3,3-4,4-1, 5-6,6-7,7-8,8-5, 1-5,2-6,3-7,4-8)
            elems20 = np.zeros((elements_lin.shape[0], 20), dtype=np.int64)
            for i, e in enumerate(elements_lin):
                v = e  # length 8
                def mid(a, b) -> int:
                    key = (a, b) if a < b else (b, a)
                    return edge_mid[key]  # type: ignore[index]

                m12 = mid(v[0], v[1]); m23 = mid(v[1], v[2]); m34 = mid(v[2], v[3]); m41 = mid(v[3], v[0])
                m56 = mid(v[4], v[5]); m67 = mid(v[5], v[6]); m78 = mid(v[6], v[7]); m85 = mid(v[7], v[4])
                m15 = mid(v[0], v[4]); m26 = mid(v[1], v[5]); m37 = mid(v[2], v[6]); m48 = mid(v[3], v[7])

                elems20[i, :] = np.array(
                    [v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                     m12, m23, m34, m41, m56, m67, m78, m85, m15, m26, m37, m48],
                    dtype=np.int64
                )
            elements = elems20
        else:
            # gmsh tet10 ordering:
            # 1..4 vertices, then edges:
            # (1-2,2-3,3-1,1-4,2-4,3-4)
            elems10 = np.zeros((elements_lin.shape[0], 10), dtype=np.int64)
            for i, e in enumerate(elements_lin):
                v = e  # length 4
                def mid(a, b) -> int:
                    key = (a, b) if a < b else (b, a)
                    return edge_mid[key]  # type: ignore[index]

                m12 = mid(v[0], v[1])
                m23 = mid(v[1], v[2])
                m31 = mid(v[2], v[0])
                m14 = mid(v[0], v[3])
                m24 = mid(v[1], v[3])
                m34 = mid(v[2], v[3])

                elems10[i, :] = np.array([v[0], v[1], v[2], v[3], m12, m23, m31, m14, m24, m34], dtype=np.int64)

            elements = elems10

        nodes = nodes_q  # upgrade nodes

    # --- Boundary faces (quadratic consistent if order=2) ---
    boundary_faces: List[List[int]] = []
    boundary_tags: List[int] = []

    def _quad8_from_quad4(q4: List[int]) -> List[int]:
        assert edge_mid is not None
        v1, v2, v3, v4 = q4
        def mid(a, b) -> int:
            key = (a, b) if a < b else (b, a)
            return edge_mid[key]
        m12 = mid(v1, v2); m23 = mid(v2, v3); m34 = mid(v3, v4); m41 = mid(v4, v1)
        # gmsh quad8 ordering: 1..4 vertices, then (1-2,2-3,3-4,4-1)
        return [v1, v2, v3, v4, m12, m23, m34, m41]

    def _tri6_from_tri3(t3: List[int]) -> List[int]:
        assert edge_mid is not None
        a, b, c = t3
        def mid(u, v) -> int:
            key = (u, v) if u < v else (v, u)
            return edge_mid[key]
        mab = mid(a, b); mbc = mid(b, c); mca = mid(c, a)
        # gmsh tri6 ordering: 1..3 vertices, then (1-2,2-3,3-1)
        return [a, b, c, mab, mbc, mca]

    # Inner/Outer cylindrical surfaces
    for iz in range(n_axial):
        for ith in range(n_circ):
            ith1 = (ith + 1) % n_circ

            # inner (ir=0): quad in (theta, axial)
            q_inner = [
                _idx(0, ith,  iz,   n_radial, n_circ, n_axial),
                _idx(0, ith1, iz,   n_radial, n_circ, n_axial),
                _idx(0, ith1, iz+1, n_radial, n_circ, n_axial),
                _idx(0, ith,  iz+1, n_radial, n_circ, n_axial),
            ]
            # outer (ir=n_radial)
            q_outer = [
                _idx(n_radial, ith,  iz,   n_radial, n_circ, n_axial),
                _idx(n_radial, ith,  iz+1, n_radial, n_circ, n_axial),
                _idx(n_radial, ith1, iz+1, n_radial, n_circ, n_axial),
                _idx(n_radial, ith1, iz,   n_radial, n_circ, n_axial),
            ]

            if cell_kind == "hex":
                boundary_faces.append(_quad8_from_quad4(q_inner) if element_order == 2 else q_inner)
                boundary_tags.append(1)
                boundary_faces.append(_quad8_from_quad4(q_outer) if element_order == 2 else q_outer)
                boundary_tags.append(2)
            else:
                # split each quad into two triangles
                tris_inner = [[q_inner[0], q_inner[1], q_inner[2]], [q_inner[0], q_inner[2], q_inner[3]]]
                tris_outer = [[q_outer[0], q_outer[1], q_outer[2]], [q_outer[0], q_outer[2], q_outer[3]]]
                for t in tris_inner:
                    boundary_faces.append(_tri6_from_tri3(t) if element_order == 2 else t)
                    boundary_tags.append(1)
                for t in tris_outer:
                    boundary_faces.append(_tri6_from_tri3(t) if element_order == 2 else t)
                    boundary_tags.append(2)

    # End caps at axial = 0 and axial = L
    for ith in range(n_circ):
        ith1 = (ith + 1) % n_circ
        for ir in range(n_radial):
            # cap at iz=0
            q_cap0 = [
                _idx(ir,   ith,  0, n_radial, n_circ, n_axial),
                _idx(ir+1, ith,  0, n_radial, n_circ, n_axial),
                _idx(ir+1, ith1, 0, n_radial, n_circ, n_axial),
                _idx(ir,   ith1, 0, n_radial, n_circ, n_axial),
            ]
            # cap at iz=n_axial
            q_capL = [
                _idx(ir,   ith,  n_axial, n_radial, n_circ, n_axial),
                _idx(ir,   ith1, n_axial, n_radial, n_circ, n_axial),
                _idx(ir+1, ith1, n_axial, n_radial, n_circ, n_axial),
                _idx(ir+1, ith,  n_axial, n_radial, n_circ, n_axial),
            ]

            if cell_kind == "hex":
                boundary_faces.append(_quad8_from_quad4(q_cap0) if element_order == 2 else q_cap0)
                boundary_tags.append(3)
                boundary_faces.append(_quad8_from_quad4(q_capL) if element_order == 2 else q_capL)
                boundary_tags.append(4)
            else:
                tris0 = [[q_cap0[0], q_cap0[1], q_cap0[2]], [q_cap0[0], q_cap0[2], q_cap0[3]]]
                trisL = [[q_capL[0], q_capL[1], q_capL[2]], [q_capL[0], q_capL[2], q_capL[3]]]
                for t in tris0:
                    boundary_faces.append(_tri6_from_tri3(t) if element_order == 2 else t)
                    boundary_tags.append(3)
                for t in trisL:
                    boundary_faces.append(_tri6_from_tri3(t) if element_order == 2 else t)
                    boundary_tags.append(4)

    boundary_faces_arr = np.asarray(boundary_faces, dtype=np.int64)
    boundary_tags_arr = np.asarray(boundary_tags, dtype=np.int64)

    # --- Fibers per element ---
    # Element center angle from average of vertex coordinates (first 8 for hex, first 4 for tet)
    if cell_kind == "hex":
        verts = elements[:, :8]
    else:
        verts = elements[:, :4]

    centers = nodes[verts].mean(axis=1)  # (M,3)
    theta_e = _theta_from_xyz(centers, axis=axis)
    e_theta, e_axial = _unit_vectors_theta_axial(theta_e, axis=axis)

    fiber_families: Dict[str, np.ndarray] = {}
    for name, ang in fiber_angles.items():
        f = np.cos(ang) * e_axial + np.sin(ang) * e_theta
        # normalize (should already be unit, but safe)
        nrm = np.linalg.norm(f, axis=1, keepdims=True)
        f = f / np.maximum(nrm, 1e-16)
        fiber_families[name] = f

    nnpe = elements.shape[1]
    mesh_dict = {
        "nodes": nodes.astype(float),
        "elements": elements.astype(np.int64),
        "element_type": element_type,
        "element_order": int(element_order),
        "fiber_families": fiber_families,
        "boundary_faces": boundary_faces_arr,
        "boundary_tags": boundary_tags_arr,
    }
    return mesh_dict


# -----------------------------
# 2) Write mesh + fibers for dolfinx
# -----------------------------

def write_tube_mesh_and_fibers_for_dolfinx(
    mesh: dict,
    msh_path: str,
    fibers_npz_path: Optional[str] = None,
) -> None:
    """
    Writes:
      - a Gmsh v4+ .msh file containing the volume mesh and tagged boundary facets
      - (optional) a .npz containing per-element fiber vectors for each family

    The .msh can be read by dolfinx via:
      from dolfinx.io import gmsh as gmshio
      domain, cell_tags, facet_tags = gmshio.read_from_msh("tube.msh", comm, gdim=3)

    The fiber .npz can be loaded with numpy and then assigned to DG0 vector Functions
    (see tube_mesh_to_dolfinx_model below).
    """
    import gmsh

    nodes = mesh["nodes"]
    elements = mesh["elements"]
    element_type = mesh["element_type"]
    order = int(mesh["element_order"])
    bfaces = mesh["boundary_faces"]
    btags = mesh["boundary_tags"]

    # gmsh element type codes
    # volume
    gmsh_vol_type = {
        "tet4": 4,    # 4-node tetra
        "tet10": 11,  # 10-node tetra
        "hex8": 5,    # 8-node hex
        "hex20": 17,  # 20-node hex
    }[element_type]
    # facets
    cell_kind = "tet" if element_type.startswith("tet") else "hex"
    gmsh_facet_type = (2 if order == 1 else 9) if cell_kind == "tet" else (3 if order == 1 else 16)

    # Split boundary faces by tag
    faces_by_tag: Dict[int, np.ndarray] = {}
    for tag in (1, 2, 3, 4):
        mask = (btags == tag)
        faces_by_tag[tag] = bfaces[mask]

    # Gmsh is not MPI-parallel; do this in serial
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("walled_tube")

    # Create discrete entities: 4 surfaces + 1 volume
    for s in (1, 2, 3, 4):
        gmsh.model.addDiscreteEntity(2, s)
    gmsh.model.addDiscreteEntity(3, 1, [1, 2, 3, 4])

    # Add physical groups (facet tags match your boundary tag numbers)
    gmsh.model.addPhysicalGroup(3, [1], 1)
    gmsh.model.setPhysicalName(3, 1, "tube_volume")
    for s in (1, 2, 3, 4):
        gmsh.model.addPhysicalGroup(2, [s], s)
        name = {1: "inner", 2: "outer", 3: "endcap_0", 4: "endcap_L"}[s]
        gmsh.model.setPhysicalName(2, s, name)

    # Nodes: gmsh uses 1-based tags
    N = nodes.shape[0]
    node_tags = np.arange(1, N + 1, dtype=np.int64)
    coords_flat = nodes.reshape(-1)

    gmsh.model.mesh.addNodes(3, 1, node_tags.tolist(), coords_flat.tolist())

    # Volume elements
    M = elements.shape[0]
    elem_tags = np.arange(1, M + 1, dtype=np.int64)
    conn = (elements + 1).astype(np.int64).reshape(-1)

    gmsh.model.mesh.addElements(
        3, 1,
        [gmsh_vol_type],
        [elem_tags.tolist()],
        [conn.tolist()]
    )

    # Facet elements: ensure globally unique element tags across surfaces
    next_tag = int(elem_tags[-1]) + 1
    for s in (1, 2, 3, 4):
        F = faces_by_tag[s]
        if F.size == 0:
            continue
        K = F.shape[0]
        face_tags = np.arange(next_tag, next_tag + K, dtype=np.int64)
        next_tag += K
        fconn = (F + 1).astype(np.int64).reshape(-1)

        gmsh.model.mesh.addElements(
            2, s,
            [gmsh_facet_type],
            [face_tags.tolist()],
            [fconn.tolist()]
        )

    gmsh.write(msh_path)
    gmsh.finalize()

    if fibers_npz_path is not None:
        payload = {}
        payload["element_type"] = np.array([element_type])
        payload["element_order"] = np.array([order], dtype=np.int64)
        for name, arr in mesh["fiber_families"].items():
            payload[f"fiber__{name}"] = np.asarray(arr, dtype=float)
        np.savez_compressed(fibers_npz_path, **payload)


# -----------------------------
# 3) Build a dolfinx model from the dict
# -----------------------------

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


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    tube = generate_walled_tube_mesh_with_fibers(
        axial_length=10.0,
        lumen_diameter=4.0,
        wall_thickness=1.0,
        n_axial=20,
        n_circ=32,
        n_radial=3,
        axis="z",
        mesh_type="hex20",
        fiber_angles={"axial": 0.0, "helical_pos": np.pi/6, "helical_neg": -np.pi/6},
    )

    # Write for dolfinx
    write_tube_mesh_and_fibers_for_dolfinx(
        tube,
        msh_path="tube.msh",
        fibers_npz_path="tube_fibers.npz",
    )

    # Build an in-memory dolfinx model
    model = tube_mesh_to_dolfinx_model(tube)
    print(model.mesh, model.facet_tags, list(model.fibers.keys()))
