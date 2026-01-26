"""Loft mesh generation from elliptical profile slices using bmesh."""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

from geometry.profile_models import EllipticalSlice

try:
    import bpy
    import bmesh

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False


def _ring_vertices(
    bm: "bmesh.types.BMesh",
    slice_data: EllipticalSlice,
    radial_segments: int,
    min_radius_u: float,
    weld_degenerate_rings: bool,
) -> Tuple[List["bmesh.types.BMVert"], bool]:
    rx = float(slice_data.rx)
    ry = float(slice_data.ry)
    center_x = float(slice_data.cx) if slice_data.cx is not None else 0.0
    center_y = float(slice_data.cy) if slice_data.cy is not None else 0.0

    if weld_degenerate_rings and (rx <= min_radius_u or ry <= min_radius_u):
        vert = bm.verts.new((center_x, center_y, slice_data.z))
        return [vert], True

    verts = []
    for i in range(radial_segments):
        theta = (2.0 * math.pi * i) / radial_segments
        x = center_x + rx * math.cos(theta)
        y = center_y + ry * math.sin(theta)
        verts.append(bm.verts.new((x, y, slice_data.z)))

    return verts, False


def _bridge_rings(
    bm: "bmesh.types.BMesh",
    ring_a: List["bmesh.types.BMVert"],
    ring_b: List["bmesh.types.BMVert"],
) -> None:
    if len(ring_a) == 1 and len(ring_b) == 1:
        return

    if len(ring_a) == 1:
        center = ring_a[0]
        for i in range(len(ring_b)):
            v1 = ring_b[i]
            v2 = ring_b[(i + 1) % len(ring_b)]
            bm.faces.new((center, v1, v2))
        return

    if len(ring_b) == 1:
        center = ring_b[0]
        for i in range(len(ring_a)):
            v1 = ring_a[i]
            v2 = ring_a[(i + 1) % len(ring_a)]
            bm.faces.new((v1, v2, center))
        return

    count = min(len(ring_a), len(ring_b))
    for i in range(count):
        v1 = ring_a[i]
        v2 = ring_a[(i + 1) % count]
        v3 = ring_b[(i + 1) % count]
        v4 = ring_b[i]
        try:
            bm.faces.new((v1, v2, v3, v4))
        except ValueError:
            continue


def _cap_ring(
    bm: "bmesh.types.BMesh",
    ring: List["bmesh.types.BMVert"],
    cap_mode: str,
) -> None:
    if len(ring) < 3:
        return

    if cap_mode == "fan":
        center_coords = [sum(v.co[i] for v in ring) / len(ring) for i in range(3)]
        center = bm.verts.new(center_coords)
        for i in range(len(ring)):
            v1 = ring[i]
            v2 = ring[(i + 1) % len(ring)]
            try:
                bm.faces.new((center, v1, v2))
            except ValueError:
                continue
        return

    if cap_mode == "ngon":
        try:
            bm.faces.new(ring)
        except ValueError:
            edges = []
            for i in range(len(ring)):
                v1 = ring[i]
                v2 = ring[(i + 1) % len(ring)]
                try:
                    edges.append(bm.edges.new((v1, v2)))
                except ValueError:
                    continue
            if edges:
                bmesh.ops.triangle_fill(bm, edges=edges, use_beauty=True)
        return

    if cap_mode != "none":
        raise ValueError(f"Unknown cap_mode: {cap_mode}")


def create_loft_mesh_from_slices(
    slices: Sequence[EllipticalSlice],
    *,
    name: str = "LoftMesh",
    radial_segments: int = 24,
    cap_mode: str = "fan",
    min_radius_u: float = 0.0,
    merge_threshold_u: float = 0.0,
    recalc_normals: bool = True,
    shade_smooth: bool = True,
    weld_degenerate_rings: bool = True,
) -> Optional[object]:
    """Create a Blender mesh object lofted from elliptical slices."""
    if not BLENDER_AVAILABLE:
        print("Warning: Blender API not available")
        return None

    if radial_segments < 3:
        raise ValueError("radial_segments must be >= 3")

    if not slices:
        raise ValueError("slices must not be empty")

    bm = bmesh.new()

    rings: List[List[bmesh.types.BMVert]] = []
    for slice_data in slices:
        ring, _ = _ring_vertices(
            bm,
            slice_data,
            radial_segments,
            min_radius_u,
            weld_degenerate_rings,
        )
        rings.append(ring)

    for ring_a, ring_b in zip(rings[:-1], rings[1:]):
        _bridge_rings(bm, ring_a, ring_b)

    if cap_mode != "none":
        _cap_ring(bm, rings[0], cap_mode)
        _cap_ring(bm, rings[-1], cap_mode)

    if merge_threshold_u > 0:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_threshold_u)

    if recalc_normals:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    if shade_smooth:
        for polygon in obj.data.polygons:
            polygon.use_smooth = True

    return obj
