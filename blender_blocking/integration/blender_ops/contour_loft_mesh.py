"""Loft mesh generation using top-view contour templates."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from geometry.profile_models import EllipticalSlice
from geometry.contour_models import ContourTemplate, ContourSlice
from geometry.contour_utils import scale_contour_2d

try:
    import bpy
    import bmesh
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False


def _ring_vertices_from_contour(
    bm: "bmesh.types.BMesh",
    slice_data: ContourSlice,
    template: ContourTemplate,
    min_radius_u: float,
    weld_degenerate_rings: bool,
    unit_scale: float = 0.01,
) -> Tuple[List["bmesh.types.BMVert"], bool]:
    """
    Generate ring vertices using contour template scaled by profile factors.

    Args:
        bm: BMesh instance
        slice_data: ContourSlice with z position and scale factors
        template: Normalized contour template (includes original_bbox)
        min_radius_u: Minimum radius for welding
        weld_degenerate_rings: Whether to collapse degenerate rings to points
        unit_scale: Pixel to world unit conversion

    Returns:
        (vertices, is_degenerate)
    """
    # Check for degenerate slice
    if weld_degenerate_rings and (
        slice_data.scale_x <= min_radius_u or slice_data.scale_y <= min_radius_u
    ):
        vert = bm.verts.new((slice_data.cx, slice_data.cy, slice_data.z))
        return [vert], True

    # Compute actual scale factors based on scaling mode
    actual_scale_x = slice_data.scale_x
    actual_scale_y = slice_data.scale_y

    if slice_data.scale_mode == "contour_native" and template.original_bbox is not None:
        # For flat objects, use contour's native dimensions
        _, _, w_px, h_px = template.original_bbox
        from geometry.contour_utils import compute_contour_native_scale
        actual_scale_x, actual_scale_y = compute_contour_native_scale(
            w_px, h_px, unit_scale, slice_data.z_scale_factor
        )

    # Scale template contour by computed factors
    scaled_2d = scale_contour_2d(
        template.points,
        scale_x=actual_scale_x,
        scale_y=actual_scale_y,
        center_x=slice_data.cx,
        center_y=slice_data.cy,
    )

    # Create 3D vertices at height z
    verts = []
    for x, y in scaled_2d:
        verts.append(bm.verts.new((x, y, slice_data.z)))

    return verts, False


def _bridge_rings(
    bm: "bmesh.types.BMesh",
    ring_a: List["bmesh.types.BMVert"],
    ring_b: List["bmesh.types.BMVert"],
) -> None:
    """Bridge two vertex rings (identical to elliptical version)."""
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
    """Cap a ring (identical to elliptical version)."""
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


def create_contour_loft_mesh(
    slices: Sequence[ContourSlice],
    template: ContourTemplate,
    *,
    name: str = "ContourLoftMesh",
    cap_mode: str = "fan",
    min_radius_u: float = 0.0,
    merge_threshold_u: float = 0.0,
    recalc_normals: bool = True,
    shade_smooth: bool = True,
    weld_degenerate_rings: bool = True,
    unit_scale: float = 0.01,
) -> Optional[object]:
    """
    Create Blender mesh object lofted from contour template.

    Args:
        slices: Sequence of ContourSlice with z positions and scale factors
        template: Normalized contour template for cross-section shape
        name: Mesh object name
        cap_mode: Capping strategy ("fan", "ngon", "none")
        min_radius_u: Minimum radius for welding degenerate rings
        merge_threshold_u: Distance threshold for vertex merging
        recalc_normals: Recalculate face normals
        shade_smooth: Enable smooth shading
        weld_degenerate_rings: Collapse degenerate rings to points
        unit_scale: Pixel to world unit conversion

    Returns:
        Blender mesh object or None on error
    """
    if not BLENDER_AVAILABLE:
        print("Warning: Blender API not available")
        return None

    if not slices:
        raise ValueError("slices must not be empty")

    if template.num_vertices < 3:
        raise ValueError("template must have at least 3 vertices")

    bm = bmesh.new()

    # Generate rings using contour template
    rings: List[List[bmesh.types.BMVert]] = []
    for slice_data in slices:
        ring, _ = _ring_vertices_from_contour(
            bm,
            slice_data,
            template,
            min_radius_u,
            weld_degenerate_rings,
            unit_scale,
        )
        rings.append(ring)

    # Bridge consecutive rings
    for ring_a, ring_b in zip(rings[:-1], rings[1:]):
        _bridge_rings(bm, ring_a, ring_b)

    # Cap ends
    if cap_mode != "none":
        _cap_ring(bm, rings[0], cap_mode)
        _cap_ring(bm, rings[-1], cap_mode)

    # Merge close vertices
    if merge_threshold_u > 0:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_threshold_u)

    # Recalculate normals
    if recalc_normals:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    # Create mesh and object
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    # Apply smooth shading
    if shade_smooth:
        for polygon in obj.data.polygons:
            polygon.use_smooth = True

    return obj
