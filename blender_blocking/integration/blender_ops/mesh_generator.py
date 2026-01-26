"""Mesh generation utilities using Blender Python API."""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional

try:
    import bpy
    import bmesh

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False


def create_mesh_from_contours(
    contours: List[np.ndarray],
    name: str = "GeneratedMesh",
    source_size: Optional[Tuple[int, int]] = None,
    canonical_size: Optional[int] = None,
    normalize_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[object]:
    """
    Create a Blender mesh from contour data.

    Args:
        contours: List of contour arrays
        name: Name for the mesh object
        source_size: Optional (width, height) of the source image
        canonical_size: Optional square canvas size for normalization

    Returns:
        Blender mesh object or None if Blender not available
    """
    if not BLENDER_AVAILABLE:
        print("Warning: Blender API not available")
        return None

    # Create a new mesh and object
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)

    # Link to scene
    bpy.context.collection.objects.link(obj)

    # Create bmesh
    bm = bmesh.new()

    # Add vertices from largest contour
    if contours:
        main_contour = max(contours, key=lambda c: len(c))
        points = [(float(p[0][0]), float(p[0][1])) for p in main_contour]

        # Drop consecutive near-duplicates
        deduped = []
        eps = 1e-6
        for x, y in points:
            if not deduped:
                deduped.append((x, y))
                continue
            last_x, last_y = deduped[-1]
            if abs(x - last_x) > eps or abs(y - last_y) > eps:
                deduped.append((x, y))

        # Drop near-duplicate points globally
        unique = []
        seen = set()
        for x, y in deduped:
            key = (round(x, 6), round(y, 6))
            if key in seen:
                continue
            seen.add(key)
            unique.append((x, y))

        if len(unique) < 3:
            raise ValueError("Not enough unique contour points to build a face")

        if normalize_bounds is not None:
            min_x, max_x, min_y, max_y = normalize_bounds
            width = max(max_x - min_x, 1.0)
            height = max(max_y - min_y, 1.0)
            center_x = min_x + width / 2.0
            center_y = min_y + height / 2.0
            scale_x = width / 2.0
            scale_y = height / 2.0
        elif source_size:
            width, height = source_size
            center_x = width / 2.0
            center_y = height / 2.0
            scale_x = max(width / 2.0, 1.0)
            scale_y = max(height / 2.0, 1.0)
        elif canonical_size:
            center_x = canonical_size / 2.0
            center_y = canonical_size / 2.0
            scale_x = max(canonical_size / 2.0, 1.0)
            scale_y = max(canonical_size / 2.0, 1.0)
        else:
            xs = [pt[0] for pt in unique]
            ys = [pt[1] for pt in unique]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            width = max(max_x - min_x, 1.0)
            height = max(max_y - min_y, 1.0)
            center_x = min_x + width / 2.0
            center_y = min_y + height / 2.0
            scale_x = width / 2.0
            scale_y = height / 2.0

        vertices = []
        for x, y in unique:
            x_norm = (x - center_x) / scale_x
            y_norm = (y - center_y) / scale_y
            vert = bm.verts.new((x_norm, y_norm, 0.0))
            vertices.append(vert)

        # Create face if we have enough vertices
        if len(vertices) >= 3:
            try:
                bm.faces.new(vertices)
            except ValueError:
                edges = []
                for i in range(len(vertices)):
                    v1 = vertices[i]
                    v2 = vertices[(i + 1) % len(vertices)]
                    try:
                        edge = bm.edges.new((v1, v2))
                        edges.append(edge)
                    except ValueError:
                        pass
                if edges:
                    bmesh.ops.triangle_fill(bm, edges=edges, use_beauty=True)

    # Update mesh
    bm.to_mesh(mesh)
    bm.free()

    return obj


def extrude_profile(obj: object, extrude_distance: float = 1.0) -> object:
    """
    Extrude a 2D profile to create a 3D mesh.

    Args:
        obj: Blender object to extrude
        extrude_distance: Distance to extrude

    Returns:
        Modified Blender object
    """
    if not BLENDER_AVAILABLE:
        print("Warning: Blender API not available")
        return obj

    # Set object as active
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Enter edit mode
    bpy.ops.object.mode_set(mode="EDIT")

    # Select all
    bpy.ops.mesh.select_all(action="SELECT")

    # Extrude
    bpy.ops.mesh.extrude_region_move(
        TRANSFORM_OT_translate={"value": (0, 0, extrude_distance)}
    )

    # Return to object mode
    bpy.ops.object.mode_set(mode="OBJECT")

    return obj


def center_extrusion(obj: object, extrude_distance: float = 1.0) -> None:
    """Center an extruded mesh along its local Z axis."""
    if not BLENDER_AVAILABLE or obj is None:
        return
    if getattr(obj, "type", None) != "MESH":
        return
    if extrude_distance == 0:
        return
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    offset = extrude_distance / 2.0
    if bm.verts:
        for vert in bm.verts:
            vert.co.z -= offset
    bm.to_mesh(obj.data)
    bm.free()


def clean_mesh_for_boolean(
    obj: object,
    *,
    merge_dist: float = 1e-6,
    dissolve_dist: float = 1e-6,
) -> None:
    """Cleanup mesh topology for boolean operations."""
    if not BLENDER_AVAILABLE or obj is None:
        return
    if getattr(obj, "type", None) != "MESH":
        return
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    if bm.verts:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_dist)
    if bm.edges:
        bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=dissolve_dist)
    if bm.faces:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()


def triangulate_object(obj: object) -> None:
    """Triangulate all faces on a mesh object for boolean robustness."""
    if not BLENDER_AVAILABLE or obj is None:
        return
    if getattr(obj, "type", None) != "MESH":
        return
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    if bm.faces:
        bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
