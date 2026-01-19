"""Mesh generation utilities using Blender Python API."""

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
    name: str = "GeneratedMesh"
) -> Optional[object]:
    """
    Create a Blender mesh from contour data.

    Args:
        contours: List of contour arrays
        name: Name for the mesh object

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

        vertices = []
        for point in main_contour:
            # Normalize coordinates to Blender scale
            x = (point[0][0] - 256) / 256.0
            y = (point[0][1] - 256) / 256.0
            z = 0.0
            vert = bm.verts.new((x, y, z))
            vertices.append(vert)

        # Create face if we have enough vertices
        if len(vertices) >= 3:
            bm.faces.new(vertices)

    # Update mesh
    bm.to_mesh(mesh)
    bm.free()

    return obj


def extrude_profile(
    obj: object,
    extrude_distance: float = 1.0
) -> object:
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
    bpy.ops.object.mode_set(mode='EDIT')

    # Select all
    bpy.ops.mesh.select_all(action='SELECT')

    # Extrude
    bpy.ops.mesh.extrude_region_move(
        TRANSFORM_OT_translate={"value": (0, 0, extrude_distance)}
    )

    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    return obj
