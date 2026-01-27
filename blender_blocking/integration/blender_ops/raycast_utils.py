"""
World-space ray casting helpers for Blender objects.

Blender's Object.ray_cast expects coordinates in the object's local space.
These helpers accept world-space origins/directions and handle transforms.
"""

from __future__ import annotations

from typing import Tuple

import bpy
from mathutils import Vector


def ray_cast_world(
    mesh_obj: bpy.types.Object,
    origin_world: Vector,
    direction_world: Vector,
    distance_world: float,
) -> Tuple[bool, Vector, Vector, int]:
    """
    Cast a ray in world space against a mesh object.

    Returns Blender-style (hit, location, normal, index) with world-space
    location/normal when a hit is found.
    """
    if distance_world <= 0.0:
        return False, Vector(), Vector(), -1

    world_to_local = mesh_obj.matrix_world.inverted()
    origin_local = world_to_local @ origin_world
    direction_local = world_to_local.to_3x3() @ direction_world

    if direction_local.length == 0.0:
        return False, Vector(), Vector(), -1

    direction_local_normalized = direction_local.normalized()
    distance_local = distance_world * direction_local.length

    hit, location, normal, index = mesh_obj.ray_cast(
        origin_local, direction_local_normalized, distance=distance_local
    )

    if not hit:
        return hit, Vector(), Vector(), index

    location_world = mesh_obj.matrix_world @ location
    normal_world = (
        mesh_obj.matrix_world.to_3x3().inverted().transposed() @ normal
    ).normalized()

    return hit, location_world, normal_world, index
