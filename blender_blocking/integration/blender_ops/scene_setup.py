"""Blender scene setup utilities."""

import math

try:
    import bpy
    import mathutils
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False


def setup_scene(clear_existing: bool = True) -> None:
    """
    Setup a clean Blender scene.

    Args:
        clear_existing: Whether to clear existing objects
    """
    if not BLENDER_AVAILABLE:
        print("Warning: Blender API not available")
        return

    if clear_existing:
        # Delete all existing mesh objects
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

    # Set render settings
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512


def add_camera(
    location: tuple = (7.5, -7.5, 5.5),
    rotation: tuple = (63.0, 0.0, 45.0)
) -> object:
    """
    Add a camera to the scene.

    Args:
        location: Camera location (x, y, z)
        rotation: Camera rotation in degrees (x, y, z)

    Returns:
        Camera object
    """
    if not BLENDER_AVAILABLE:
        print("Warning: Blender API not available")
        return None

    # Create camera
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object

    # Set rotation (convert to radians)
    camera.rotation_euler = [math.radians(r) for r in rotation]

    # Set as active camera
    bpy.context.scene.camera = camera

    return camera


def add_lighting(
    light_type: str = 'SUN',
    location: tuple = (0, 0, 10),
    energy: float = 1.0
) -> object:
    """
    Add lighting to the scene.

    Args:
        light_type: Type of light ('SUN', 'POINT', 'SPOT', 'AREA')
        location: Light location (x, y, z)
        energy: Light intensity

    Returns:
        Light object
    """
    if not BLENDER_AVAILABLE:
        print("Warning: Blender API not available")
        return None

    # Create light
    bpy.ops.object.light_add(type=light_type, location=location)
    light = bpy.context.object

    # Set energy
    light.data.energy = energy

    return light
