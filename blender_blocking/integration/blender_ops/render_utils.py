"""Rendering utilities for Blender."""

from pathlib import Path
from typing import Dict, Tuple
import math

try:
    import bpy
    import mathutils
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False


def render_orthogonal_views(
    output_dir: str,
    views: list = ['front', 'side', 'top']
) -> Dict[str, str]:
    """
    Render orthogonal views of the scene.

    Args:
        output_dir: Directory to save renders
        views: List of views to render

    Returns:
        Dictionary mapping view names to output file paths
    """
    if not BLENDER_AVAILABLE:
        print("Warning: Blender API not available")
        return {}

    output_paths = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate scene bounds to center camera
    # Get all mesh objects
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if mesh_objects:
        # Calculate bounding box of all meshes
        min_coords = [float('inf'), float('inf'), float('inf')]
        max_coords = [float('-inf'), float('-inf'), float('-inf')]

        for obj in mesh_objects:
            for vertex in obj.bound_box:
                world_coord = obj.matrix_world @ mathutils.Vector(vertex)
                for i in range(3):
                    min_coords[i] = min(min_coords[i], world_coord[i])
                    max_coords[i] = max(max_coords[i], world_coord[i])

        center_x = (min_coords[0] + max_coords[0]) / 2
        center_y = (min_coords[1] + max_coords[1]) / 2
        center_z = (min_coords[2] + max_coords[2]) / 2
    else:
        # Fallback to origin
        center_x, center_y, center_z = 0, 0, 0

    # Camera settings for orthogonal views, centered on mesh
    view_settings = {
        'front': {
            'location': (center_x, center_y - 10, center_z),
            'rotation': (math.radians(90), 0, 0)
        },
        'side': {
            'location': (center_x + 10, center_y, center_z),
            'rotation': (math.radians(90), 0, math.radians(90))
        },
        'top': {
            'location': (center_x, center_y, center_z + 10),
            'rotation': (0, 0, 0)
        }
    }

    camera = bpy.context.scene.camera
    if not camera:
        # Create camera if none exists
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        bpy.context.scene.camera = camera

    # Set to orthographic
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 5.0

    for view in views:
        if view in view_settings:
            settings = view_settings[view]

            # Set camera position and rotation
            camera.location = settings['location']
            camera.rotation_euler = settings['rotation']

            # Render
            output_file = output_path / f"{view}.png"
            bpy.context.scene.render.filepath = str(output_file)
            bpy.ops.render.render(write_still=True)

            output_paths[view] = str(output_file)

    return output_paths


def save_render(output_path: str) -> None:
    """
    Render and save the current scene.

    Args:
        output_path: Path to save the render
    """
    if not BLENDER_AVAILABLE:
        print("Warning: Blender API not available")
        return

    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
