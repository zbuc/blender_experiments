"""
Generate turntable image sequence for multi-view testing.

Creates 12 lateral views (30° spacing) + 1 top view of a test object.
Renders as silhouettes for Visual Hull reconstruction testing.

Usage:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python generate_turntable_sequence.py -- --object vase --output test_images/turntable_vase/
"""

from __future__ import annotations

import sys
import bpy
import math
from pathlib import Path
from mathutils import Vector
from typing import Tuple

# Add to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from integration.blender_ops.camera_framing import compute_bounds_world
from integration.blender_ops.silhouette_render import (
    render_silhouette_frame,
    set_camera_orbit,
    set_camera_top,
    silhouette_session,
)


def clear_scene() -> None:
    """Remove all objects from scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def create_vase() -> bpy.types.Object:
    """Create a simple vase shape for testing."""
    # Create cylinder
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=2.0, location=(0, 0, 0))
    vase = bpy.context.active_object

    # Enter edit mode and taper
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")

    # Simple scaling to create vase-like shape
    bpy.ops.transform.resize(value=(1.2, 1.2, 1.0), constraint_axis=(True, True, False))

    bpy.ops.object.mode_set(mode="OBJECT")

    return vase


def create_bottle() -> bpy.types.Object:
    """Create a simple bottle shape for testing."""
    # Create main body
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1.5, location=(0, 0, 0.75))
    body = bpy.context.active_object

    # Create neck
    bpy.ops.mesh.primitive_cylinder_add(radius=0.2, depth=0.5, location=(0, 0, 1.75))
    neck = bpy.context.active_object

    # Union them
    modifier = body.modifiers.new(name="Union", type="BOOLEAN")
    modifier.operation = "UNION"
    modifier.object = neck

    bpy.context.view_layer.objects.active = body
    bpy.ops.object.modifier_apply(modifier="Union")
    bpy.data.objects.remove(neck)

    return body


def create_cube() -> bpy.types.Object:
    """Create a cube for testing."""
    bpy.ops.mesh.primitive_cube_add(size=1.5, location=(0, 0, 0))
    return bpy.context.active_object


def setup_camera(
    obj: bpy.types.Object, margin_frac: float = 0.08
) -> Tuple[bpy.types.Object, Vector, float, float, float]:
    """Setup orthographic camera using object bounds."""
    bounds_min, bounds_max = compute_bounds_world([obj])
    center = (bounds_min + bounds_max) / 2.0
    width = bounds_max.x - bounds_min.x
    depth = bounds_max.y - bounds_min.y
    height = bounds_max.z - bounds_min.z
    max_dim = max(width, depth, height, 1e-3)
    distance = max_dim * 2.0
    ortho_scale = max_dim * (1.0 + 2.0 * margin_frac)
    top_ortho_scale = max(width, depth, 1e-3) * (1.0 + 2.0 * margin_frac)

    bpy.ops.object.camera_add(location=(center.x + distance, center.y, center.z))
    camera = bpy.context.active_object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = ortho_scale

    # Point at object center
    direction = center - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    bpy.context.scene.camera = camera
    return camera, center, distance, ortho_scale, top_ortho_scale


def generate_turntable_sequence(
    object_name: str = "vase",
    output_dir: str = "test_images/turntable/",
    num_views: int = 12,
) -> None:
    """
    Generate turntable sequence.

    Args:
        object_name: 'vase', 'bottle', or 'cube'
        output_dir: Output directory path
        num_views: Number of lateral views (evenly spaced around 360°)
    """
    if num_views <= 0:
        raise ValueError("num_views must be >= 1")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"GENERATING TURNTABLE SEQUENCE: {object_name}")
    print(f"{'='*70}")
    print(f"Output directory: {output_path}")
    print(f"Lateral views: {num_views} (every {360/num_views}°)")
    print(f"Total views: {num_views + 1} (lateral + top)")

    # Clear scene
    clear_scene()

    # Create object
    print(f"\nCreating {object_name}...")
    if object_name == "vase":
        obj = create_vase()
    elif object_name == "bottle":
        obj = create_bottle()
    elif object_name == "cube":
        obj = create_cube()
    else:
        raise ValueError(f"Unknown object: {object_name}")

    # Setup camera
    camera, center, distance, ortho_scale, top_ortho_scale = setup_camera(obj)

    # Render lateral views (turntable)
    print(f"\nRendering {num_views} lateral views...")
    angle_step = 360.0 / num_views

    with silhouette_session(
        target_objects=[obj],
        camera=camera,
        resolution=(512, 512),
        color_mode="BW",
        transparent_bg=False,
        engine="BLENDER_EEVEE",
        background_color=(1.0, 1.0, 1.0, 1.0),
        silhouette_color=(0.0, 0.0, 0.0, 1.0),
    ) as session:
        for i in range(num_views):
            angle = i * angle_step
            angle_rad = math.radians(angle)
            set_camera_orbit(session.camera, center, distance, angle_rad, ortho_scale)

            filename = f"view_{int(angle):03d}.png"
            render_silhouette_frame(session, output_path / filename)

        # Render top view
        print(f"\nRendering top view...")
        set_camera_top(session.camera, center, distance, top_ortho_scale)
        render_silhouette_frame(session, output_path / "top.png")

    print(f"\n{'='*70}")
    print(f"COMPLETE: {num_views + 1} views rendered")
    print(f"{'='*70}")
    print(f"\nOutput files:")
    for f in sorted(output_path.glob("*.png")):
        print(f"  {f.name}")


def main() -> None:
    """Parse arguments and generate sequence."""
    # Parse command line arguments
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    # Defaults
    object_name = "vase"
    output_dir = "test_images/turntable_vase/"
    num_views = 12

    # Parse simple arguments
    i = 0
    while i < len(argv):
        if argv[i] == "--object" and i + 1 < len(argv):
            object_name = argv[i + 1]
            i += 2
        elif argv[i] == "--output" and i + 1 < len(argv):
            output_dir = argv[i + 1]
            i += 2
        elif argv[i] == "--num-views" and i + 1 < len(argv):
            num_views = int(argv[i + 1])
            i += 2
        else:
            i += 1

    # Generate sequence
    generate_turntable_sequence(object_name, output_dir, num_views)


if __name__ == "__main__":
    main()
