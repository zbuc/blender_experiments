"""
Generate turntable image sequence for multi-view testing.

Creates 12 lateral views (30° spacing) + 1 top view of a test object.
Renders as silhouettes for Visual Hull reconstruction testing.

Usage:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python generate_turntable_sequence.py -- --object vase --output test_images/turntable_vase/
"""

import sys
import bpy
import math
from pathlib import Path
from mathutils import Vector

# Add to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def clear_scene():
    """Remove all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def create_vase():
    """Create a simple vase shape for testing."""
    # Create cylinder
    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.5,
        depth=2.0,
        location=(0, 0, 0)
    )
    vase = bpy.context.active_object

    # Enter edit mode and taper
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    # Simple scaling to create vase-like shape
    bpy.ops.transform.resize(value=(1.2, 1.2, 1.0), constraint_axis=(True, True, False))

    bpy.ops.object.mode_set(mode='OBJECT')

    return vase


def create_bottle():
    """Create a simple bottle shape for testing."""
    # Create main body
    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.5,
        depth=1.5,
        location=(0, 0, 0.75)
    )
    body = bpy.context.active_object

    # Create neck
    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.2,
        depth=0.5,
        location=(0, 0, 1.75)
    )
    neck = bpy.context.active_object

    # Union them
    modifier = body.modifiers.new(name='Union', type='BOOLEAN')
    modifier.operation = 'UNION'
    modifier.object = neck

    bpy.context.view_layer.objects.active = body
    bpy.ops.object.modifier_apply(modifier='Union')
    bpy.data.objects.remove(neck)

    return body


def create_cube():
    """Create a cube for testing."""
    bpy.ops.mesh.primitive_cube_add(size=1.5, location=(0, 0, 0))
    return bpy.context.active_object


def setup_camera(distance=5.0):
    """Setup orthographic camera."""
    bpy.ops.object.camera_add(location=(distance, 0, 0))
    camera = bpy.context.active_object
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 4.0

    # Point at origin
    direction = Vector((0, 0, 0)) - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    bpy.context.scene.camera = camera
    return camera


def setup_lighting():
    """Setup basic lighting."""
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 5))
    light = bpy.context.active_object
    light.data.energy = 1.0
    return light


def render_silhouette(output_path, resolution=512):
    """Render current view as silhouette."""
    scene = bpy.context.scene

    # Setup render settings
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'BW'
    scene.render.filepath = str(output_path)

    # Setup for silhouette rendering
    scene.render.engine = 'BLENDER_EEVEE'
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)  # White background

    # Set object to black
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            if not obj.data.materials:
                mat = bpy.data.materials.new(name="Black")
                mat.use_nodes = True
                mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0, 0, 0, 1)
                obj.data.materials.append(mat)

    # Render
    bpy.ops.render.render(write_still=True)
    print(f"   Rendered: {output_path}")


def generate_turntable_sequence(object_name='vase', output_dir='test_images/turntable/', num_views=12):
    """
    Generate turntable sequence.

    Args:
        object_name: 'vase', 'bottle', or 'cube'
        output_dir: Output directory path
        num_views: Number of lateral views (evenly spaced around 360°)
    """
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
    if object_name == 'vase':
        obj = create_vase()
    elif object_name == 'bottle':
        obj = create_bottle()
    elif object_name == 'cube':
        obj = create_cube()
    else:
        raise ValueError(f"Unknown object: {object_name}")

    # Setup camera and lighting
    camera = setup_camera()
    light = setup_lighting()

    # Render lateral views (turntable)
    print(f"\nRendering {num_views} lateral views...")
    angle_step = 360.0 / num_views

    for i in range(num_views):
        angle = i * angle_step
        angle_rad = math.radians(angle)

        # Position camera
        distance = 5.0
        camera.location.x = distance * math.cos(angle_rad)
        camera.location.y = distance * math.sin(angle_rad)
        camera.location.z = 0.0

        # Point at origin
        direction = Vector((0, 0, 0)) - camera.location
        camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

        # Render
        filename = f"view_{int(angle):03d}.png"
        render_silhouette(output_path / filename)

    # Render top view
    print(f"\nRendering top view...")
    camera.location = (0, 0, 5.0)
    camera.rotation_euler = (0, 0, 0)
    render_silhouette(output_path / "top.png")

    print(f"\n{'='*70}")
    print(f"COMPLETE: {num_views + 1} views rendered")
    print(f"{'='*70}")
    print(f"\nOutput files:")
    for f in sorted(output_path.glob("*.png")):
        print(f"  {f.name}")


def main():
    """Parse arguments and generate sequence."""
    # Parse command line arguments
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    # Defaults
    object_name = 'vase'
    output_dir = 'test_images/turntable_vase/'
    num_views = 12

    # Parse simple arguments
    i = 0
    while i < len(argv):
        if argv[i] == '--object' and i + 1 < len(argv):
            object_name = argv[i + 1]
            i += 2
        elif argv[i] == '--output' and i + 1 < len(argv):
            output_dir = argv[i + 1]
            i += 2
        elif argv[i] == '--num-views' and i + 1 < len(argv):
            num_views = int(argv[i + 1])
            i += 2
        else:
            i += 1

    # Generate sequence
    generate_turntable_sequence(object_name, output_dir, num_views)


if __name__ == "__main__":
    main()
