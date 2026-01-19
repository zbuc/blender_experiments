"""
Simple example of primitive placement and mesh joining in Blender.
Run this script in Blender's Text Editor or via command line:
  blender --python example_simple.py
"""

import bpy
from mathutils import Vector
from primitive_placement import SliceAnalyzer, PrimitivePlacer, MeshJoiner, clear_scene


def create_simple_form():
    """Create a simple organic form using primitives."""

    # Clear the scene
    clear_scene()

    # Define a vertical volume
    bounds_min = Vector((-1.5, -1.5, 0))
    bounds_max = Vector((1.5, 1.5, 4))

    # Analyze with fewer slices for simpler result
    analyzer = SliceAnalyzer(bounds_min, bounds_max, num_slices=8)
    slice_data = analyzer.get_all_slice_data()

    # Place cylinder primitives
    placer = PrimitivePlacer()
    primitives = placer.place_primitives_from_slices(slice_data, primitive_type='CYLINDER')

    # Join using boolean union
    joiner = MeshJoiner()
    final_mesh = joiner.join_with_boolean_union(primitives, target_name="Simple_Form")

    # Position camera to view the result
    if 'Camera' in bpy.data.objects:
        camera = bpy.data.objects['Camera']
        camera.location = (7, -7, 4)
        camera.rotation_euler = (1.1, 0, 0.785)

    print(f"Created {final_mesh.name} with {len(final_mesh.data.vertices)} vertices")
    return final_mesh


if __name__ == "__main__":
    create_simple_form()
