"""
Demo script for Slice-based Shape Matching Algorithm

This script demonstrates how to use the slice-based shape matcher
to compare different 3D shapes in Blender.
"""

import bpy
import mathutils
from mathutils import Vector
from slice_shape_matcher import SliceBasedShapeMatcher


def create_test_shapes():
    """Create a set of test shapes for demonstration."""
    # Clear existing mesh objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)

    shapes = []

    # Create a cylinder
    bpy.ops.mesh.primitive_cylinder_add(radius=1, depth=2, location=(0, 0, 0))
    cylinder = bpy.context.active_object
    cylinder.name = "Cylinder_1"
    shapes.append(cylinder)

    # Create another cylinder (should be very similar)
    bpy.ops.mesh.primitive_cylinder_add(radius=1.1, depth=2.1, location=(3, 0, 0))
    cylinder2 = bpy.context.active_object
    cylinder2.name = "Cylinder_2"
    shapes.append(cylinder2)

    # Create a cone (somewhat similar to cylinder)
    bpy.ops.mesh.primitive_cone_add(radius1=1, depth=2, location=(6, 0, 0))
    cone = bpy.context.active_object
    cone.name = "Cone"
    shapes.append(cone)

    # Create a cube (very different from cylinder)
    bpy.ops.mesh.primitive_cube_add(size=2, location=(9, 0, 0))
    cube = bpy.context.active_object
    cube.name = "Cube"
    shapes.append(cube)

    # Create a UV sphere
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(12, 0, 0))
    sphere = bpy.context.active_object
    sphere.name = "Sphere"
    shapes.append(sphere)

    return shapes


def compare_all_shapes(shapes, num_slices=25):
    """Compare all pairs of shapes and display results."""
    matcher = SliceBasedShapeMatcher(num_slices=num_slices)

    print("\n" + "="*70)
    print(" " * 20 + "SHAPE MATCHING RESULTS")
    print("="*70)

    results = []

    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            obj1, obj2 = shapes[i], shapes[j]
            result = matcher.match_shapes(obj1, obj2)

            results.append({
                'obj1': obj1.name,
                'obj2': obj2.name,
                'similarity': result['similarity'],
                'quality': result['match_quality']
            })

            print(f"\n{obj1.name:15} vs {obj2.name:15}")
            print(f"  Similarity: {result['similarity']:.4f} ({result['match_quality']})")
            print(f"  Slices: {result['num_slices_obj1']} vs {result['num_slices_obj2']}")

    print("\n" + "="*70)
    print("RANKING (Most similar pairs first):")
    print("="*70)

    results.sort(key=lambda x: x['similarity'], reverse=True)

    for idx, result in enumerate(results, 1):
        print(f"{idx}. {result['obj1']:15} <-> {result['obj2']:15}  "
              f"Score: {result['similarity']:.4f} ({result['quality']})")

    print("="*70 + "\n")

    return results


def analyze_single_pair(obj1_name, obj2_name, num_slices=30, axis='Z'):
    """Analyze a specific pair of objects in detail."""
    try:
        obj1 = bpy.data.objects[obj1_name]
        obj2 = bpy.data.objects[obj2_name]
    except KeyError as e:
        print(f"Error: Object not found - {e}")
        return None

    if obj1.type != 'MESH' or obj2.type != 'MESH':
        print("Error: Both objects must be mesh objects")
        return None

    matcher = SliceBasedShapeMatcher(num_slices=num_slices)

    print(f"\nDetailed Analysis: {obj1_name} vs {obj2_name}")
    print("="*50)

    # Get profiles
    profiles1 = matcher.slice_mesh(obj1, axis)
    profiles2 = matcher.slice_mesh(obj2, axis)

    print(f"Slicing along {axis}-axis with {num_slices} slices")
    print(f"\nObject 1 ({obj1_name}):")
    print(f"  Total slices with geometry: {len(profiles1)}")

    if profiles1:
        areas1 = [p.area for p in profiles1]
        print(f"  Area range: {min(areas1):.3f} - {max(areas1):.3f}")
        print(f"  Average area: {sum(areas1)/len(areas1):.3f}")

    print(f"\nObject 2 ({obj2_name}):")
    print(f"  Total slices with geometry: {len(profiles2)}")

    if profiles2:
        areas2 = [p.area for p in profiles2]
        print(f"  Area range: {min(areas2):.3f} - {max(areas2):.3f}")
        print(f"  Average area: {sum(areas2)/len(areas2):.3f}")

    # Compare
    result = matcher.match_shapes(obj1, obj2, axis)

    print(f"\nMatch Result:")
    print(f"  Similarity Score: {result['similarity']:.4f}")
    print(f"  Match Quality: {result['match_quality']}")
    print("="*50 + "\n")

    return result


def main():
    """Main demo function."""
    print("\n" + "="*70)
    print(" " * 15 + "SLICE-BASED SHAPE MATCHER DEMO")
    print("="*70)

    # Create test shapes
    print("\nCreating test shapes...")
    shapes = create_test_shapes()
    print(f"Created {len(shapes)} test shapes: {', '.join(s.name for s in shapes)}")

    # Compare all shapes
    print("\nComparing all shape pairs...")
    results = compare_all_shapes(shapes, num_slices=25)

    # Detailed analysis of most similar pair
    if results:
        most_similar = results[0]
        print("\nDetailed analysis of most similar pair:")
        analyze_single_pair(most_similar['obj1'], most_similar['obj2'], num_slices=40)

    print("\nDemo complete!")
    print("\nTo compare your own objects:")
    print("  1. Select two mesh objects in Blender")
    print("  2. Run: from slice_shape_matcher import test_slice_matcher")
    print("  3. Run: test_slice_matcher()")


if __name__ == "__main__":
    main()
