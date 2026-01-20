"""
Test script for ellipsoid primitive support.

Validates:
1. Ellipsoids can be created with independent X, Y, Z radii
2. Boolean union operations work correctly with ellipsoids
3. Ellipsoids render correctly

Run with: blender --background --python test_ellipsoid.py
"""

import bpy
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import directly from the primitives module
from primitives import primitives
from placement.primitive_placement import PrimitivePlacer, MeshJoiner, clear_scene
from mathutils import Vector

# Get the spawn_ellipsoid function
spawn_ellipsoid = primitives.spawn_ellipsoid


def test_spawn_ellipsoid():
    """Test that ellipsoids can be created with independent X, Y, Z radii."""
    print("\n=== Test 1: Spawn ellipsoid with independent radii ===")

    clear_scene()

    # Helper function for floating point comparison
    def approx_equal(a, b, tolerance=1e-5):
        return abs(a - b) < tolerance

    # Test 1: Tall ellipsoid (bottle-like)
    obj1 = spawn_ellipsoid(
        rx=0.5,
        ry=0.5,
        rz=2.0,
        location=(0, 0, 0),
        name="TallEllipsoid"
    )

    assert obj1 is not None, "Failed to create tall ellipsoid"
    assert obj1.name == "TallEllipsoid", f"Name mismatch: {obj1.name}"
    assert approx_equal(obj1.scale[0], 0.5), f"X scale incorrect: {obj1.scale[0]}"
    assert approx_equal(obj1.scale[1], 0.5), f"Y scale incorrect: {obj1.scale[1]}"
    assert approx_equal(obj1.scale[2], 2.0), f"Z scale incorrect: {obj1.scale[2]}"
    print(f"  ✓ Created tall ellipsoid: rx=0.5, ry=0.5, rz=2.0")

    # Test 2: Wide ellipsoid (disc-like)
    obj2 = spawn_ellipsoid(
        rx=2.0,
        ry=2.0,
        rz=0.5,
        location=(3, 0, 0),
        name="WideEllipsoid"
    )

    assert obj2 is not None, "Failed to create wide ellipsoid"
    assert approx_equal(obj2.scale[0], 2.0), f"X scale incorrect: {obj2.scale[0]}"
    assert approx_equal(obj2.scale[2], 0.5), f"Z scale incorrect: {obj2.scale[2]}"
    print(f"  ✓ Created wide ellipsoid: rx=2.0, ry=2.0, rz=0.5")

    # Test 3: Asymmetric ellipsoid
    obj3 = spawn_ellipsoid(
        rx=1.0,
        ry=1.5,
        rz=0.8,
        location=(-3, 0, 0),
        name="AsymmetricEllipsoid"
    )

    assert obj3 is not None, "Failed to create asymmetric ellipsoid"
    assert approx_equal(obj3.scale[0], 1.0), f"X scale incorrect: {obj3.scale[0]}"
    assert approx_equal(obj3.scale[1], 1.5), f"Y scale incorrect: {obj3.scale[1]}"
    assert approx_equal(obj3.scale[2], 0.8), f"Z scale incorrect: {obj3.scale[2]}"
    print(f"  ✓ Created asymmetric ellipsoid: rx=1.0, ry=1.5, rz=0.8")

    print("✅ Test 1 PASSED: All ellipsoids created successfully\n")
    return [obj1, obj2, obj3]


def test_ellipsoid_in_primitive_placer():
    """Test that PrimitivePlacer supports ELLIPSOID primitive type."""
    print("=== Test 2: PrimitivePlacer with ELLIPSOID ===")

    clear_scene()

    def approx_equal(a, b, tolerance=1e-5):
        return abs(a - b) < tolerance

    placer = PrimitivePlacer()

    # Create ellipsoid using PrimitivePlacer
    obj = placer.create_primitive(
        primitive_type='ELLIPSOID',
        location=(0, 0, 0),
        scale=(1.5, 0.8, 2.0)
    )

    assert obj is not None, "Failed to create ellipsoid via PrimitivePlacer"
    assert approx_equal(obj.scale[0], 1.5), f"X scale incorrect: {obj.scale[0]}"
    assert approx_equal(obj.scale[1], 0.8), f"Y scale incorrect: {obj.scale[1]}"
    assert approx_equal(obj.scale[2], 2.0), f"Z scale incorrect: {obj.scale[2]}"
    print(f"  ✓ PrimitivePlacer created ellipsoid with scale (1.5, 0.8, 2.0)")

    print("✅ Test 2 PASSED: PrimitivePlacer supports ELLIPSOID\n")
    return obj


def test_boolean_union_with_ellipsoids():
    """Test that boolean union operations work correctly with ellipsoids."""
    print("=== Test 3: Boolean union with ellipsoids ===")

    clear_scene()

    # Create multiple ellipsoids that overlap
    ellipsoids = []

    # Bottom ellipsoid
    obj1 = spawn_ellipsoid(
        rx=1.0,
        ry=1.0,
        rz=1.5,
        location=(0, 0, 0.75),
        name="Ellipsoid_Bottom"
    )
    ellipsoids.append(obj1)

    # Middle ellipsoid (slightly smaller)
    obj2 = spawn_ellipsoid(
        rx=0.8,
        ry=0.8,
        rz=1.2,
        location=(0, 0, 2.0),
        name="Ellipsoid_Middle"
    )
    ellipsoids.append(obj2)

    # Top ellipsoid (smaller)
    obj3 = spawn_ellipsoid(
        rx=0.6,
        ry=0.6,
        rz=1.0,
        location=(0, 0, 3.2),
        name="Ellipsoid_Top"
    )
    ellipsoids.append(obj3)

    print(f"  ✓ Created {len(ellipsoids)} overlapping ellipsoids")

    # Join with boolean union
    joiner = MeshJoiner()
    try:
        result = joiner.join_with_boolean_union(ellipsoids, target_name="Ellipsoid_Union")
        assert result is not None, "Boolean union returned None"
        assert result.name == "Ellipsoid_Union", f"Result name incorrect: {result.name}"
        print(f"  ✓ Boolean union succeeded: created '{result.name}'")

        # Verify the result is a valid mesh
        assert result.type == 'MESH', f"Result is not a mesh: {result.type}"
        assert len(result.data.vertices) > 0, "Result mesh has no vertices"
        print(f"  ✓ Result is valid mesh with {len(result.data.vertices)} vertices")

        print("✅ Test 3 PASSED: Boolean union works with ellipsoids\n")
        return result

    except Exception as e:
        print(f"  ✗ Boolean union failed: {e}")
        raise


def test_ellipsoid_stacked_creation():
    """Test creating a vase-like shape from stacked ellipsoids."""
    print("=== Test 4: Vase-like shape from stacked ellipsoids ===")

    clear_scene()

    placer = PrimitivePlacer()

    # Define a vertical profile for a vase shape
    # Bottom (wide) -> Middle (narrow) -> Top (medium)
    profile_data = [
        {'z': 0.5, 'rx': 1.2, 'ry': 1.2, 'rz': 0.8},   # Wide base
        {'z': 1.5, 'rx': 0.9, 'ry': 0.9, 'rz': 0.8},   # Tapering
        {'z': 2.5, 'rx': 0.6, 'ry': 0.6, 'rz': 0.8},   # Narrow neck
        {'z': 3.5, 'rx': 0.8, 'ry': 0.8, 'rz': 0.8},   # Flaring out
        {'z': 4.5, 'rx': 1.0, 'ry': 1.0, 'rz': 0.8},   # Top opening
    ]

    objects = []
    for i, data in enumerate(profile_data):
        obj = placer.create_primitive(
            primitive_type='ELLIPSOID',
            location=(0, 0, data['z']),
            scale=(data['rx'], data['ry'], data['rz'])
        )
        obj.name = f"VaseSegment_{i:02d}"
        objects.append(obj)

    print(f"  ✓ Created {len(objects)} ellipsoid segments")

    # Join them into a vase
    joiner = MeshJoiner()
    vase = joiner.join_with_boolean_union(objects, target_name="Vase")

    assert vase is not None, "Failed to create vase"
    assert vase.type == 'MESH', "Vase is not a mesh"
    print(f"  ✓ Created vase with {len(vase.data.vertices)} vertices")

    print("✅ Test 4 PASSED: Vase-like shape created from ellipsoids\n")
    return vase


def run_all_tests():
    """Run all ellipsoid tests."""
    print("\n" + "="*60)
    print("ELLIPSOID PRIMITIVE TEST SUITE")
    print("="*60)

    try:
        test_spawn_ellipsoid()
        test_ellipsoid_in_primitive_placer()
        test_boolean_union_with_ellipsoids()
        test_ellipsoid_stacked_creation()

        print("="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
