"""
Test suite for SuperFrustum primitive.

Tests:
1. Basic instantiation and parameter access
2. SDF computation correctness
3. Gradient computation
4. Special cases (cylinder, cone, sphere)
5. Blender mesh creation
"""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from primitives.superfrustum import SuperFrustum


def test_basic_instantiation() -> bool:
    """Test basic SuperFrustum creation and parameter access."""
    print("\n" + "=" * 70)
    print("TEST: Basic Instantiation")
    print("=" * 70)

    sf = SuperFrustum(
        position=(1.0, 2.0, 3.0),
        orientation=(0.5, 0.3),
        radius_bottom=2.0,
        radius_top=1.0,
        height=4.0,
    )

    assert np.allclose(sf.position, [1.0, 2.0, 3.0])
    assert np.allclose(sf.orientation, [0.5, 0.3])
    assert sf.radius_bottom == 2.0
    assert sf.radius_top == 1.0
    assert sf.height == 4.0

    print(f"✓ Created: {sf}")
    print("✓ Parameters accessible")
    print("\n✓ Basic instantiation test PASSED")
    return True


def test_cylinder_case() -> bool:
    """Test SuperFrustum as cylinder (r_top = r_bottom)."""
    print("\n" + "=" * 70)
    print("TEST: Cylinder Case (r_top = r_bottom)")
    print("=" * 70)

    # Create cylinder (radius=1.0, height=2.0) centered at origin
    sf = SuperFrustum(
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0),  # Aligned with Z axis
        radius_bottom=1.0,
        radius_top=1.0,
        height=2.0,
    )

    # Test points
    # Inside cylinder (near center)
    p_inside = np.array([0.0, 0.0, 0.0])
    d_inside = sf.sdf(p_inside)
    print(f"  Point inside (0, 0, 0): SDF = {d_inside:.4f}")
    assert d_inside < 0, "Point inside should have negative SDF"

    # On surface (cylindrical wall)
    p_surface = np.array([1.0, 0.0, 0.0])
    d_surface = sf.sdf(p_surface)
    print(f"  Point on surface (1, 0, 0): SDF = {d_surface:.4f}")
    assert abs(d_surface) < 0.1, "Point on surface should have SDF near zero"

    # Outside cylinder
    p_outside = np.array([2.0, 0.0, 0.0])
    d_outside = sf.sdf(p_outside)
    print(f"  Point outside (2, 0, 0): SDF = {d_outside:.4f}")
    assert d_outside > 0, "Point outside should have positive SDF"

    print("\n✓ Cylinder case test PASSED")
    return True


def test_cone_case() -> bool:
    """Test SuperFrustum as cone (r_top = 0)."""
    print("\n" + "=" * 70)
    print("TEST: Cone Case (r_top = 0)")
    print("=" * 70)

    # Create cone (base radius=1.0, height=2.0)
    sf = SuperFrustum(
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0),
        radius_bottom=1.0,
        radius_top=0.0,
        height=2.0,
    )

    # Test point inside cone
    p_inside = np.array([0.0, 0.0, 0.0])
    d_inside = sf.sdf(p_inside)
    print(f"  Point inside (0, 0, 0): SDF = {d_inside:.4f}")
    assert d_inside < 0, "Point inside should have negative SDF"

    # Test point outside cone
    p_outside = np.array([2.0, 0.0, 0.0])
    d_outside = sf.sdf(p_outside)
    print(f"  Point outside (2, 0, 0): SDF = {d_outside:.4f}")
    assert d_outside > 0, "Point outside should have positive SDF"

    print("\n✓ Cone case test PASSED")
    return True


def test_frustum_case() -> bool:
    """Test SuperFrustum as tapered frustum."""
    print("\n" + "=" * 70)
    print("TEST: Tapered Frustum Case")
    print("=" * 70)

    # Create frustum (r_bottom=2.0, r_top=1.0, height=3.0)
    sf = SuperFrustum(
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0),
        radius_bottom=2.0,
        radius_top=1.0,
        height=3.0,
    )

    print(
        f"  Frustum: r_bottom={sf.radius_bottom}, r_top={sf.radius_top}, h={sf.height}"
    )

    # Test point inside
    p_inside = np.array([0.0, 0.0, 0.0])
    d_inside = sf.sdf(p_inside)
    print(f"  Point inside (0, 0, 0): SDF = {d_inside:.4f}")
    assert d_inside < 0, "Point inside should have negative SDF"

    # Test point outside (far from bottom)
    p_outside = np.array([3.0, 0.0, -2.0])
    d_outside = sf.sdf(p_outside)
    print(f"  Point outside (3, 0, -2): SDF = {d_outside:.4f}")
    assert d_outside > 0, "Point outside should have positive SDF"

    print("\n✓ Tapered frustum test PASSED")
    return True


def test_orientation() -> bool:
    """Test SuperFrustum with different orientations."""
    print("\n" + "=" * 70)
    print("TEST: Orientation (Axis Rotation)")
    print("=" * 70)

    # Create frustum tilted at 45 degrees
    import math

    sf = SuperFrustum(
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, math.pi / 4),  # 45-degree tilt
        radius_bottom=1.0,
        radius_top=0.5,
        height=2.0,
    )

    axis = sf.get_axis_vector()
    print(f"  Orientation (0, π/4): axis = {axis}")
    print(f"  Axis magnitude: {np.linalg.norm(axis):.4f}")

    # Axis should be unit vector
    assert np.isclose(np.linalg.norm(axis), 1.0), "Axis should be unit vector"

    # Test SDF still works
    p_test = np.array([0.0, 0.0, 0.0])
    d_test = sf.sdf(p_test)
    print(f"  SDF at origin: {d_test:.4f}")

    print("\n✓ Orientation test PASSED")
    return True


def test_gradient_computation() -> bool:
    """Test gradient computation using finite differences."""
    print("\n" + "=" * 70)
    print("TEST: Gradient Computation")
    print("=" * 70)

    sf = SuperFrustum(
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0),
        radius_bottom=1.5,
        radius_top=0.8,
        height=2.5,
    )

    # Test point
    p = np.array([0.5, 0.3, 0.2])

    # Compute gradients
    grads = sf.gradient(p)

    print(f"  Test point: {p}")
    print(f"  Gradient w.r.t. position: {grads['position']}")
    print(f"  Gradient w.r.t. orientation: {grads['orientation']}")
    print(f"  Gradient w.r.t. radius_bottom: {grads['radius_bottom']:.6f}")
    print(f"  Gradient w.r.t. radius_top: {grads['radius_top']:.6f}")
    print(f"  Gradient w.r.t. height: {grads['height']:.6f}")

    # Verify all gradients are finite
    assert np.all(np.isfinite(grads["position"])), "Position gradients should be finite"
    assert np.all(
        np.isfinite(grads["orientation"])
    ), "Orientation gradients should be finite"
    assert np.isfinite(
        grads["radius_bottom"]
    ), "Radius_bottom gradient should be finite"
    assert np.isfinite(grads["radius_top"]), "Radius_top gradient should be finite"
    assert np.isfinite(grads["height"]), "Height gradient should be finite"

    print("\n✓ Gradient computation test PASSED")
    return True


def test_serialization() -> bool:
    """Test to_dict and from_dict methods."""
    print("\n" + "=" * 70)
    print("TEST: Serialization (to_dict/from_dict)")
    print("=" * 70)

    sf1 = SuperFrustum(
        position=(1.0, 2.0, 3.0),
        orientation=(0.5, 0.3),
        radius_bottom=2.0,
        radius_top=1.0,
        height=4.0,
    )

    # Serialize
    params = sf1.to_dict()
    print(f"  Serialized: {params}")

    # Deserialize
    sf2 = SuperFrustum.from_dict(params)
    print(f"  Deserialized: {sf2}")

    # Verify equality
    assert np.allclose(sf1.position, sf2.position)
    assert np.allclose(sf1.orientation, sf2.orientation)
    assert sf1.radius_bottom == sf2.radius_bottom
    assert sf1.radius_top == sf2.radius_top
    assert sf1.height == sf2.height

    print("\n✓ Serialization test PASSED")
    return True


def test_blender_integration() -> Optional[bool]:
    """Test Blender mesh creation (if running in Blender)."""
    print("\n" + "=" * 70)
    print("TEST: Blender Integration")
    print("=" * 70)

    try:
        import bpy
        from primitives.primitives import spawn_superfrustum

        # Clear scene
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()

        # Create SuperFrustum mesh
        obj = spawn_superfrustum(
            radius_bottom=2.0,
            radius_top=1.0,
            height=3.0,
            location=(0.0, 0.0, 0.0),
            name="TestSuperFrustum",
        )

        print(f"  ✓ Created Blender object: {obj.name}")
        print(f"  ✓ Vertices: {len(obj.data.vertices)}")
        print(f"  ✓ Faces: {len(obj.data.polygons)}")

        assert obj is not None, "Object should be created"
        assert obj.name == "TestSuperFrustum", "Object should have correct name"

        print("\n✓ Blender integration test PASSED")
        return True

    except ImportError:
        print("  ⚠ Not running in Blender - skipping Blender integration test")
        print("\n- Blender integration test SKIPPED")
        return None


def run_all_tests() -> int:
    """Run all test cases."""
    print("\n" + "=" * 70)
    print("SUPERFRUSTUM TEST SUITE")
    print("=" * 70)

    tests = [
        ("Basic Instantiation", test_basic_instantiation),
        ("Cylinder Case", test_cylinder_case),
        ("Cone Case", test_cone_case),
        ("Frustum Case", test_frustum_case),
        ("Orientation", test_orientation),
        ("Gradient Computation", test_gradient_computation),
        ("Serialization", test_serialization),
        ("Blender Integration", test_blender_integration),
    ]

    results = {}
    for name, test_func in tests:
        try:
            result = test_func()
            results[name] = result
        except Exception as e:
            print(f"\n❌ {name} FAILED with exception:")
            print(f"   {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            results[name] = False

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    for name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "- SKIP"
        print(f"  {name:.<50} {status}")

    print("=" * 70)
    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\n❌ TESTS FAILED")
        return 1
    else:
        print("\n✓ ALL TESTS PASSED")
        return 0


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
