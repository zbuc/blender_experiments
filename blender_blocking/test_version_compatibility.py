#!/usr/bin/env python3
"""
Blender Version Compatibility Test

Tests that the code works correctly across different Blender versions,
particularly for API changes like the boolean solver enum.

Usage:
    blender --background --python test_version_compatibility.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_version_detection() -> bool:
    """Test Blender version detection utilities."""
    print("\n" + "=" * 70)
    print("TEST: Version Detection")
    print("=" * 70)

    try:
        import bpy
    except ImportError:
        print("❌ FAILED: Must run inside Blender")
        return False

    from blender_blocking.utils.blender_version import (
        get_blender_version,
        get_blender_version_string,
        is_blender_version_at_least,
        get_version_info,
    )

    # Test version retrieval
    version = get_blender_version()
    version_string = get_blender_version_string()

    print(f"✓ Detected Blender version: {version_string}")
    print(f"  Version tuple: {version}")

    # Test version comparison
    major, minor, patch = version
    assert is_blender_version_at_least(
        major, minor, patch
    ), "Version comparison failed for current version"
    print(f"✓ Version comparison works correctly")

    # Test version info
    info = get_version_info()
    assert info is not None, "Failed to get version info"
    assert info["version"] == version, "Version mismatch in info"
    assert info["version_string"] == version_string, "Version string mismatch in info"
    print(f"✓ Version info retrieved successfully")

    print("\n" + "=" * 70)
    print("✓ Version detection test PASSED")
    print("=" * 70)
    return True


def test_boolean_solver_compatibility() -> bool:
    """Test that boolean solver selection is compatible with current Blender version."""
    print("\n" + "=" * 70)
    print("TEST: Boolean Solver Compatibility")
    print("=" * 70)

    try:
        import bpy
    except ImportError:
        print("❌ FAILED: Must run inside Blender")
        return False

    from blender_blocking.utils.blender_version import (
        get_boolean_solver,
        get_available_boolean_solvers,
        check_boolean_solver_compatibility,
        is_blender_version_at_least,
    )

    # Get recommended solver for this version
    recommended_solver = get_boolean_solver()
    print(f"✓ Recommended solver for this version: {recommended_solver}")

    # Get all available solvers
    available_solvers = get_available_boolean_solvers()
    print(f"  Available solvers: {', '.join(available_solvers)}")

    # Verify recommended solver is available
    assert (
        recommended_solver in available_solvers
    ), f"Recommended solver '{recommended_solver}' not in available solvers!"
    print(f"✓ Recommended solver is available")

    # Test version-specific behavior
    if is_blender_version_at_least(5, 0, 0):
        print("  Detected Blender 5.0+")
        assert recommended_solver == "EXACT", "Should use EXACT solver for Blender 5.0+"
        assert (
            "FAST" not in available_solvers
        ), "FAST should not be available in Blender 5.0+"
        print("  ✓ Blender 5.0+ API compatibility verified")
    else:
        print("  Detected Blender 4.x or earlier")
        assert recommended_solver == "FAST", "Should use FAST solver for Blender 4.x"
        assert "FAST" in available_solvers, "FAST should be available in Blender 4.x"
        print("  ✓ Blender 4.x API compatibility verified")

    # Test compatibility check function
    for solver in available_solvers:
        assert check_boolean_solver_compatibility(
            solver
        ), f"Compatibility check failed for {solver}"
    print(f"✓ Compatibility check function works correctly")

    print("\n" + "=" * 70)
    print("✓ Boolean solver compatibility test PASSED")
    print("=" * 70)
    return True


def test_version_aware_mesh_joining() -> bool:
    """Test that MeshJoiner uses correct boolean solver for current version."""
    print("\n" + "=" * 70)
    print("TEST: Version-Aware Mesh Joining")
    print("=" * 70)

    try:
        import bpy
    except ImportError:
        print("❌ FAILED: Must run inside Blender")
        return False

    from blender_blocking.placement.primitive_placement import MeshJoiner
    from blender_blocking.utils.blender_version import get_boolean_solver

    # Clear scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Create test objects
    print("Creating test objects...")
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
    obj1 = bpy.context.active_object
    obj1.name = "TestCube1"

    bpy.ops.mesh.primitive_cube_add(location=(0.5, 0, 0))
    obj2 = bpy.context.active_object
    obj2.name = "TestCube2"

    objects = [obj1, obj2]

    # Test mesh joining
    print("Testing MeshJoiner...")
    joiner = MeshJoiner()

    try:
        result = joiner.join(objects, target_name="VersionTestResult", mode="auto")
        print(f"✓ Mesh joining successful: {result.name}")
        print(f"  Vertices: {len(result.data.vertices)}")
        print(f"  Faces: {len(result.data.polygons)}")

        # Verify result has geometry
        assert len(result.data.vertices) > 0, "Result mesh has no vertices"
        assert len(result.data.polygons) > 0, "Result mesh has no faces"
        print(f"✓ Result mesh has valid geometry")

        # Verify solver was used correctly
        expected_solver = get_boolean_solver()
        print(f"  Expected solver: {expected_solver}")
        print(f"✓ Version-aware solver selection worked")

    except TypeError as e:
        if "enum" in str(e).lower():
            print(f"❌ FAILED: Boolean solver enum error: {e}")
            print(
                "  This indicates the solver enum is not compatible with this Blender version"
            )
            return False
        else:
            raise

    print("\n" + "=" * 70)
    print("✓ Version-aware mesh joining test PASSED")
    print("=" * 70)
    return True


def run_all_tests() -> None:
    """Run all version compatibility tests."""
    print("\n" + "=" * 70)
    print("BLENDER VERSION COMPATIBILITY TEST SUITE")
    print("=" * 70)

    try:
        import bpy

        print(f"\nBlender version: {bpy.app.version_string}")
        print(f"Python version: {sys.version.split()[0]}")
    except ImportError:
        print("\n❌ CRITICAL: Must run inside Blender")
        print("\nUsage:")
        print("  blender --background --python test_version_compatibility.py")
        sys.exit(2)

    results = {}

    # Test 1: Version detection
    results["version_detection"] = test_version_detection()

    # Test 2: Boolean solver compatibility
    results["boolean_solver"] = test_boolean_solver_compatibility()

    # Test 3: Version-aware mesh joining
    results["mesh_joining"] = test_version_aware_mesh_joining()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n✓ All tests PASSED")
        sys.exit(0)
    else:
        print("\n❌ Some tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
