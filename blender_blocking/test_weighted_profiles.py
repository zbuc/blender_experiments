"""
Test weighted profile extraction approach.

Validates that:
1. Profile extraction at cardinal angles works
2. Opposite view averaging produces reasonable results
3. Combined profile suitable for SliceAnalyzer
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import importlib.util

def import_module_from_path(module_name, file_path):
    """Import module from file path bypassing __init__.py imports."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import weighted profile extractor
weighted_extractor = import_module_from_path(
    "weighted_profile_extractor",
    Path(__file__).parent / "integration" / "shape_matching" / "weighted_profile_extractor.py"
)

import bpy
import numpy as np
from mathutils import Vector


def clear_scene():
    """Clear all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def create_test_vase():
    """Create test vase (same as ground truth test)."""
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=2.0, location=(0, 0, 0))
    vase = bpy.context.active_object
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.transform.resize(value=(1.2, 1.2, 1.0), constraint_axis=(True, True, False))
    bpy.ops.object.mode_set(mode='OBJECT')
    return vase


def test_weighted_profiles():
    """Test weighted profile extraction."""
    print("\n" + "="*70)
    print("WEIGHTED PROFILE EXTRACTION TEST")
    print("="*70)

    print("\nGoal: Validate weighted profile approach")
    print("Expected: Orthogonal profiles at cardinal angles")
    print("Expected: Radii ~0.0-0.6 for vase (radius 0.5 * 1.2 = 0.6)")

    # Create test vase
    print("\n" + "="*70)
    print("STEP 1: Create Test Vase")
    print("="*70)

    clear_scene()
    vase = create_test_vase()
    print("✓ Created vase mesh")
    print(f"  Expected radius: ~0.6")
    print(f"  Expected height: 2.0")

    # Extract weighted profiles
    print("\n" + "="*70)
    print("STEP 2: Extract Weighted Profiles")
    print("="*70)

    bounds_min = Vector((-2.0, -2.0, -2.0))
    bounds_max = Vector((2.0, 2.0, 2.0))

    profiles = weighted_extractor.extract_weighted_profiles(
        vase,
        num_heights=20,
        bounds_min=bounds_min,
        bounds_max=bounds_max
    )

    print(f"\n✓ Extraction complete")
    print(f"  Front profile: {len(profiles['front'])} samples")
    print(f"  Side profile: {len(profiles['side'])} samples")
    print(f"  All profiles: {len(profiles['all'])} angles")

    # Analyze profiles
    print("\n" + "="*70)
    print("STEP 3: Analyze Profile Quality")
    print("="*70)

    front_radii = [r for _, r in profiles['front']]
    side_radii = [r for _, r in profiles['side']]

    print(f"\nFront Profile (0° + 180° averaged):")
    print(f"  Min radius: {min(front_radii):.4f}")
    print(f"  Max radius: {max(front_radii):.4f}")
    print(f"  Mean radius: {np.mean(front_radii):.4f}")

    print(f"\nSide Profile (90° + 270° averaged):")
    print(f"  Min radius: {min(side_radii):.4f}")
    print(f"  Max radius: {max(side_radii):.4f}")
    print(f"  Mean radius: {np.mean(side_radii):.4f}")

    # Expected values for vase (radius 0.5 * scale 1.2 = 0.6)
    expected_max_radius = 0.6
    tolerance = 0.1  # 10% tolerance

    # Check front profile
    front_max = max(front_radii)
    front_error = abs(front_max - expected_max_radius) / expected_max_radius

    print(f"\nFront Profile Assessment:")
    print(f"  Expected max radius: {expected_max_radius:.3f}")
    print(f"  Measured max radius: {front_max:.3f}")
    print(f"  Error: {front_error*100:.1f}%")

    if front_error < tolerance:
        print(f"  ✓ PASS: Within {tolerance*100:.0f}% tolerance")
    else:
        print(f"  ✗ FAIL: Exceeds {tolerance*100:.0f}% tolerance")

    # Check side profile
    side_max = max(side_radii)
    side_error = abs(side_max - expected_max_radius) / expected_max_radius

    print(f"\nSide Profile Assessment:")
    print(f"  Expected max radius: {expected_max_radius:.3f}")
    print(f"  Measured max radius: {side_max:.3f}")
    print(f"  Error: {side_error*100:.1f}%")

    if side_error < tolerance:
        print(f"  ✓ PASS: Within {tolerance*100:.0f}% tolerance")
    else:
        print(f"  ✗ FAIL: Exceeds {tolerance*100:.0f}% tolerance")

    # Combine profiles using different methods
    print("\n" + "="*70)
    print("STEP 4: Test Profile Combination Methods")
    print("="*70)

    for method in ['max', 'mean', 'geometric_mean']:
        combined = weighted_extractor.combine_orthogonal_profiles(
            profiles['front'],
            profiles['side'],
            method=method
        )

        combined_radii = [r for _, r in combined]
        print(f"\n{method.upper()} method:")
        print(f"  Min radius: {min(combined_radii):.4f}")
        print(f"  Max radius: {max(combined_radii):.4f}")
        print(f"  Mean radius: {np.mean(combined_radii):.4f}")

    # Overall assessment
    print("\n" + "="*70)
    print("OVERALL ASSESSMENT")
    print("="*70)

    if front_error < tolerance and side_error < tolerance:
        print("\n✓ SUCCESS: Weighted profile extraction produces accurate measurements")
        print("  Both front and side profiles within tolerance")
        print("  Ready for integration with SliceAnalyzer")
    else:
        print("\n⚠ WARNING: Profile measurements outside tolerance")
        print("  May need coordinate system adjustments")
        print("  Check bounds and mesh scale")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_weighted_profiles()
