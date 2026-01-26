"""
End-to-end test of weighted profile approach with SliceAnalyzer.

Tests researcher's hypothesis (hq-vqxe):
- SliceAnalyzer may compensate for 2x-scaled input internally
- End-to-end IoU could be correct despite input scaling error

Pipeline:
1. Create ground truth vase
2. Voxelize for IoU measurement
3. Extract weighted profiles (front + side, 2x-scaled)
4. Combine into single profile
5. Feed to SliceAnalyzer
6. Place primitives
7. Measure IoU

Success: IoU ≥ 0.85
Fallback: IoU < 0.80 (try empirical 2x scaling or debug)
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
from placement.primitive_placement import SliceAnalyzer, PrimitivePlacer, MeshJoiner


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


def voxelize_mesh_ground_truth(mesh_obj, resolution=128):
    """Voxelize mesh for ground truth IoU measurement."""
    bounds_min = Vector((-2.0, -2.0, -2.0))
    bounds_max = Vector((2.0, 2.0, 2.0))

    bounds_size = bounds_max - bounds_min
    voxel_size = bounds_size / resolution

    voxel_grid = np.zeros((resolution, resolution, resolution), dtype=bool)

    print(f"  Voxelizing at {resolution}³ resolution...")
    total_voxels = resolution ** 3
    voxels_inside = 0

    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                voxel_pos = bounds_min + Vector((i + 0.5, j + 0.5, k + 0.5)) * voxel_size

                # Raycast from 3 directions (majority voting)
                hits = 0
                for direction in [Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))]:
                    ray_origin = voxel_pos - direction * 10
                    result, location, normal, index = mesh_obj.ray_cast(ray_origin, direction)
                    if result:
                        hits += 1

                if hits >= 2:
                    voxel_grid[i, j, k] = True
                    voxels_inside += 1

    print(f"  ✓ Voxelization complete: {voxels_inside:,} / {total_voxels:,} voxels")

    return voxel_grid, bounds_min, bounds_max


def compute_iou(grid_a, grid_b):
    """Compute IoU between two voxel grids."""
    intersection = np.logical_and(grid_a, grid_b).sum()
    union = np.logical_or(grid_a, grid_b).sum()
    return intersection / union if union > 0 else 0.0


def test_weighted_profiles_endtoend():
    """Test weighted profile end-to-end pipeline."""
    print("\n" + "="*70)
    print("WEIGHTED PROFILE END-TO-END TEST")
    print("="*70)

    print("\nResearcher's Hypothesis (hq-vqxe):")
    print("SliceAnalyzer may compensate for 2x-scaled input internally")
    print("\nPipeline: Weighted Profiles → SliceAnalyzer → Primitives → IoU")
    print("Success: IoU ≥ 0.85")
    print("Fallback: IoU < 0.80")

    resolution = 128
    bounds_min = Vector((-2.0, -2.0, -2.0))
    bounds_max = Vector((2.0, 2.0, 2.0))

    # Step 1: Create ground truth vase
    print("\n" + "="*70)
    print("STEP 1: Create Ground Truth Vase")
    print("="*70)

    clear_scene()
    vase = create_test_vase()
    print("✓ Created vase mesh")

    # Voxelize for ground truth
    ground_truth_voxels, gt_bounds_min, gt_bounds_max = voxelize_mesh_ground_truth(
        vase, resolution=resolution
    )
    print(f"✓ Ground truth: {ground_truth_voxels.sum():,} voxels ({ground_truth_voxels.sum()/ground_truth_voxels.size*100:.1f}%)")

    # Step 2: Extract weighted profiles
    print("\n" + "="*70)
    print("STEP 2: Extract Weighted Profiles")
    print("="*70)

    profiles = weighted_extractor.extract_weighted_profiles(
        vase,
        num_heights=20,
        bounds_min=bounds_min,
        bounds_max=bounds_max
    )

    front_radii = [r for _, r in profiles['front']]
    side_radii = [r for _, r in profiles['side']]

    print(f"\n✓ Profiles extracted:")
    print(f"  Front: {min(front_radii):.3f} to {max(front_radii):.3f} (mean {np.mean(front_radii):.3f})")
    print(f"  Side: {min(side_radii):.3f} to {max(side_radii):.3f} (mean {np.mean(side_radii):.3f})")
    print(f"  Note: Radii are 2x too small (expected max ~0.6, got ~0.3)")

    # Step 3: Combine orthogonal profiles
    print("\n" + "="*70)
    print("STEP 3: Combine Front/Side Profiles")
    print("="*70)

    # Try all combination methods to see which works best
    for method in ['max', 'mean', 'geometric_mean']:
        combined = weighted_extractor.combine_orthogonal_profiles(
            profiles['front'],
            profiles['side'],
            method=method
        )

        combined_radii = [r for _, r in combined]
        print(f"\n{method.upper()} method:")
        print(f"  Radius: {min(combined_radii):.3f} to {max(combined_radii):.3f} (mean {np.mean(combined_radii):.3f})")

    # Use MAX method for actual pipeline (most conservative)
    combined_profile = weighted_extractor.combine_orthogonal_profiles(
        profiles['front'],
        profiles['side'],
        method='max'
    )
    print(f"\nUsing MAX method for SliceAnalyzer")

    # Step 4: SliceAnalyzer + Primitive Placement
    print("\n" + "="*70)
    print("STEP 4: SliceAnalyzer + Primitive Placement")
    print("="*70)

    analyzer = SliceAnalyzer(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        num_slices=20,
        vertical_profile=combined_profile
    )

    # Analyze slices
    slice_data = []
    for i in range(20):
        z = -1.0 + (i / 19.0) * 2.0
        slice_info = analyzer.analyze_slice(z)
        slice_data.append(slice_info)

    print(f"✓ Analyzed {len(slice_data)} slices")

    # Place primitives
    placer = PrimitivePlacer()
    objects = placer.place_primitives_from_slices(slice_data, primitive_type='CYLINDER')
    print(f"✓ Placed {len(objects)} cylinder primitives")

    # Join primitives
    if len(objects) > 0:
        final_mesh = MeshJoiner.join_with_boolean_union(objects, target_name="WeightedProfile_Primitives")
        print(f"✓ Joined into single mesh: {final_mesh.name}")
        print(f"  Vertices: {len(final_mesh.data.vertices):,}")
        print(f"  Faces: {len(final_mesh.data.polygons):,}")
    else:
        print("✗ No primitives placed - CRITICAL FAILURE")
        return

    # Step 5: Measure IoU
    print("\n" + "="*70)
    print("STEP 5: Measure IoU")
    print("="*70)

    primitives_voxels, _, _ = voxelize_mesh_ground_truth(final_mesh, resolution=resolution)
    print(f"✓ Primitive mesh voxels: {primitives_voxels.sum():,}")

    iou_primitives = compute_iou(primitives_voxels, ground_truth_voxels)
    print(f"✓ Primitives IoU vs ground truth: {iou_primitives:.4f}")

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\nGround Truth: {ground_truth_voxels.sum():,} voxels")
    print(f"Weighted Profile Primitives: {primitives_voxels.sum():,} voxels")
    print(f"IoU: {iou_primitives:.4f}")

    # Assessment
    baseline_iou = 0.875  # Known single-profile baseline
    target_iou = 0.85     # Success threshold

    print(f"\nComparison to Baseline:")
    print(f"  Single-profile baseline: {baseline_iou:.3f} IoU")
    print(f"  Weighted profile result: {iou_primitives:.4f} IoU")
    print(f"  Improvement: {iou_primitives - baseline_iou:+.4f} ({(iou_primitives - baseline_iou)/baseline_iou*100:+.1f}%)")

    if iou_primitives >= target_iou:
        print(f"\n✓ SUCCESS: IoU ≥ {target_iou:.2f}")
        print("  Weighted profile approach works!")
        print("  SliceAnalyzer compensates for 2x scaling (researcher's hypothesis CORRECT)")
        print("  Ready for production integration")
    elif iou_primitives >= 0.80:
        print(f"\n≈ MARGINAL: {0.80:.2f} ≤ IoU < {target_iou:.2f}")
        print("  Close to success, may need minor adjustments")
        print("  Consider empirical 2x scaling factor")
    else:
        print(f"\n✗ FAILURE: IoU < 0.80")
        print("  Weighted profile approach doesn't improve over baseline")
        print("  Next steps:")
        print("  1. Try empirical 2x scaling (multiply radii by 2.0)")
        print("  2. Debug extract_profile_at_angle coordinate transforms")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_weighted_profiles_endtoend()
