"""
Test Phase 2 multi-profile approach using GROUND TRUTH mesh.

This bypasses Visual Hull mesh creation issues and tests the core question:
"Does multi-profile extraction improve IoU over single-profile?"

Pipeline:
1. Create ground truth vase mesh
2. Voxelize for IoU measurement
3. Extract multi-angle profiles FROM GROUND TRUTH MESH (not Visual Hull!)
4. Feed to SliceAnalyzer
5. Place primitives
6. Measure IoU improvement

If this shows improvement, Phase 2 concept is validated.
If not, the multi-profile approach itself has issues.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple
from types import ModuleType

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import using importlib to bypass cv2 dependency
import importlib.util


def import_module_from_path(module_name: str, file_path: Path) -> ModuleType:
    """Import module from file path bypassing __init__.py imports."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import mesh_profile_extractor
profile_extractor = import_module_from_path(
    "mesh_profile_extractor",
    Path(__file__).parent
    / "integration"
    / "shape_matching"
    / "mesh_profile_extractor.py",
)

import bpy
import numpy as np
from mathutils import Vector
from integration.blender_ops.raycast_utils import ray_cast_world


def clear_scene() -> None:
    """Clear all objects from scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def create_test_mesh_vase() -> bpy.types.Object:
    """Create vase for testing (same as integration test)."""
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=2.0, location=(0, 0, 0))
    vase = bpy.context.active_object
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.transform.resize(value=(1.2, 1.2, 1.0), constraint_axis=(True, True, False))
    bpy.ops.object.mode_set(mode="OBJECT")
    return vase


def voxelize_mesh_ground_truth(
    mesh_obj: bpy.types.Object, resolution: int = 128
) -> Tuple[np.ndarray, Vector, Vector]:
    """Voxelize mesh for ground truth IoU measurement."""
    # Use fixed large bounds (same as integration test)
    bounds_min = Vector((-2.0, -2.0, -2.0))
    bounds_max = Vector((2.0, 2.0, 2.0))

    bounds_size = bounds_max - bounds_min
    voxel_size = bounds_size / resolution

    voxel_grid = np.zeros((resolution, resolution, resolution), dtype=bool)

    print(f"  Voxelizing at {resolution}³ resolution...")
    total_voxels = resolution**3
    progress_every = max(total_voxels // 10, 1)
    voxels_processed = 0
    voxels_inside = 0

    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                voxel_pos = (
                    bounds_min + Vector((i + 0.5, j + 0.5, k + 0.5)) * voxel_size
                )

                # Raycast from 3 directions (majority voting)
                hits = 0
                for direction in [
                    Vector((1, 0, 0)),
                    Vector((0, 1, 0)),
                    Vector((0, 0, 1)),
                ]:
                    ray_origin = voxel_pos - direction * 10
                    result, location, normal, index = ray_cast_world(
                        mesh_obj, ray_origin, direction, 20.0
                    )
                    if result:
                        hits += 1

                if hits >= 2:
                    voxel_grid[i, j, k] = True
                    voxels_inside += 1

                voxels_processed += 1
                if voxels_processed % progress_every == 0:
                    progress = (voxels_processed / total_voxels) * 100
                    print(
                        f"    Progress: {progress:.0f}% ({voxels_inside:,} voxels inside)"
                    )

    print(f"  ✓ Voxelization complete: {voxels_inside:,} / {total_voxels:,} voxels")

    return voxel_grid, bounds_min, bounds_max


def compute_iou(grid_a: np.ndarray, grid_b: np.ndarray) -> float:
    """Compute IoU between two voxel grids."""
    intersection = np.logical_and(grid_a, grid_b).sum()
    union = np.logical_or(grid_a, grid_b).sum()
    return intersection / union if union > 0 else 0.0


def test_groundtruth_multiprofile() -> None:
    """Test multi-profile approach using ground truth mesh."""
    print("\n" + "=" * 70)
    print("PHASE 2 GROUND TRUTH TEST")
    print("=" * 70)
    print("\nTest: Multi-profile extraction from GROUND TRUTH mesh")
    print("Goal: Validate Phase 2 concept independent of Visual Hull issues")

    resolution = 128
    num_profiles = 12

    # Step 1: Create ground truth vase
    print("\n" + "=" * 70)
    print("STEP 1: Create Ground Truth Vase")
    print("=" * 70)

    clear_scene()
    ground_truth_vase = create_test_mesh_vase()
    print("✓ Created vase mesh")
    print(f"  Dimensions: radius ~0.6, height 2.0")
    print(f"  Bbox: approximately (1.2, 1.2, 2.0)")

    # Voxelize for ground truth
    ground_truth_voxels, gt_bounds_min, gt_bounds_max = voxelize_mesh_ground_truth(
        ground_truth_vase, resolution=resolution
    )
    print(
        f"✓ Ground truth: {ground_truth_voxels.sum():,} voxels ({ground_truth_voxels.sum()/ground_truth_voxels.size*100:.1f}%)"
    )

    # Step 2: Extract multi-angle profiles from GROUND TRUTH (not Visual Hull!)
    print("\n" + "=" * 70)
    print("STEP 2: Extract Multi-Angle Profiles from Ground Truth")
    print("=" * 70)
    print(f"Extracting {num_profiles} profiles at different angles...")

    profiles = profile_extractor.extract_multi_angle_profiles(
        ground_truth_vase,
        num_angles=num_profiles,
        num_heights=20,
        bounds_min=gt_bounds_min,
        bounds_max=gt_bounds_max,
    )
    print(f"✓ Extracted {len(profiles)} profiles")

    # Combine profiles using median
    combined_profile = profile_extractor.combine_profiles(profiles, method="median")
    print(f"✓ Combined profile: {len(combined_profile)} samples")

    radii = [r for _, r in combined_profile]
    print(f"  Radius range: {min(radii):.3f} to {max(radii):.3f}")
    print(f"  Mean radius: {np.mean(radii):.3f}")

    # Step 3: Use SliceAnalyzer with multi-profile
    print("\n" + "=" * 70)
    print("STEP 3: SliceAnalyzer + Primitive Placement")
    print("=" * 70)

    from placement.primitive_placement import SliceAnalyzer, PrimitivePlacer, MeshJoiner

    # Create SliceAnalyzer with combined profile
    analyzer = SliceAnalyzer(
        bounds_min=gt_bounds_min,
        bounds_max=gt_bounds_max,
        num_slices=20,
        vertical_profile=combined_profile,
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
    objects = placer.place_primitives_from_slices(slice_data, primitive_type="CYLINDER")
    print(f"✓ Placed {len(objects)} cylinder primitives")

    # Join primitives
    if len(objects) > 0:
        final_mesh = MeshJoiner.join_with_boolean_union(
            objects, target_name="GroundTruthMultiProfile_Primitives"
        )
        print(f"✓ Joined into single mesh: {final_mesh.name}")
        print(f"  Vertices: {len(final_mesh.data.vertices):,}")
        print(f"  Faces: {len(final_mesh.data.polygons):,}")
    else:
        print("✗ No primitives placed")
        return

    # Step 4: Measure IoU
    print("\n" + "=" * 70)
    print("STEP 4: Measure IoU")
    print("=" * 70)

    primitives_voxels, _, _ = voxelize_mesh_ground_truth(
        final_mesh, resolution=resolution
    )
    print(f"✓ Primitive mesh voxels: {primitives_voxels.sum():,}")

    iou_primitives = compute_iou(primitives_voxels, ground_truth_voxels)
    print(f"✓ Primitives IoU vs ground truth: {iou_primitives:.4f}")

    # Compare to baseline (what would single-profile achieve?)
    # For reference, Visual Hull alone achieves 0.7814 IoU
    # Current pipeline with single profile achieves ~0.875 IoU

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nGround Truth: {ground_truth_voxels.sum():,} voxels")
    print(f"\nMulti-Profile Primitives:")
    print(f"  Voxels: {primitives_voxels.sum():,}")
    print(f"  IoU: {iou_primitives:.4f}")

    # Assessment
    baseline_iou = 0.875  # Known single-profile baseline from existing pipeline
    improvement = iou_primitives - baseline_iou

    print(f"\nComparison to Single-Profile Baseline (~{baseline_iou:.3f} IoU):")
    print(f"  Improvement: {improvement:+.4f} ({improvement/baseline_iou*100:+.1f}%)")

    if iou_primitives > baseline_iou + 0.02:
        print(f"\n  ✓ SIGNIFICANT IMPROVEMENT: Multi-profile approach works!")
        print(f"    Phase 2 concept validated.")
        print(f"    Next: Fix Visual Hull mesh creation (Option D or E)")
    elif iou_primitives > baseline_iou - 0.02:
        print(f"\n  ≈ SIMILAR PERFORMANCE: Multi-profile comparable to single-profile")
        print(f"    Phase 2 may not provide significant benefit.")
        print(f"    Consider alternative approaches.")
    else:
        print(f"\n  ✗ REGRESSION: Multi-profile performs worse than single-profile")
        print(f"    Phase 2 approach has fundamental issues.")
        print(f"    Need to reconsider strategy.")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_groundtruth_multiprofile()
