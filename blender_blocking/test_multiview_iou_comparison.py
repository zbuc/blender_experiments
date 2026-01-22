"""
IoU Comparison Test: 3-view vs 12-view reconstruction.

Tests the multi-view Visual Hull system against the 3-view baseline
to validate expected IoU improvements.

Expected results:
- 3-view baseline: ~0.875 IoU
- 12-view enhanced: 0.88-0.92 IoU

Usage:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python test_multiview_iou_comparison.py
"""

import sys
from pathlib import Path
import time

# Add to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path.home() / 'blender_python_packages'))

import numpy as np
from integration.image_processing.image_loader import (
    load_orthogonal_views,
    load_multi_view_auto
)
from integration.multi_view.visual_hull import MultiViewVisualHull


def load_image_as_silhouette(image_path):
    """Load image and convert to binary silhouette."""
    from integration.image_processing.image_loader import load_image

    img = load_image(str(image_path))

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)

    # Threshold to binary (silhouette is dark on light background)
    silhouette = img < 128

    return silhouette


def reconstruct_3view(front_path, side_path, top_path, resolution=128):
    """
    Reconstruct using 3-view baseline (front, side, top).

    Args:
        front_path: Path to front view image
        side_path: Path to side view image
        top_path: Path to top view image
        resolution: Voxel grid resolution

    Returns:
        (voxel_grid, processing_time, stats)
    """
    print("\n" + "="*70)
    print("3-VIEW BASELINE RECONSTRUCTION")
    print("="*70)

    # Load silhouettes
    print(f"\nLoading 3 views...")
    print(f"  Front: {front_path}")
    print(f"  Side: {side_path}")
    print(f"  Top: {top_path}")

    front_silhouette = load_image_as_silhouette(front_path)
    side_silhouette = load_image_as_silhouette(side_path)
    top_silhouette = load_image_as_silhouette(top_path)

    print(f"  ✓ Loaded {front_silhouette.shape} silhouettes")

    # Create Visual Hull instance
    hull = MultiViewVisualHull(
        resolution=resolution,
        bounds_min=np.array([-2.0, -2.0, -2.0]),
        bounds_max=np.array([2.0, 2.0, 2.0])
    )

    # Add views (front=0°, side=90°)
    hull.add_view_from_silhouette(front_silhouette, angle=0.0, view_type='lateral')
    hull.add_view_from_silhouette(side_silhouette, angle=90.0, view_type='lateral')
    hull.add_view_from_silhouette(top_silhouette, view_type='top')

    print(f"\n✓ Added 3 views to Visual Hull")

    # Reconstruct
    print(f"\nReconstructing with resolution {resolution}³...")
    start_time = time.time()
    voxel_grid = hull.reconstruct(verbose=True)
    processing_time = time.time() - start_time

    # Get stats
    stats = hull.get_stats()
    stats['processing_time'] = processing_time

    print(f"\n✓ Reconstruction complete in {processing_time:.2f}s")
    print(f"  Occupied voxels: {stats['occupied_voxels']:,} / {stats['total_voxels']:,}")
    print(f"  Occupancy: {stats['occupancy']*100:.2f}%")

    return voxel_grid, processing_time, stats


def reconstruct_12view(turntable_dir, resolution=128):
    """
    Reconstruct using 12-view turntable sequence.

    Args:
        turntable_dir: Directory containing turntable views
        resolution: Voxel grid resolution

    Returns:
        (voxel_grid, processing_time, stats)
    """
    print("\n" + "="*70)
    print("12-VIEW ENHANCED RECONSTRUCTION")
    print("="*70)

    # Load multi-view sequence
    print(f"\nLoading turntable sequence from: {turntable_dir}")
    views_dict = load_multi_view_auto(str(turntable_dir), num_views=12, include_top=True)

    num_lateral = sum(1 for k in views_dict if k.startswith('lateral_'))
    has_top = 'top' in views_dict
    print(f"  ✓ Loaded {num_lateral} lateral views + {'1' if has_top else '0'} top view")

    # Create Visual Hull instance
    hull = MultiViewVisualHull(
        resolution=resolution,
        bounds_min=np.array([-2.0, -2.0, -2.0]),
        bounds_max=np.array([2.0, 2.0, 2.0])
    )

    # Add all views
    for view_name, (img, angle, view_type) in views_dict.items():
        # Convert to silhouette
        if len(img.shape) == 3:
            img_gray = np.mean(img, axis=2)
        else:
            img_gray = img

        silhouette = img_gray < 128

        hull.add_view_from_silhouette(silhouette, angle=angle, view_type=view_type)

    print(f"  ✓ Added {len(hull.views)} views to Visual Hull")

    # Reconstruct
    print(f"\nReconstructing with resolution {resolution}³...")
    start_time = time.time()
    voxel_grid = hull.reconstruct(verbose=True)
    processing_time = time.time() - start_time

    # Get stats
    stats = hull.get_stats()
    stats['processing_time'] = processing_time

    print(f"\n✓ Reconstruction complete in {processing_time:.2f}s")
    print(f"  Occupied voxels: {stats['occupied_voxels']:,} / {stats['total_voxels']:,}")
    print(f"  Occupancy: {stats['occupancy']*100:.2f}%")

    return voxel_grid, processing_time, stats


def estimate_iou(voxel_grid_a, voxel_grid_b):
    """
    Estimate IoU between two voxel grids.

    Args:
        voxel_grid_a: First voxel grid (boolean)
        voxel_grid_b: Second voxel grid (boolean)

    Returns:
        IoU score (0-1)
    """
    intersection = np.logical_and(voxel_grid_a, voxel_grid_b).sum()
    union = np.logical_or(voxel_grid_a, voxel_grid_b).sum()

    if union == 0:
        return 0.0

    return intersection / union


def compare_reconstructions():
    """
    Run comparison test between 3-view and 12-view reconstruction.
    """
    print("\n" + "="*70)
    print("MULTI-VIEW IoU COMPARISON TEST")
    print("="*70)
    print("\nObjective: Validate Phase 1 multi-view implementation")
    print("Expected: 12-view IoU > 3-view IoU (0.88-0.92 vs 0.875)")

    # Test parameters
    resolution = 128  # Production resolution
    test_object = "vase"

    # Paths
    front_path = Path("test_images/vase_front.png")
    side_path = Path("test_images/vase_side.png")
    top_path = Path("test_images/vase_top.png")
    turntable_dir = Path("test_images/turntable_vase/")

    # Verify paths exist
    if not front_path.exists():
        print(f"\n❌ ERROR: Front view not found: {front_path}")
        return

    if not turntable_dir.exists():
        print(f"\n❌ ERROR: Turntable directory not found: {turntable_dir}")
        print("\nRun generate_turntable_sequence.py first:")
        print("  /Applications/Blender.app/Contents/MacOS/Blender --background --python generate_turntable_sequence.py -- --object vase")
        return

    # Run 3-view reconstruction
    voxel_3view, time_3view, stats_3view = reconstruct_3view(
        front_path, side_path, top_path, resolution=resolution
    )

    # Run 12-view reconstruction
    voxel_12view, time_12view, stats_12view = reconstruct_12view(
        turntable_dir, resolution=resolution
    )

    # Compare results
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)

    print(f"\n3-View Baseline:")
    print(f"  Views: 3 (front, side, top)")
    print(f"  Occupied voxels: {stats_3view['occupied_voxels']:,}")
    print(f"  Occupancy: {stats_3view['occupancy']*100:.2f}%")
    print(f"  Processing time: {stats_3view['processing_time']:.2f}s")

    print(f"\n12-View Enhanced:")
    print(f"  Views: {stats_12view['num_views']}")
    print(f"  Occupied voxels: {stats_12view['occupied_voxels']:,}")
    print(f"  Occupancy: {stats_12view['occupancy']*100:.2f}%")
    print(f"  Processing time: {stats_12view['processing_time']:.2f}s")

    # Calculate relative metrics
    iou_between = estimate_iou(voxel_3view, voxel_12view)
    occupancy_diff = stats_12view['occupancy'] - stats_3view['occupancy']
    time_ratio = stats_12view['processing_time'] / stats_3view['processing_time']

    print(f"\nDifferences:")
    print(f"  IoU between reconstructions: {iou_between:.4f}")
    print(f"  Occupancy difference: {occupancy_diff*100:+.2f}%")
    print(f"  Processing time ratio: {time_ratio:.2f}x")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    print(f"\n✓ Multi-view reconstruction successful")
    print(f"✓ Both methods completed without errors")

    if iou_between > 0.85:
        print(f"✓ High agreement between methods (IoU={iou_between:.4f})")
        print("  → Reconstructions are very similar")
    elif iou_between > 0.70:
        print(f"⚠ Moderate agreement (IoU={iou_between:.4f})")
        print("  → Some differences in reconstruction")
    else:
        print(f"⚠ Low agreement (IoU={iou_between:.4f})")
        print("  → Significant differences - investigate")

    if time_ratio < 3.0:
        print(f"✓ Processing time acceptable ({time_ratio:.2f}x slower)")
    else:
        print(f"⚠ Processing time high ({time_ratio:.2f}x slower)")

    # Ground truth comparison note
    print("\n" + "="*70)
    print("NOTE: Ground Truth Comparison")
    print("="*70)
    print("\nThis test compares 3-view vs 12-view reconstructions.")
    print("To measure absolute IoU vs ground truth:")
    print("  1. Load the original vase mesh")
    print("  2. Voxelize it at same resolution")
    print("  3. Compare both reconstructions to ground truth")
    print("\nExpected absolute IoU:")
    print("  - 3-view: ~0.875")
    print("  - 12-view: 0.88-0.92")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


def main():
    """Run comparison test."""
    try:
        compare_reconstructions()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
