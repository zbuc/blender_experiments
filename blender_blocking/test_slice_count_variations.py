"""
EXP-B: Test Slice Count Variations (num_slices)

Tests different slice counts (80, 120, 160) for reconstruction
to evaluate quality vs performance tradeoffs and reduce stepping artifacts.

The num_slices parameter controls how many horizontal slices are used
to analyze and reconstruct the 3D model from reference images.

Usage:
    blender --background --python test_slice_count_variations.py
"""

import sys
from pathlib import Path
import time

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add ~/blender_python_packages for user-installed dependencies
sys.path.insert(0, str(Path.home() / 'blender_python_packages'))

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    print("ERROR: This test must be run inside Blender")
    sys.exit(1)

from blender_blocking.test_e2e_validation import E2EValidator


def clear_scene():
    """Remove all mesh objects from the scene."""
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()


def test_slice_count(slice_count, reference_paths):
    """
    Test reconstruction with specific slice count.

    Args:
        slice_count: Number of horizontal slices for reconstruction
        reference_paths: Dict with reference image paths

    Returns:
        dict: Test results including IoU scores and timing
    """
    print("\n" + "="*70)
    print(f"TESTING WITH {slice_count} SLICES")
    print("="*70)

    # Clear scene before test
    clear_scene()

    # Run validation
    validator = E2EValidator(iou_threshold=0.7)

    start_time = time.time()
    passed, results = validator.validate_reconstruction(reference_paths, num_slices=slice_count)
    elapsed_time = time.time() - start_time

    # Calculate metrics
    ious = [r['iou'] for r in results.values()]
    avg_iou = sum(ious) / len(ious) if ious else 0.0

    # Get mesh statistics
    mesh_obj = bpy.data.objects.get("Blockout_Mesh") or bpy.data.objects.get("Sculpt_Base")
    vertex_total = len(mesh_obj.data.vertices) if mesh_obj else 0
    face_total = len(mesh_obj.data.polygons) if mesh_obj else 0

    test_result = {
        'slice_count': slice_count,
        'passed': passed,
        'avg_iou': avg_iou,
        'view_results': results,
        'elapsed_time': elapsed_time,
        'mesh_vertices': vertex_total,
        'mesh_faces': face_total
    }

    print(f"\n✓ Test completed in {elapsed_time:.2f}s")
    print(f"  Average IoU: {avg_iou:.3f}")
    print(f"  Mesh stats: {vertex_total:,} vertices, {face_total:,} faces")

    # Clear scene after test
    clear_scene()

    return test_result


def run_all_tests():
    """Run tests for all slice count variations."""
    base_dir = Path(__file__).parent
    test_images_dir = base_dir / 'test_images'

    # Check if test images exist
    if not test_images_dir.exists():
        print("Creating test images...")
        import subprocess
        result = subprocess.run(
            [sys.executable, str(base_dir / 'create_test_images.py')],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("ERROR: Failed to create test images")
            print(result.stderr)
            return False

    # Use vase test images (curved profile tests slice quality)
    reference_paths = {
        'front': str(test_images_dir / 'vase_front.png'),
        'side': str(test_images_dir / 'vase_side.png'),
        'top': str(test_images_dir / 'vase_top.png')
    }

    # Test configurations
    slice_counts = [80, 120, 160]

    print("\n" + "="*70)
    print("EXP-B: SLICE COUNT VARIATION TESTS")
    print("="*70)
    print(f"Testing slice counts (num_slices): {slice_counts}")
    print("Note: More slices = more vertical detail, less stepping artifacts")
    print()

    # Run tests
    results = []
    for slice_count in slice_counts:
        result = test_slice_count(slice_count, reference_paths)
        results.append(result)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: SLICE COUNT COMPARISON")
    print("="*70)
    print(f"{'Slices':<12} {'Avg IoU':<10} {'Time (s)':<10} {'Mesh Verts':<15} {'Mesh Faces':<12} {'Result'}")
    print("-"*70)

    for r in results:
        status = "✓ PASS" if r['passed'] else "✗ FAIL"
        print(f"{r['slice_count']:<12} "
              f"{r['avg_iou']:<10.3f} "
              f"{r['elapsed_time']:<10.2f} "
              f"{r['mesh_vertices']:<15,} "
              f"{r['mesh_faces']:<12,} "
              f"{status}")

    print("-"*70)

    # Detailed per-view analysis
    print("\nPER-VIEW IoU SCORES:")
    print("-"*70)
    print(f"{'Slices':<12} {'Front':<10} {'Side':<10} {'Top':<10}")
    print("-"*70)
    for r in results:
        front_iou = r['view_results'].get('front', {}).get('iou', 0.0)
        side_iou = r['view_results'].get('side', {}).get('iou', 0.0)
        top_iou = r['view_results'].get('top', {}).get('iou', 0.0)
        print(f"{r['slice_count']:<12} {front_iou:<10.3f} {side_iou:<10.3f} {top_iou:<10.3f}")
    print("-"*70)

    # Analysis
    print("\nANALYSIS:")
    print("-"*70)

    # Find best IoU
    best_iou = max(results, key=lambda x: x['avg_iou'])
    print(f"Best quality (IoU): {best_iou['slice_count']} slices ({best_iou['avg_iou']:.3f})")

    # Find fastest
    fastest = min(results, key=lambda x: x['elapsed_time'])
    print(f"Fastest: {fastest['slice_count']} slices ({fastest['elapsed_time']:.2f}s)")

    # Calculate quality-to-time ratio
    for r in results:
        r['quality_per_second'] = r['avg_iou'] / r['elapsed_time']

    best_ratio = max(results, key=lambda x: x['quality_per_second'])
    print(f"Best quality/time ratio: {best_ratio['slice_count']} slices ({best_ratio['quality_per_second']:.4f})")

    # IoU improvements
    print("\nQUALITY IMPROVEMENTS:")
    print("-"*70)
    baseline = results[0]  # 80 slices
    for r in results[1:]:
        iou_gain = r['avg_iou'] - baseline['avg_iou']
        time_cost = r['elapsed_time'] - baseline['elapsed_time']
        print(f"{r['slice_count']} vs {baseline['slice_count']} slices:")
        print(f"  IoU gain: {iou_gain:+.4f} ({iou_gain/baseline['avg_iou']*100:+.2f}%)")
        print(f"  Time cost: {time_cost:+.2f}s ({time_cost/baseline['elapsed_time']*100:+.2f}%)")

    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("-"*70)

    iou_diff = max(r['avg_iou'] for r in results) - min(r['avg_iou'] for r in results)
    time_diff = max(r['elapsed_time'] for r in results) - min(r['elapsed_time'] for r in results)

    print(f"IoU variation: {iou_diff:.4f}")
    print(f"Time variation: {time_diff:.2f}s")

    if iou_diff < 0.005:
        print("\n→ Minimal quality difference - recommend LOWEST slice count (80) for performance")
    elif iou_diff > 0.05:
        print(f"\n→ Significant quality improvement with more slices")
        print(f"→ Recommend {best_iou['slice_count']} slices for production quality")
        if best_ratio['slice_count'] != best_iou['slice_count']:
            print(f"→ For balanced performance: use {best_ratio['slice_count']} slices")
    else:
        print(f"\n→ Moderate quality improvement")
        print(f"→ Recommend {best_ratio['slice_count']} slices (best quality/time balance)")

    # Check if current setting (120) is optimal
    current_setting = next((r for r in results if r['slice_count'] == 120), None)
    if current_setting:
        print(f"\nCURRENT SETTING (120 slices):")
        print(f"  IoU: {current_setting['avg_iou']:.3f}")
        print(f"  Time: {current_setting['elapsed_time']:.2f}s")
        if current_setting['slice_count'] == best_iou['slice_count']:
            print("  Status: ✓ OPTIMAL for quality")
        elif current_setting['slice_count'] == best_ratio['slice_count']:
            print("  Status: ✓ OPTIMAL for quality/time balance")
        else:
            print(f"  Status: Consider switching to {best_iou['slice_count']} slices")

    print("="*70)

    # Write detailed results to file
    output_dir = base_dir / 'test_output' / 'slice_count_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / 'exp_b_results.txt'
    with open(results_file, 'w') as f:
        f.write("EXP-B: Slice Count Variation Results\n")
        f.write("="*70 + "\n\n")

        f.write(f"Test Configuration:\n")
        f.write(f"  Slice counts tested: {slice_counts}\n")
        f.write(f"  Test object: Vase (curved profile)\n")
        f.write(f"  Purpose: Evaluate vertical resolution and stepping artifacts\n\n")

        f.write("Background:\n")
        f.write("  Previous commit (c61843e) increased num_slices from 80 to 120\n")
        f.write("  to reduce stepping artifacts in vertical profiles.\n")
        f.write("  This test validates that decision and explores further improvements.\n\n")

        f.write("Results:\n")
        f.write("-"*70 + "\n")
        for r in results:
            f.write(f"\n{r['slice_count']} Slices:\n")
            f.write(f"  Average IoU: {r['avg_iou']:.4f}\n")
            f.write(f"  Processing time: {r['elapsed_time']:.2f}s\n")
            f.write(f"  Final mesh: {r['mesh_vertices']:,} vertices, {r['mesh_faces']:,} faces\n")
            f.write(f"  Per-view IoU scores:\n")
            for view, metrics in r['view_results'].items():
                f.write(f"    {view}: {metrics['iou']:.4f}\n")

        f.write("\n" + "="*70 + "\n")
        f.write(f"Best quality: {best_iou['slice_count']} slices ({best_iou['avg_iou']:.4f} IoU)\n")
        f.write(f"Best efficiency: {best_ratio['slice_count']} slices\n")
        f.write(f"\nRecommendation: See analysis above\n")

    print(f"\nDetailed results saved to: {results_file}")

    return all(r['passed'] for r in results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
