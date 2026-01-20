"""
Experimental test for improved profile interpolation methods.

This module tests different interpolation techniques to improve the quality
of profile extraction and vertex refinement, with the goal of improving IoU scores.
"""

import sys
from pathlib import Path

# Add Blender Python packages to path FIRST before any numpy/scipy imports
blender_python_packages = Path.home() / "blender_python_packages"
if blender_python_packages.exists():
    sys.path.insert(0, str(blender_python_packages))

# Now import dependencies
import numpy as np
from scipy.interpolate import interp1d, CubicSpline, PchipInterpolator
from scipy.ndimage import median_filter
from typing import List, Tuple, Callable
import cv2

# Import local modules
sys.path.insert(0, str(Path(__file__).parent))

from integration.shape_matching.profile_extractor import (
    extract_silhouette_from_image,
    validate_profile
)
from test_integration import create_test_images

# Check if running in Blender
try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    print("ERROR: This test must be run inside Blender")
    sys.exit(1)

from main_integration import BlockingWorkflow
from integration.blender_ops.render_utils import render_orthogonal_views
from integration.shape_matching.shape_matcher import compare_silhouettes


def interpolate_linear(valid_positions, valid_widths, invalid_positions):
    """Linear interpolation (current baseline method)."""
    interp_func = interp1d(
        valid_positions,
        valid_widths,
        kind='linear',
        fill_value='extrapolate'
    )
    return interp_func(invalid_positions)


def interpolate_cubic(valid_positions, valid_widths, invalid_positions):
    """Cubic spline interpolation (smooth curves)."""
    try:
        interp_func = interp1d(
            valid_positions,
            valid_widths,
            kind='cubic',
            fill_value='extrapolate'
        )
        return interp_func(invalid_positions)
    except ValueError:
        # Fall back to linear if not enough points for cubic
        return interpolate_linear(valid_positions, valid_widths, invalid_positions)


def interpolate_quadratic(valid_positions, valid_widths, invalid_positions):
    """Quadratic interpolation (moderate smoothness)."""
    try:
        interp_func = interp1d(
            valid_positions,
            valid_widths,
            kind='quadratic',
            fill_value='extrapolate'
        )
        return interp_func(invalid_positions)
    except ValueError:
        # Fall back to linear if not enough points
        return interpolate_linear(valid_positions, valid_widths, invalid_positions)


def interpolate_pchip(valid_positions, valid_widths, invalid_positions):
    """
    PCHIP (Piecewise Cubic Hermite Interpolating Polynomial).
    Preserves monotonicity and avoids overshoot.
    """
    try:
        interp_func = PchipInterpolator(valid_positions, valid_widths, extrapolate=True)
        return interp_func(invalid_positions)
    except ValueError:
        return interpolate_linear(valid_positions, valid_widths, invalid_positions)


def interpolate_cubic_spline(valid_positions, valid_widths, invalid_positions):
    """
    CubicSpline with natural boundary conditions.
    More control over boundary behavior than scipy's interp1d.
    """
    try:
        # Use 'natural' boundary conditions (second derivative = 0 at boundaries)
        cs = CubicSpline(valid_positions, valid_widths, bc_type='natural', extrapolate=True)
        return cs(invalid_positions)
    except ValueError:
        return interpolate_linear(valid_positions, valid_widths, invalid_positions)


def interpolate_cubic_spline_clamped(valid_positions, valid_widths, invalid_positions):
    """
    CubicSpline with clamped boundary conditions.
    First derivatives at boundaries are set to zero (flat ends).
    """
    try:
        # Use 'clamped' boundary conditions with zero derivatives
        cs = CubicSpline(valid_positions, valid_widths, bc_type='clamped', extrapolate=True)
        return cs(invalid_positions)
    except ValueError:
        return interpolate_linear(valid_positions, valid_widths, invalid_positions)


def extract_vertical_profile_with_method(
    image: np.ndarray,
    interpolation_method: Callable,
    num_samples: int = 100,
    median_filter_size: int = 3
) -> List[Tuple[float, float]]:
    """
    Extract vertical profile using a specified interpolation method.

    This is a modified version of profile_extractor.extract_vertical_profile()
    that allows testing different interpolation methods.

    Args:
        image: Input image
        interpolation_method: Function to use for interpolation
        num_samples: Number of vertical samples
        median_filter_size: Size of median filter (0 to disable)

    Returns:
        List of (height, radius) tuples
    """
    if image is None or image.size == 0:
        raise ValueError("Image is empty or None")

    # Convert to filled silhouette
    silhouette = extract_silhouette_from_image(image)
    height, width = silhouette.shape

    if height < 2 or width < 2:
        raise ValueError(f"Image too small: {silhouette.shape}")

    # Initialize profile storage
    widths = []

    # Sample at regular vertical intervals
    for i in range(num_samples):
        y = int(height - 1 - (i / (num_samples - 1)) * (height - 1))
        row = silhouette[y, :]

        # Find leftmost and rightmost filled pixels
        filled_positions = np.where(row > 127)[0]

        if len(filled_positions) >= 2:
            left_edge = filled_positions[0]
            right_edge = filled_positions[-1]
            measured_width = right_edge - left_edge
        elif len(filled_positions) == 1:
            measured_width = 1
        else:
            measured_width = np.nan

        widths.append(measured_width)

    # Convert to numpy array
    widths = np.array(widths)

    # Handle missing data with specified interpolation method
    if np.isnan(widths).any():
        valid_indices = ~np.isnan(widths)
        if valid_indices.sum() >= 2:
            valid_positions = np.where(valid_indices)[0]
            valid_widths = widths[valid_indices]
            invalid_positions = np.where(~valid_indices)[0]

            # Use the specified interpolation method
            widths[invalid_positions] = interpolation_method(
                valid_positions, valid_widths, invalid_positions
            )
        else:
            widths = np.full(num_samples, width * 0.8)

    # Clamp widths to non-negative
    widths = np.maximum(widths, 0)

    # Apply median filter if specified
    if median_filter_size > 0:
        widths = median_filter(widths, size=median_filter_size)

    # Normalize widths to 0-1 range
    max_width = np.max(widths)
    if max_width > 0:
        normalized_widths = widths / max_width
    else:
        normalized_widths = np.ones(num_samples) * 0.8

    # Create (height, radius) tuples
    profile = []
    for i in range(num_samples):
        height_normalized = i / (num_samples - 1)
        radius_normalized = normalized_widths[i]
        profile.append((height_normalized, radius_normalized))

    return profile


def test_interpolation_method(
    method_name: str,
    interpolation_func: Callable,
    reference_paths: dict,
    num_slices: int = 120
) -> dict:
    """
    Test a specific interpolation method on the full pipeline.

    Args:
        method_name: Name of the method for reporting
        interpolation_func: Interpolation function to test
        reference_paths: Dict with 'front', 'side', 'top' image paths
        num_slices: Number of slices for reconstruction

    Returns:
        Dictionary with test results including IoU scores
    """
    print(f"\n{'='*60}")
    print(f"Testing: {method_name}")
    print(f"{'='*60}")

    try:
        # Patch the profile extraction to use our method
        import integration.shape_matching.profile_extractor as pe

        # Save original function
        original_extract = pe.extract_vertical_profile

        # Create patched version
        def patched_extract(image, num_samples=100):
            return extract_vertical_profile_with_method(
                image, interpolation_func, num_samples
            )

        # Apply patch
        pe.extract_vertical_profile = patched_extract

        # Clear scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # Run the workflow
        print(f"  [1/3] Generating 3D model...")
        workflow = BlockingWorkflow(
            front_path=reference_paths.get('front'),
            side_path=reference_paths.get('side'),
            top_path=reference_paths.get('top')
        )
        mesh = workflow.run_full_workflow(num_slices=num_slices)

        if not mesh:
            # Restore original function
            pe.extract_vertical_profile = original_extract
            return {
                'method': method_name,
                'error': 'Failed to generate mesh',
                'iou': None
            }

        print(f"  ✓ Generated mesh: {mesh.name}")

        # Setup rendering for clean silhouettes
        print(f"  [2/3] Rendering views...")
        scene = bpy.context.scene
        scene.render.film_transparent = True
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.resolution_x = 512
        scene.render.resolution_y = 512
        scene.render.engine = 'BLENDER_EEVEE'
        scene.eevee.taa_render_samples = 1

        # Render orthogonal views
        output_dir = Path(__file__).parent / 'test_output' / 'interpolation_test' / method_name.replace(' ', '_')
        output_dir.mkdir(parents=True, exist_ok=True)
        rendered_paths = render_orthogonal_views(str(output_dir))

        if not rendered_paths:
            # Restore original function
            pe.extract_vertical_profile = original_extract
            return {
                'method': method_name,
                'error': 'Failed to render views',
                'iou': None
            }

        # Compare with references
        print(f"  [3/3] Comparing with references...")
        ious = []

        for view in ['front', 'side', 'top']:
            if view not in reference_paths or view not in rendered_paths:
                continue

            # Extract silhouettes using the same method as E2E test
            ref_img = cv2.imread(reference_paths[view], cv2.IMREAD_UNCHANGED)
            render_img = cv2.imread(rendered_paths[view], cv2.IMREAD_UNCHANGED)

            # Extract reference silhouette
            if len(ref_img.shape) == 3 and ref_img.shape[2] == 4:
                ref_silhouette = (ref_img[:, :, 3] > 128).astype(np.uint8) * 255
            else:
                gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if len(ref_img.shape) == 3 else ref_img
                ref_silhouette = (gray < 128).astype(np.uint8) * 255

            # Extract render silhouette
            if len(render_img.shape) == 3 and render_img.shape[2] == 4:
                render_silhouette = (render_img[:, :, 3] > 128).astype(np.uint8) * 255
            else:
                gray = cv2.cvtColor(render_img, cv2.COLOR_BGR2GRAY) if len(render_img.shape) == 3 else render_img
                render_silhouette = (gray < 128).astype(np.uint8) * 255

            # Compare
            iou, details = compare_silhouettes(ref_silhouette, render_silhouette)
            ious.append(iou)
            print(f"    {view}: IoU = {iou:.4f}")

        # Restore original function
        pe.extract_vertical_profile = original_extract

        # Calculate average IoU
        avg_iou = sum(ious) / len(ious) if ious else 0.0

        return {
            'method': method_name,
            'iou': avg_iou,
            'ious_by_view': dict(zip(['front', 'side', 'top'][:len(ious)], ious)),
            'error': None,
            'mesh_name': mesh.name
        }

    except Exception as e:
        print(f"  ✗ Error testing {method_name}: {e}")
        import traceback
        traceback.print_exc()

        # Restore original function if we patched it
        try:
            import integration.shape_matching.profile_extractor as pe
            if 'original_extract' in locals():
                pe.extract_vertical_profile = original_extract
        except:
            pass

        return {
            'method': method_name,
            'error': str(e),
            'iou': None
        }


def run_comparison_tests(shape: str = 'vase', num_slices: int = 120):
    """
    Run comparison tests across all interpolation methods.

    Args:
        shape: Shape to test ('bottle' or 'vase')
        num_slices: Number of slices for reconstruction
    """
    print("=" * 80)
    print("PROFILE INTERPOLATION METHOD COMPARISON")
    print("=" * 80)
    print(f"Shape: {shape}")
    print(f"Slices: {num_slices}")

    # Create test images
    print("\nCreating test images...")
    test_images_dir = Path(__file__).parent / 'test_images'
    views = create_test_images(output_dir=str(test_images_dir), shape=shape)

    if not views:
        print("ERROR: Failed to create test images")
        return []

    reference_paths = {
        'front': views['front'],
        'side': views['side'],
        'top': views['top']
    }

    # Define methods to test
    methods = [
        ("Linear (Baseline)", interpolate_linear),
        ("Cubic Spline", interpolate_cubic),
        ("Quadratic", interpolate_quadratic),
        ("PCHIP (Monotonic)", interpolate_pchip),
        ("CubicSpline Natural BC", interpolate_cubic_spline),
        ("CubicSpline Clamped BC", interpolate_cubic_spline_clamped),
    ]

    results = []

    # Test each method
    for method_name, method_func in methods:
        result = test_interpolation_method(
            method_name,
            method_func,
            reference_paths,
            num_slices
        )
        results.append(result)

        if result.get('iou') is not None:
            print(f"  ✓ {method_name}: IoU = {result['iou']:.4f}")
        else:
            print(f"  ✗ {method_name}: FAILED - {result.get('error', 'Unknown error')}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)
    print(f"{'Method':<30} {'Avg IoU':<15} {'Front':<10} {'Side':<10} {'Top':<10}")
    print("-" * 80)

    for result in results:
        method = result['method']
        iou = result.get('iou')
        ious_by_view = result.get('ious_by_view', {})

        if iou is not None:
            iou_str = f"{iou:.4f}"
            front_str = f"{ious_by_view.get('front', 0):.4f}" if 'front' in ious_by_view else "N/A"
            side_str = f"{ious_by_view.get('side', 0):.4f}" if 'side' in ious_by_view else "N/A"
            top_str = f"{ious_by_view.get('top', 0):.4f}" if 'top' in ious_by_view else "N/A"
        else:
            iou_str = "FAILED"
            front_str = side_str = top_str = "-"

        print(f"{method:<30} {iou_str:<15} {front_str:<10} {side_str:<10} {top_str:<10}")

    # Find best method
    valid_results = [r for r in results if r.get('iou') is not None]
    if valid_results:
        best = max(valid_results, key=lambda x: x['iou'])
        baseline = next((r for r in results if 'Baseline' in r['method']), None)

        print("\n" + "=" * 80)
        print(f"BEST METHOD: {best['method']} with IoU = {best['iou']:.4f}")
        if baseline and baseline.get('iou') is not None:
            improvement = best['iou'] - baseline['iou']
            pct_improvement = (improvement / baseline['iou']) * 100
            print(f"Improvement over baseline: {improvement:+.4f} ({pct_improvement:+.2f}%)")
        print("=" * 80)

    return results


if __name__ == "__main__":
    import sys

    # Parse arguments - when run via Blender, args after '--' are script args
    # Example: blender --background --python script.py -- vase 120
    try:
        dash_dash_index = sys.argv.index('--')
        script_args = sys.argv[dash_dash_index + 1:]
    except ValueError:
        script_args = []

    # Get shape from command line or use default
    shape = script_args[0] if len(script_args) > 0 else "vase"
    num_slices = int(script_args[1]) if len(script_args) > 1 else 120

    print(f"Testing interpolation methods with {shape} shape and {num_slices} slices")
    results = run_comparison_tests(shape, num_slices)
