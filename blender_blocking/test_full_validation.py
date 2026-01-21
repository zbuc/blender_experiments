"""
Full Validation Suite - E2E + Multi-View Consistency (EXP-J)

Runs complete validation of 3D reconstruction:
1. E2E Validation: Compares rendered views to reference images (IoU)
2. Multi-View Consistency: Checks geometric consistency between views

This provides comprehensive quality assurance:
- E2E ensures the model matches the input
- Consistency ensures the model is geometrically sound

Usage:
    # Headless (for CI/CD)
    blender --background --python test_full_validation.py

    # In Blender GUI
    Run this script in Blender's scripting workspace
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path.home() / 'blender_python_packages'))

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    print("ERROR: This test must be run inside Blender")
    sys.exit(1)

from blender_blocking.main_integration import BlockingWorkflow
from blender_blocking.integration.blender_ops.render_utils import render_orthogonal_views
from blender_blocking.test_e2e_validation import E2EValidator
from blender_blocking.multi_view_consistency import MultiViewConsistencyValidator


class FullValidator:
    """
    Combined E2E and multi-view consistency validator.

    Provides comprehensive validation:
    - E2E: Rendered views match references (accuracy)
    - Consistency: Views are geometrically consistent (soundness)
    """

    def __init__(self, iou_threshold=0.7, consistency_tolerance=0.05):
        """
        Initialize full validator.

        Args:
            iou_threshold: Minimum IoU for E2E validation (0-1)
            consistency_tolerance: Fractional tolerance for consistency (0.05 = 5%)
        """
        self.e2e_validator = E2EValidator(iou_threshold=iou_threshold)
        self.consistency_validator = MultiViewConsistencyValidator(tolerance=consistency_tolerance)
        self.results = {}

    def validate_full(self, reference_paths, num_slices=12):
        """
        Run full validation suite.

        Args:
            reference_paths: Dict with 'front', 'side', 'top' reference image paths
            num_slices: Number of slices for reconstruction

        Returns:
            Tuple of (passed: bool, results: dict)
        """
        print("="*70)
        print(" "*15 + "FULL VALIDATION SUITE")
        print(" "*10 + "E2E + Multi-View Consistency (EXP-J)")
        print("="*70)

        # Step 1: Generate 3D model
        print("\n[STEP 1/4] Generating 3D model from reference images...")
        print("-"*70)
        workflow = BlockingWorkflow(
            front_path=reference_paths.get('front'),
            side_path=reference_paths.get('side'),
            top_path=reference_paths.get('top')
        )
        mesh = workflow.run_full_workflow(num_slices=num_slices)

        if not mesh:
            print("ERROR: Failed to generate 3D model")
            return False, {}

        print(f"✓ Generated mesh: {mesh.name}")

        # Step 2: Setup rendering
        print("\n[STEP 2/4] Setting up render configuration...")
        print("-"*70)
        self.e2e_validator.setup_render_settings()
        print("✓ Render settings configured")

        # Step 3: Render orthogonal views
        print("\n[STEP 3/4] Rendering orthogonal views...")
        print("-"*70)
        output_dir = Path(__file__).parent / 'test_output' / 'full_validation_renders'
        rendered_paths = render_orthogonal_views(str(output_dir))

        if not rendered_paths:
            print("ERROR: Failed to render views")
            return False, {}

        for view, path in rendered_paths.items():
            print(f"✓ Rendered {view}: {path}")

        # Step 4: Run validations
        print("\n[STEP 4/4] Running validation tests...")
        print("="*70)

        # 4a: E2E Validation
        print("\n>>> VALIDATION 1: E2E (Rendered vs Reference)")
        print("-"*70)
        e2e_passed, e2e_results = self._run_e2e_validation(
            reference_paths,
            rendered_paths
        )
        self.results['e2e'] = {
            'passed': e2e_passed,
            'results': e2e_results
        }

        # 4b: Multi-View Consistency
        print("\n>>> VALIDATION 2: Multi-View Consistency")
        print("-"*70)
        consistency_passed, consistency_results = self.consistency_validator.validate_consistency(
            rendered_paths
        )
        self.results['consistency'] = {
            'passed': consistency_passed,
            'results': consistency_results
        }

        # Overall result
        all_passed = e2e_passed and consistency_passed

        # Final summary
        print("\n" + "="*70)
        print(" "*25 + "FINAL RESULTS")
        print("="*70)
        print(f"E2E Validation (IoU):         {'✓ PASSED' if e2e_passed else '✗ FAILED'}")
        print(f"Multi-View Consistency:       {'✓ PASSED' if consistency_passed else '✗ FAILED'}")
        print("-"*70)
        print(f"Overall Result:               {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
        print("="*70)

        return all_passed, self.results

    def _run_e2e_validation(self, reference_paths, rendered_paths):
        """
        Run E2E validation (internal helper).

        Args:
            reference_paths: Reference image paths
            rendered_paths: Rendered image paths

        Returns:
            Tuple of (passed, results)
        """
        results = {}
        ious = []

        for view in ['front', 'side', 'top']:
            if view not in reference_paths or view not in rendered_paths:
                print(f"⚠ Skipping {view} (not available)")
                continue

            # Extract silhouettes
            ref_silhouette = self.e2e_validator.extract_silhouette(reference_paths[view])
            render_silhouette = self.e2e_validator.extract_silhouette(rendered_paths[view])

            # Save debug silhouettes
            from PIL import Image
            debug_dir = Path(__file__).parent / 'test_output' / 'full_validation_debug'
            debug_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(ref_silhouette).save(debug_dir / f"{view}_ref_silhouette.png")
            Image.fromarray(render_silhouette).save(debug_dir / f"{view}_render_silhouette.png")

            # Compare
            from blender_blocking.integration.shape_matching.shape_matcher import compare_silhouettes
            iou, details = compare_silhouettes(ref_silhouette, render_silhouette)

            results[view] = {
                'iou': iou,
                'intersection': details['intersection'],
                'union': details['union'],
                'pixel_difference': details['pixel_difference']
            }

            ious.append(iou)
            status = '✓ PASS' if iou >= self.e2e_validator.iou_threshold else '✗ FAIL'
            print(f"  {view:8s} IoU: {iou:.3f}  {status}")

        if ious:
            avg_iou = sum(ious) / len(ious)
            passed = avg_iou >= self.e2e_validator.iou_threshold
            print(f"\nAverage IoU: {avg_iou:.3f} (threshold: {self.e2e_validator.iou_threshold:.3f})")
            return passed, results
        else:
            return False, {}

    def print_detailed_results(self):
        """Print comprehensive results from all validators."""
        if not self.results:
            print("No results to display")
            return

        print("\n" + "="*70)
        print(" "*20 + "DETAILED RESULTS")
        print("="*70)

        # E2E Results
        if 'e2e' in self.results:
            print("\n### E2E VALIDATION (IoU Metrics) ###")
            print("-"*70)
            e2e_results = self.results['e2e']['results']
            print(f"{'View':<10} {'IoU':>8} {'Intersection':>12} {'Union':>10} {'PixDiff':>10}")
            print("-"*70)
            for view, metrics in e2e_results.items():
                print(f"{view:<10} "
                      f"{metrics['iou']:>8.3f} "
                      f"{metrics['intersection']:>12d} "
                      f"{metrics['union']:>10d} "
                      f"{metrics['pixel_difference']:>10.2f}")

        # Consistency Results
        if 'consistency' in self.results:
            print("\n### MULTI-VIEW CONSISTENCY ###")
            print("-"*70)
            self.consistency_validator.print_detailed_results()

        print("="*70)


def test_full_validation_with_sample_images():
    """
    Run full validation suite with built-in sample images.

    Returns:
        bool: All validations passed
    """
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

    # Use vase test images
    reference_paths = {
        'front': str(test_images_dir / 'vase_front.png'),
        'side': str(test_images_dir / 'vase_side.png'),
        'top': str(test_images_dir / 'vase_top.png')
    }

    # Run full validation
    validator = FullValidator(iou_threshold=0.7, consistency_tolerance=0.05)
    passed, results = validator.validate_full(reference_paths, num_slices=120)

    # Print detailed results
    validator.print_detailed_results()

    return passed


def test_full_validation_with_custom_images(front, side, top, num_slices=12):
    """
    Run full validation suite with custom images.

    Args:
        front: Path to front view image
        side: Path to side view image
        top: Path to top view image
        num_slices: Number of slices for reconstruction

    Returns:
        bool: All validations passed
    """
    reference_paths = {
        'front': front,
        'side': side,
        'top': top
    }

    validator = FullValidator(iou_threshold=0.7, consistency_tolerance=0.05)
    passed, results = validator.validate_full(reference_paths, num_slices=num_slices)

    validator.print_detailed_results()

    return passed


if __name__ == "__main__":
    # Run test with sample images
    success = test_full_validation_with_sample_images()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
