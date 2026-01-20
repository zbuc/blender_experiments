"""
Experiment C: Profile Interpolation Quality Test

Tests different interpolation methods for profile extraction and measures
their impact on 3D reconstruction accuracy (IoU).

Methods tested:
- Baseline: Linear interpolation (current)
- C1: Cubic spline interpolation
- C2: Median filtering + linear
- C3: Gaussian smoothing + linear
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add ~/blender_python_packages for user-installed dependencies
sys.path.insert(0, str(Path.home() / 'blender_python_packages'))

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    print("ERROR: This experiment must be run inside Blender")
    sys.exit(1)


# Import E2EValidator and custom validator class
from blender_blocking.test_e2e_validation import E2EValidator
from blender_blocking.main_integration import BlockingWorkflow


class CustomE2EValidator(E2EValidator):
    """Custom validator that accepts interpolation_method parameter."""

    def __init__(self, iou_threshold=0.7, interpolation_method='linear'):
        super().__init__(iou_threshold)
        self.interpolation_method = interpolation_method

    def validate_reconstruction(self, reference_paths, num_slices=12):
        """Run validation with custom interpolation method."""
        print("="*60)
        print("E2E VALIDATION TEST")
        print("="*60)

        # Step 1: Generate 3D model with specified interpolation method
        print("\n[1/4] Generating 3D model from reference images...")
        print(f"  Using interpolation method: {self.interpolation_method}")
        workflow = BlockingWorkflow(
            front_path=reference_paths.get('front'),
            side_path=reference_paths.get('side'),
            top_path=reference_paths.get('top'),
            interpolation_method=self.interpolation_method
        )
        mesh = workflow.run_full_workflow(num_slices=num_slices)

        if not mesh:
            print("ERROR: Failed to generate 3D model")
            return False, {}

        print(f"✓ Generated mesh: {mesh.name}")

        # Step 2-4: Continue with normal validation process
        from pathlib import Path
        print("\n[2/4] Setting up render configuration...")
        self.setup_render_settings()
        print("✓ Render settings configured")

        print("\n[3/4] Rendering orthogonal views...")
        from blender_blocking.integration.blender_ops.render_utils import render_orthogonal_views
        output_dir = Path(__file__).parent / 'test_output' / 'e2e_renders'
        rendered_paths = render_orthogonal_views(str(output_dir))

        if not rendered_paths:
            print("ERROR: Failed to render views")
            return False, {}

        for view, path in rendered_paths.items():
            print(f"✓ Rendered {view}: {path}")

        # Step 4: Compare with references
        print("\n[4/4] Comparing rendered views to reference images...")
        self.results = {}
        ious = []

        for view in ['front', 'side', 'top']:
            if view not in reference_paths or view not in rendered_paths:
                print(f"⚠ Skipping {view} (not available)")
                continue

            # Extract silhouettes
            ref_silhouette = self.extract_silhouette(reference_paths[view])
            render_silhouette = self.extract_silhouette(rendered_paths[view])

            # Debug: Save silhouettes for inspection
            from PIL import Image
            debug_dir = Path(__file__).parent / 'test_output' / 'debug_silhouettes'
            debug_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(ref_silhouette).save(debug_dir / f"{view}_ref_silhouette.png")
            Image.fromarray(render_silhouette).save(debug_dir / f"{view}_render_silhouette.png")

            # Compare
            from blender_blocking.integration.shape_matching.shape_matcher import compare_silhouettes
            iou, details = compare_silhouettes(ref_silhouette, render_silhouette)

            self.results[view] = {
                'iou': iou,
                'intersection': details['intersection'],
                'union': details['union'],
                'pixel_difference': details['pixel_difference']
            }

            ious.append(iou)
            print(f"  {view:8s} IoU: {iou:.3f}", end="")
            print(f"  {'✓ PASS' if iou >= self.iou_threshold else '✗ FAIL'}")

        # Calculate overall result
        if ious:
            avg_iou = sum(ious) / len(ious)
            passed = avg_iou >= self.iou_threshold

            print("\n" + "="*60)
            print(f"Average IoU: {avg_iou:.3f}")
            print(f"Threshold:   {self.iou_threshold:.3f}")
            print(f"Result:      {'✓ PASSED' if passed else '✗ FAILED'}")
            print("="*60)

            return passed, self.results
        else:
            print("ERROR: No views to compare")
            return False, {}


def run_experiment():
    """Run the interpolation quality experiment."""
    print("="*70)
    print("EXPERIMENT C: PROFILE INTERPOLATION QUALITY")
    print("="*70)
    print()

    # Setup test images
    base_dir = Path(__file__).parent
    test_images_dir = base_dir.parent / 'test_images'

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
            return

    reference_paths = {
        'front': str(test_images_dir / 'vase_front.png'),
        'side': str(test_images_dir / 'vase_side.png'),
        'top': str(test_images_dir / 'vase_top.png')
    }

    # Test configurations
    methods = [
        ('linear', 'Baseline: Linear interpolation'),
        ('cubic', 'C1: Cubic spline interpolation'),
        ('median_linear', 'C2: Median filtering + linear'),
        ('gaussian_linear', 'C3: Gaussian smoothing + linear')
    ]

    results = []
    baseline_iou = None

    # Run test for each method
    for method_name, method_description in methods:
        print()
        print("="*70)
        print(f"Testing: {method_description}")
        print("="*70)
        print()

        # Run E2E validation with specified interpolation method
        validator = CustomE2EValidator(
            iou_threshold=0.7,
            interpolation_method=method_name
        )
        passed, metrics = validator.validate_reconstruction(
            reference_paths,
            num_slices=120
        )

        # Calculate average IoU
        if metrics:
            ious = [m['iou'] for m in metrics.values()]
            avg_iou = sum(ious) / len(ious)

            # Store baseline for comparison
            if method_name == 'linear':
                baseline_iou = avg_iou

            # Calculate delta from baseline
            delta = (avg_iou - baseline_iou) if baseline_iou else 0.0

            result = {
                'method': method_name,
                'description': method_description,
                'avg_iou': float(avg_iou),
                'delta_from_baseline': float(delta),
                'views': {
                    view: {
                        'iou': float(m['iou']),
                        'intersection': int(m['intersection']),
                        'union': int(m['union'])
                    }
                    for view, m in metrics.items()
                },
                'passed': bool(passed),
                'timestamp': datetime.now().isoformat()
            }

            results.append(result)

            print()
            print(f"Result: Avg IoU = {avg_iou:.4f}", end="")
            if baseline_iou:
                print(f" (Δ {delta:+.4f})")
            else:
                print()
        else:
            print("ERROR: No metrics returned")

    # Print summary
    print()
    print("="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print()
    print(f"{'Method':<25} {'Avg IoU':>10} {'Delta':>10} {'Status':>10}")
    print("-"*70)

    for result in results:
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"{result['description']:<25} "
              f"{result['avg_iou']:>10.4f} "
              f"{result['delta_from_baseline']:>+10.4f} "
              f"{status:>10}")

    print("-"*70)

    # Find best method
    best_result = max(results, key=lambda r: r['avg_iou'])
    print()
    print(f"Best method: {best_result['description']}")
    print(f"Best IoU: {best_result['avg_iou']:.4f}")
    print(f"Improvement: {best_result['delta_from_baseline']:+.4f}")

    # Log results to experiment_log.jsonl
    log_file = base_dir / 'experiment_log.jsonl'
    print()
    print(f"Logging results to {log_file}")

    with open(log_file, 'a') as f:
        log_entry = {
            'experiment': 'C',
            'name': 'profile_interpolation_quality',
            'timestamp': datetime.now().isoformat(),
            'baseline_iou': baseline_iou,
            'results': results
        }
        f.write(json.dumps(log_entry) + '\n')

    print("Experiment complete!")
    print("="*70)


if __name__ == "__main__":
    if not BLENDER_AVAILABLE:
        print("ERROR: This script must be run inside Blender")
        print("Usage: blender --background --python experiment_profile_interpolation.py")
        sys.exit(1)

    run_experiment()
