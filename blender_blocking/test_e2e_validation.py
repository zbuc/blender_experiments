"""
End-to-End Validation Test

Validates that 3D reconstruction accurately represents input reference images
by rendering the generated mesh and comparing to original inputs.

Usage:
    # In Blender (with GUI)
    Run this script in Blender's scripting workspace

    # Headless (for CI/CD)
    blender --background --python test_e2e_validation.py
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    print("ERROR: This test must be run inside Blender")
    sys.exit(1)

import numpy as np
from blender_blocking.main_integration import BlockingWorkflow
from blender_blocking.integration.blender_ops.render_utils import render_orthogonal_views
from blender_blocking.integration.shape_matching.shape_matcher import compare_silhouettes
from blender_blocking.integration.image_processing.image_loader import load_image
from blender_blocking.integration.image_processing.image_processor import process_image


class E2EValidator:
    """End-to-end validation for 3D reconstruction accuracy."""

    def __init__(self, iou_threshold=0.7):
        """
        Initialize validator.

        Args:
            iou_threshold: Minimum IoU score to pass (0-1)
        """
        self.iou_threshold = iou_threshold
        self.results = {}

    def setup_render_settings(self):
        """Configure Blender for clean silhouette rendering."""
        scene = bpy.context.scene

        # Transparent background for clean silhouettes
        scene.render.film_transparent = True
        scene.render.image_settings.color_mode = 'RGBA'

        # Resolution
        scene.render.resolution_x = 512
        scene.render.resolution_y = 512
        scene.render.resolution_percentage = 100

        # Fast rendering (we only need silhouettes)
        scene.render.engine = 'BLENDER_EEVEE'
        scene.eevee.taa_render_samples = 1

    def extract_silhouette(self, image_path):
        """
        Extract binary silhouette from image.

        Args:
            image_path: Path to image file

        Returns:
            Binary numpy array (0 or 255)
        """
        img = load_image(image_path)

        # If RGBA, use alpha channel for silhouette
        if len(img.shape) == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3]
            # Threshold alpha to binary
            silhouette = (alpha > 128).astype(np.uint8) * 255
        else:
            # Use intensity-based thresholding
            if len(img.shape) == 3:
                gray = np.mean(img, axis=2).astype(np.uint8)
            else:
                gray = img

            # Binary threshold
            silhouette = (gray < 128).astype(np.uint8) * 255

        return silhouette

    def validate_reconstruction(self, reference_paths, num_slices=12):
        """
        Run full validation loop.

        Args:
            reference_paths: Dict with 'front', 'side', 'top' image paths
            num_slices: Number of slices for reconstruction

        Returns:
            Tuple of (passed: bool, results: dict)
        """
        print("="*60)
        print("E2E VALIDATION TEST")
        print("="*60)

        # Step 1: Generate 3D model
        print("\n[1/4] Generating 3D model from reference images...")
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
        print("\n[2/4] Setting up render configuration...")
        self.setup_render_settings()
        print("✓ Render settings configured")

        # Step 3: Render orthogonal views
        print("\n[3/4] Rendering orthogonal views...")
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

            # Compare
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

    def print_detailed_results(self):
        """Print detailed comparison results."""
        if not self.results:
            print("No results to display")
            return

        print("\nDetailed Results:")
        print("-" * 60)
        print(f"{'View':<10} {'IoU':>8} {'Intersection':>12} {'Union':>10} {'PixDiff':>10}")
        print("-" * 60)

        for view, metrics in self.results.items():
            print(f"{view:<10} "
                  f"{metrics['iou']:>8.3f} "
                  f"{metrics['intersection']:>12d} "
                  f"{metrics['union']:>10d} "
                  f"{metrics['pixel_difference']:>10.2f}")

        print("-" * 60)


def test_with_sample_images():
    """Test with built-in sample images."""
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

    # Run validation
    validator = E2EValidator(iou_threshold=0.7)
    passed, results = validator.validate_reconstruction(reference_paths, num_slices=12)

    # Print detailed results
    validator.print_detailed_results()

    return passed


def test_with_custom_images(front, side, top):
    """
    Test with custom reference images.

    Args:
        front: Path to front view image
        side: Path to side view image
        top: Path to top view image

    Returns:
        bool: Test passed
    """
    reference_paths = {
        'front': front,
        'side': side,
        'top': top
    }

    validator = E2EValidator(iou_threshold=0.7)
    passed, results = validator.validate_reconstruction(reference_paths, num_slices=12)

    validator.print_detailed_results()

    return passed


if __name__ == "__main__":
    # Run test with sample images
    success = test_with_sample_images()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
