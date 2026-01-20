"""Test E2E validation with elliptical vase to verify directional profile fix."""

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

from blender_blocking.test_e2e_validation import E2EValidator

def main():
    """Run E2E test with elliptical vase."""
    base_dir = Path(__file__).parent
    test_images_dir = base_dir / 'test_images'

    reference_paths = {
        'front': str(test_images_dir / 'elliptical_vase_front.png'),
        'side': str(test_images_dir / 'elliptical_vase_side.png'),
        'top': str(test_images_dir / 'elliptical_vase_top.png')
    }

    print("="*70)
    print("ELLIPTICAL VASE E2E VALIDATION")
    print("="*70)
    print("\nThis test validates the directional profile fix:")
    print("  - Front profile (narrower) → X-axis")
    print("  - Side profile (wider) → Y-axis")
    print("  - Should create elliptical cross-section matching top view")
    print("="*70)

    # Run validation with many slices
    validator = E2EValidator(iou_threshold=0.7)
    passed, results = validator.validate_reconstruction(reference_paths, num_slices=120)

    # Print detailed results
    validator.print_detailed_results()

    return passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
