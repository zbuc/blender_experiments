"""
Multi-View Silhouette Consistency Validation (EXP-J)

Validates geometric consistency between orthogonal views. Unlike E2E validation
which compares each rendered view to its reference image, this validator checks
that the views are geometrically consistent with EACH OTHER.

For orthogonal views (front, side, top) of a 3D object:
- Front and side views should have the same height
- Width from front view should match Y-extent in top view
- Depth from side view should match X-extent in top view
- Profile measurements should be consistent across views

Usage:
    # In Blender
    from blender_blocking.multi_view_consistency import MultiViewConsistencyValidator

    validator = MultiViewConsistencyValidator(tolerance=0.05)
    passed, results = validator.validate_consistency(rendered_paths)
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from typing import Dict, Tuple, List

# Add ~/blender_python_packages for dependencies
sys.path.insert(0, str(Path.home() / 'blender_python_packages'))

from blender_blocking.integration.image_processing.image_loader import load_image
from blender_blocking.integration.shape_matching.profile_extractor import extract_silhouette_from_image


class MultiViewConsistencyValidator:
    """
    Validates geometric consistency between multiple orthogonal views.

    Checks that rendered views are internally consistent, which validates
    that the 3D reconstruction is geometrically sound.
    """

    def __init__(self, tolerance=0.05):
        """
        Initialize validator.

        Args:
            tolerance: Fractional tolerance for consistency checks (0.05 = 5%)
        """
        self.tolerance = tolerance
        self.results = {}

    def extract_silhouette(self, image_path: str) -> np.ndarray:
        """
        Extract binary silhouette from image.

        Args:
            image_path: Path to image file

        Returns:
            Binary numpy array (0 or 255)
        """
        img = load_image(image_path)
        return extract_silhouette_from_image(img)

    def get_bounding_box(self, silhouette: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Get bounding box of silhouette.

        Args:
            silhouette: Binary silhouette image

        Returns:
            Tuple of (min_y, max_y, min_x, max_x)
        """
        # Find all filled pixels
        filled_pixels = np.where(silhouette > 127)

        if len(filled_pixels[0]) == 0:
            return (0, 0, 0, 0)

        min_y = np.min(filled_pixels[0])
        max_y = np.max(filled_pixels[0])
        min_x = np.min(filled_pixels[1])
        max_x = np.max(filled_pixels[1])

        return (min_y, max_y, min_x, max_x)

    def extract_vertical_profile(self, silhouette: np.ndarray, num_samples: int = 100) -> np.ndarray:
        """
        Extract width at each height from silhouette.

        Args:
            silhouette: Binary silhouette image
            num_samples: Number of vertical samples

        Returns:
            Array of widths (in pixels) at each height
        """
        height, width = silhouette.shape
        widths = []

        for i in range(num_samples):
            # Sample from bottom to top
            y = int(height - 1 - (i / (num_samples - 1)) * (height - 1))
            row = silhouette[y, :]

            # Find leftmost and rightmost filled pixels
            filled_positions = np.where(row > 127)[0]

            if len(filled_positions) >= 2:
                measured_width = filled_positions[-1] - filled_positions[0]
            elif len(filled_positions) == 1:
                measured_width = 1
            else:
                measured_width = 0

            widths.append(measured_width)

        return np.array(widths)

    def extract_horizontal_profile(self, silhouette: np.ndarray, num_samples: int = 100) -> np.ndarray:
        """
        Extract depth at each width from silhouette (for top view).

        Args:
            silhouette: Binary silhouette image
            num_samples: Number of horizontal samples

        Returns:
            Array of depths (in pixels) at each horizontal position
        """
        height, width = silhouette.shape
        depths = []

        for i in range(num_samples):
            # Sample from left to right
            x = int((i / (num_samples - 1)) * (width - 1))
            col = silhouette[:, x]

            # Find topmost and bottommost filled pixels
            filled_positions = np.where(col > 127)[0]

            if len(filled_positions) >= 2:
                measured_depth = filled_positions[-1] - filled_positions[0]
            elif len(filled_positions) == 1:
                measured_depth = 1
            else:
                measured_depth = 0

            depths.append(measured_depth)

        return np.array(depths)

    def check_height_consistency(self, front_silhouette: np.ndarray,
                                 side_silhouette: np.ndarray) -> Tuple[bool, Dict]:
        """
        Check that front and side views have the same vertical extent.

        Args:
            front_silhouette: Front view silhouette
            side_silhouette: Side view silhouette

        Returns:
            Tuple of (passed, metrics_dict)
        """
        front_bbox = self.get_bounding_box(front_silhouette)
        side_bbox = self.get_bounding_box(side_silhouette)

        front_height = front_bbox[1] - front_bbox[0]
        side_height = side_bbox[1] - side_bbox[0]

        # Calculate relative difference
        max_height = max(front_height, side_height)
        if max_height == 0:
            relative_diff = 0
        else:
            relative_diff = abs(front_height - side_height) / max_height

        passed = relative_diff <= self.tolerance

        metrics = {
            'front_height': front_height,
            'side_height': side_height,
            'relative_diff': relative_diff,
            'tolerance': self.tolerance,
            'passed': passed
        }

        return passed, metrics

    def check_profile_consistency(self, front_silhouette: np.ndarray,
                                  side_silhouette: np.ndarray,
                                  top_silhouette: np.ndarray,
                                  num_samples: int = 100) -> Tuple[bool, Dict]:
        """
        Check that width/depth profiles are consistent with top view.

        At each height H:
        - Front view max width should correlate with top view Y-extent
        - Side view max depth should correlate with top view X-extent

        Args:
            front_silhouette: Front view silhouette
            side_silhouette: Side view silhouette
            top_silhouette: Top view silhouette
            num_samples: Number of profile samples

        Returns:
            Tuple of (passed, metrics_dict)
        """
        # Extract profiles
        front_widths = self.extract_vertical_profile(front_silhouette, num_samples)
        side_depths = self.extract_vertical_profile(side_silhouette, num_samples)

        # Extract top view bounding box
        top_bbox = self.get_bounding_box(top_silhouette)
        top_width = top_bbox[3] - top_bbox[2]  # X extent
        top_height = top_bbox[1] - top_bbox[0]  # Y extent

        # Get maximum widths/depths
        max_front_width = np.max(front_widths) if len(front_widths) > 0 else 0
        max_side_depth = np.max(side_depths) if len(side_depths) > 0 else 0

        # Check consistency: front max width should match top Y-extent
        # and side max depth should match top X-extent
        front_top_diff = abs(max_front_width - top_height) / max(max_front_width, top_height, 1)
        side_top_diff = abs(max_side_depth - top_width) / max(max_side_depth, top_width, 1)

        avg_diff = (front_top_diff + side_top_diff) / 2
        passed = avg_diff <= self.tolerance

        metrics = {
            'max_front_width': max_front_width,
            'max_side_depth': max_side_depth,
            'top_y_extent': top_height,
            'top_x_extent': top_width,
            'front_top_diff': front_top_diff,
            'side_top_diff': side_top_diff,
            'avg_diff': avg_diff,
            'tolerance': self.tolerance,
            'passed': passed
        }

        return passed, metrics

    def check_bounding_box_consistency(self, front_silhouette: np.ndarray,
                                       side_silhouette: np.ndarray,
                                       top_silhouette: np.ndarray) -> Tuple[bool, Dict]:
        """
        Check that bounding boxes are consistent across views.

        For orthogonal views:
        - Front width should match top Y-extent
        - Side width should match top X-extent
        - Front height should match side height

        Args:
            front_silhouette: Front view silhouette
            side_silhouette: Side view silhouette
            top_silhouette: Top view silhouette

        Returns:
            Tuple of (passed, metrics_dict)
        """
        front_bbox = self.get_bounding_box(front_silhouette)
        side_bbox = self.get_bounding_box(side_silhouette)
        top_bbox = self.get_bounding_box(top_silhouette)

        # Extract dimensions
        front_height = front_bbox[1] - front_bbox[0]
        front_width = front_bbox[3] - front_bbox[2]

        side_height = side_bbox[1] - side_bbox[0]
        side_width = side_bbox[3] - side_bbox[2]

        top_height = top_bbox[1] - top_bbox[0]  # Y extent
        top_width = top_bbox[3] - top_bbox[2]   # X extent

        # Check consistency
        errors = []

        # Height consistency (already checked separately, but include here)
        height_diff = abs(front_height - side_height) / max(front_height, side_height, 1)
        errors.append(height_diff)

        # Width-to-top-Y consistency
        width_y_diff = abs(front_width - top_height) / max(front_width, top_height, 1)
        errors.append(width_y_diff)

        # Depth-to-top-X consistency
        depth_x_diff = abs(side_width - top_width) / max(side_width, top_width, 1)
        errors.append(depth_x_diff)

        max_error = max(errors)
        avg_error = sum(errors) / len(errors)

        passed = max_error <= self.tolerance

        metrics = {
            'front_height': front_height,
            'front_width': front_width,
            'side_height': side_height,
            'side_width': side_width,
            'top_height': top_height,
            'top_width': top_width,
            'height_diff': height_diff,
            'width_y_diff': width_y_diff,
            'depth_x_diff': depth_x_diff,
            'max_error': max_error,
            'avg_error': avg_error,
            'tolerance': self.tolerance,
            'passed': passed
        }

        return passed, metrics

    def validate_consistency(self, rendered_paths: Dict[str, str]) -> Tuple[bool, Dict]:
        """
        Run all consistency checks on rendered orthogonal views.

        Args:
            rendered_paths: Dict with 'front', 'side', 'top' image paths

        Returns:
            Tuple of (passed, results_dict)
        """
        print("="*60)
        print("MULTI-VIEW CONSISTENCY VALIDATION (EXP-J)")
        print("="*60)

        # Extract silhouettes
        print("\n[1/4] Extracting silhouettes from rendered views...")
        silhouettes = {}
        for view in ['front', 'side', 'top']:
            if view not in rendered_paths:
                print(f"ERROR: Missing {view} view")
                return False, {}

            silhouettes[view] = self.extract_silhouette(rendered_paths[view])
            print(f"✓ Extracted {view} silhouette")

        # Run consistency checks
        self.results = {}

        # Check 1: Height consistency
        print("\n[2/4] Checking height consistency (front vs side)...")
        height_passed, height_metrics = self.check_height_consistency(
            silhouettes['front'],
            silhouettes['side']
        )
        self.results['height_consistency'] = height_metrics
        print(f"  Height difference: {height_metrics['relative_diff']:.3f}", end="")
        print(f"  {'✓ PASS' if height_passed else '✗ FAIL'}")

        # Check 2: Profile consistency
        print("\n[3/4] Checking profile consistency (views vs top)...")
        profile_passed, profile_metrics = self.check_profile_consistency(
            silhouettes['front'],
            silhouettes['side'],
            silhouettes['top']
        )
        self.results['profile_consistency'] = profile_metrics
        print(f"  Front-top difference: {profile_metrics['front_top_diff']:.3f}")
        print(f"  Side-top difference: {profile_metrics['side_top_diff']:.3f}")
        print(f"  Average: {profile_metrics['avg_diff']:.3f}", end="")
        print(f"  {'✓ PASS' if profile_passed else '✗ FAIL'}")

        # Check 3: Bounding box consistency
        print("\n[4/4] Checking bounding box consistency...")
        bbox_passed, bbox_metrics = self.check_bounding_box_consistency(
            silhouettes['front'],
            silhouettes['side'],
            silhouettes['top']
        )
        self.results['bounding_box_consistency'] = bbox_metrics
        print(f"  Height consistency: {bbox_metrics['height_diff']:.3f}")
        print(f"  Width-Y consistency: {bbox_metrics['width_y_diff']:.3f}")
        print(f"  Depth-X consistency: {bbox_metrics['depth_x_diff']:.3f}")
        print(f"  Max error: {bbox_metrics['max_error']:.3f}", end="")
        print(f"  {'✓ PASS' if bbox_passed else '✗ FAIL'}")

        # Overall result
        all_passed = height_passed and profile_passed and bbox_passed

        print("\n" + "="*60)
        print(f"Tolerance: {self.tolerance:.3f}")
        print(f"Result:    {'✓ ALL CHECKS PASSED' if all_passed else '✗ SOME CHECKS FAILED'}")
        print("="*60)

        return all_passed, self.results

    def print_detailed_results(self):
        """Print detailed consistency check results."""
        if not self.results:
            print("No results to display")
            return

        print("\nDetailed Consistency Results:")
        print("-" * 60)

        for check_name, metrics in self.results.items():
            print(f"\n{check_name.upper().replace('_', ' ')}:")
            for key, value in metrics.items():
                if key == 'passed':
                    continue
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")

        print("-" * 60)
