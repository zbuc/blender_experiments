"""Profile extraction utilities for vertical silhouette analysis.

Extracts vertical radius profiles from reference silhouettes for vertex refinement.
"""

import numpy as np
import cv2
from typing import List, Tuple


def extract_silhouette_from_image(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    """
    Extract binary silhouette from image.

    Args:
        image: Input image (grayscale or color)
        threshold: Threshold value for binarization (0-255)

    Returns:
        Binary silhouette as numpy array (0 or 255)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            # RGBA - use alpha channel
            alpha = image[:, :, 3]
            silhouette = (alpha > threshold).astype(np.uint8) * 255
        elif image.shape[2] == 3:
            # RGB - convert to grayscale and threshold
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            silhouette = (gray > threshold).astype(np.uint8) * 255
        else:
            silhouette = image
    else:
        # Already grayscale
        silhouette = (image > threshold).astype(np.uint8) * 255

    return silhouette


def extract_vertical_profile(
    image: np.ndarray,
    num_samples: int = 100,
    threshold: int = 128
) -> List[Tuple[float, float]]:
    """
    Extract vertical radius profile from silhouette image.

    Samples the silhouette at different heights and calculates the average
    horizontal radius (half-width) at each height.

    Args:
        image: Input image (will be converted to binary silhouette)
        num_samples: Number of height samples to extract
        threshold: Threshold for silhouette extraction

    Returns:
        List of (height_normalized, radius) tuples where:
        - height_normalized is in range [0, 1] (0=bottom, 1=top)
        - radius is the average half-width at that height (in pixels)
    """
    # Extract binary silhouette
    silhouette = extract_silhouette_from_image(image, threshold)

    height, width = silhouette.shape

    if height == 0 or width == 0:
        return [(0.0, 0.0), (1.0, 0.0)]

    # Find vertical bounds of the silhouette
    vertical_profile = np.any(silhouette > 0, axis=1)
    y_coords = np.where(vertical_profile)[0]

    if len(y_coords) == 0:
        # No silhouette found
        return [(0.0, 0.0), (1.0, 0.0)]

    y_min = y_coords[0]
    y_max = y_coords[-1]
    y_range = y_max - y_min

    if y_range == 0:
        # Silhouette has zero height
        return [(0.0, 0.0), (1.0, 0.0)]

    # Sample at different heights
    profile = []

    for i in range(num_samples):
        # Normalized height from 0 (bottom) to 1 (top)
        height_norm = i / (num_samples - 1) if num_samples > 1 else 0.5

        # Convert to pixel row (flip Y since image coordinates are top-down)
        y_pixel = int(y_min + height_norm * y_range)
        y_pixel = max(0, min(height - 1, y_pixel))

        # Get horizontal slice at this height
        row = silhouette[y_pixel, :]

        # Find horizontal extent
        x_coords = np.where(row > 0)[0]

        if len(x_coords) == 0:
            # No silhouette at this height
            radius = 0.0
        else:
            # Calculate average radius (half of the width)
            x_min = x_coords[0]
            x_max = x_coords[-1]
            x_center = (x_min + x_max) / 2.0
            x_extent = x_max - x_min

            # Radius is half the width
            radius = x_extent / 2.0

            # Normalize radius to 0-1 range based on image width
            radius = radius / (width / 2.0)

        profile.append((height_norm, radius))

    return profile


def extract_horizontal_profile(
    image: np.ndarray,
    num_samples: int = 100,
    threshold: int = 128
) -> List[Tuple[float, float]]:
    """
    Extract horizontal radius profile from silhouette image.

    Similar to extract_vertical_profile but samples along the horizontal axis.
    Useful for side-view profiles.

    Args:
        image: Input image (will be converted to binary silhouette)
        num_samples: Number of horizontal samples to extract
        threshold: Threshold for silhouette extraction

    Returns:
        List of (position_normalized, radius) tuples where:
        - position_normalized is in range [0, 1] (0=left, 1=right)
        - radius is the average half-height at that position (in pixels)
    """
    # Extract binary silhouette
    silhouette = extract_silhouette_from_image(image, threshold)

    height, width = silhouette.shape

    if height == 0 or width == 0:
        return [(0.0, 0.0), (1.0, 0.0)]

    # Find horizontal bounds of the silhouette
    horizontal_profile = np.any(silhouette > 0, axis=0)
    x_coords = np.where(horizontal_profile)[0]

    if len(x_coords) == 0:
        # No silhouette found
        return [(0.0, 0.0), (1.0, 0.0)]

    x_min = x_coords[0]
    x_max = x_coords[-1]
    x_range = x_max - x_min

    if x_range == 0:
        # Silhouette has zero width
        return [(0.0, 0.0), (1.0, 0.0)]

    # Sample at different horizontal positions
    profile = []

    for i in range(num_samples):
        # Normalized position from 0 (left) to 1 (right)
        pos_norm = i / (num_samples - 1) if num_samples > 1 else 0.5

        # Convert to pixel column
        x_pixel = int(x_min + pos_norm * x_range)
        x_pixel = max(0, min(width - 1, x_pixel))

        # Get vertical slice at this position
        col = silhouette[:, x_pixel]

        # Find vertical extent
        y_coords = np.where(col > 0)[0]

        if len(y_coords) == 0:
            # No silhouette at this position
            radius = 0.0
        else:
            # Calculate average radius (half of the height)
            y_min_local = y_coords[0]
            y_max_local = y_coords[-1]
            y_extent = y_max_local - y_min_local

            # Radius is half the height
            radius = y_extent / 2.0

            # Normalize radius to 0-1 range based on image height
            radius = radius / (height / 2.0)

        profile.append((pos_norm, radius))

    return profile
