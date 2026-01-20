"""
Vertical profile extraction from reference images.

This module extracts width-at-height profiles from images by converting them
to filled silhouettes first, enabling accurate 3D mesh reconstruction.
"""

import numpy as np
import cv2
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.interpolate import interp1d
from typing import List, Tuple


def interpolate_profile(
    widths: np.ndarray,
    method: str = 'linear'
) -> np.ndarray:
    """
    Interpolate missing values in profile with various methods.

    Args:
        widths: Array of width measurements (may contain NaN)
        method: Interpolation method - 'linear', 'cubic', 'median_linear', 'gaussian_linear'

    Returns:
        Array with NaN values interpolated
    """
    # Handle missing data (NaN values) with interpolation
    if np.isnan(widths).any():
        valid_indices = ~np.isnan(widths)
        if valid_indices.sum() >= 2:
            # Get valid data points
            valid_positions = np.where(valid_indices)[0]
            valid_widths = widths[valid_indices]

            # Apply preprocessing based on method
            if method == 'median_linear':
                # C2: Apply median filter to valid widths before interpolation
                valid_widths = median_filter(valid_widths, size=3)
            elif method == 'gaussian_linear':
                # C3: Apply Gaussian smoothing to valid widths before interpolation
                valid_widths = gaussian_filter1d(valid_widths, sigma=1.0)

            # Choose interpolation kind
            if method == 'cubic':
                # C1: Cubic spline interpolation
                interp_kind = 'cubic'
            else:
                # Linear interpolation (baseline, median_linear, gaussian_linear)
                interp_kind = 'linear'

            # Create interpolation function
            interp_func = interp1d(
                valid_positions,
                valid_widths,
                kind=interp_kind,
                fill_value='extrapolate'
            )

            # Interpolate missing values
            invalid_positions = np.where(~valid_indices)[0]
            widths = widths.copy()
            widths[invalid_positions] = interp_func(invalid_positions)
        else:
            # Not enough valid data - return as is
            pass

    return widths


def extract_silhouette_from_image(image: np.ndarray) -> np.ndarray:
    """
    Extract filled silhouette mask from image.

    Args:
        image: Input image (grayscale or color)

    Returns:
        Binary mask (0 or 255) where object pixels are 255
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            # RGBA: use alpha channel
            gray = image[:, :, 3]
        else:
            # RGB: convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Threshold to create binary mask
    # Assume object is darker than background (common for line art/silhouettes)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Fill any holes in the silhouette
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create filled silhouette
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 255, -1)  # -1 thickness = filled

    return filled


def extract_vertical_profile(
    image: np.ndarray,
    num_samples: int = 100,
    interpolation_method: str = 'linear'
) -> List[Tuple[float, float]]:
    """
    Extract vertical profile from image.

    Converts image to filled silhouette, then scans vertically to measure
    width at each height. Returns normalized (height, radius) tuples.

    Args:
        image: Input image (can be original image or edge-detected)
        num_samples: Number of vertical samples to take
        interpolation_method: Method for interpolating missing values
            - 'linear': Linear interpolation (baseline)
            - 'cubic': Cubic spline interpolation
            - 'median_linear': Median filter + linear interpolation
            - 'gaussian_linear': Gaussian smoothing + linear interpolation

    Returns:
        List of (height, radius) tuples where:
            - height: Normalized 0 (bottom) to 1 (top)
            - radius: Normalized 0 (thinnest) to 1 (widest)
    """
    if image is None or image.size == 0:
        raise ValueError("Image is empty or None")

    # Convert to filled silhouette first
    silhouette = extract_silhouette_from_image(image)

    height, width = silhouette.shape

    if height < 2 or width < 2:
        raise ValueError(f"Image too small: {silhouette.shape}")

    # Initialize profile storage
    widths = []

    # Sample at regular vertical intervals (from bottom to top of image)
    # Note: Image coordinates have y=0 at top, but we treat bottom as z=0
    for i in range(num_samples):
        # Map sample index to image row (inverted: bottom = high row, top = low row)
        y = int(height - 1 - (i / (num_samples - 1)) * (height - 1))

        # Get the row of pixels
        row = silhouette[y, :]

        # Find leftmost and rightmost filled pixels
        filled_positions = np.where(row > 127)[0]

        if len(filled_positions) >= 2:
            # Measure width between outermost filled pixels
            left_edge = filled_positions[0]
            right_edge = filled_positions[-1]
            measured_width = right_edge - left_edge
        elif len(filled_positions) == 1:
            # Single pixel - minimal width
            measured_width = 1
        else:
            # No filled pixels - assign NaN for later interpolation
            measured_width = np.nan

        widths.append(measured_width)

    # Convert to numpy array for processing
    widths = np.array(widths)

    # Handle missing data (NaN values) with interpolation
    widths = interpolate_profile(widths, method=interpolation_method)

    # Check if we still have NaN values (not enough valid data)
    if np.isnan(widths).any():
        # Not enough valid data - use uniform profile
        widths = np.full(num_samples, width * 0.8)

    # Clamp widths to non-negative (extrapolation can produce negative values)
    widths = np.maximum(widths, 0)

    # Apply median filter to reduce noise (size=3 preserves more detail than size=5)
    widths = median_filter(widths, size=3)

    # Normalize widths to 0-1 range
    max_width = np.max(widths)
    if max_width > 0:
        normalized_widths = widths / max_width
    else:
        # Fallback: uniform profile
        normalized_widths = np.ones(num_samples) * 0.8

    # Create (height, radius) tuples
    # Height: 0 at bottom, 1 at top
    profile = []
    for i in range(num_samples):
        height_normalized = i / (num_samples - 1)
        radius_normalized = normalized_widths[i]
        profile.append((height_normalized, radius_normalized))

    return profile


def smooth_profile(
    profile: List[Tuple[float, float]],
    window_size: int = 5
) -> List[Tuple[float, float]]:
    """
    Apply additional smoothing to a profile.

    Args:
        profile: List of (height, radius) tuples
        window_size: Size of smoothing window

    Returns:
        Smoothed profile
    """
    if not profile or len(profile) < window_size:
        return profile

    heights = np.array([h for h, r in profile])
    radii = np.array([r for h, r in profile])

    # Apply median filter
    smoothed_radii = median_filter(radii, size=window_size)

    # Reconstruct profile
    return [(h, r) for h, r in zip(heights, smoothed_radii)]


def validate_profile(profile: List[Tuple[float, float]]) -> bool:
    """
    Validate that a profile is well-formed.

    Args:
        profile: List of (height, radius) tuples

    Returns:
        True if valid, False otherwise
    """
    if not profile:
        return False

    # Check that heights are monotonically increasing
    heights = [h for h, r in profile]
    if not all(heights[i] <= heights[i+1] for i in range(len(heights)-1)):
        return False

    # Check that all radii are in valid range [0, 1]
    radii = [r for h, r in profile]
    if not all(0 <= r <= 1 for r in radii):
        return False

    # Check that heights span [0, 1]
    if heights[0] != 0.0 or heights[-1] != 1.0:
        return False

    return True
