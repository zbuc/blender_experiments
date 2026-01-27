"""
Vertical profile extraction from reference images.

This module extracts width-at-height profiles from images by converting them
to filled silhouettes first, enabling accurate 3D mesh reconstruction.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d
from typing import List, Optional, Tuple

from geometry.silhouette import bbox_from_mask, extract_binary_silhouette


def extract_silhouette_from_image(image: np.ndarray) -> np.ndarray:
    """
    Extract filled silhouette mask from image.

    Args:
        image: Input image (grayscale or color)

    Returns:
        Binary mask (0 or 255) where object pixels are 255
    """
    mask = extract_binary_silhouette(image)
    return (mask.astype(np.uint8) * 255).astype(np.uint8)


def extract_vertical_profile(
    image: np.ndarray,
    num_samples: int = 100,
    *,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    already_silhouette: bool = False,
    smoothing_window: int = 3,
) -> List[Tuple[float, float]]:
    """
    Extract vertical profile from image.

    Converts image to filled silhouette, then scans vertically to measure
    width at each height. Returns normalized (height, radius) tuples.

    Args:
        image: Input image (can be original image or edge-detected)
        num_samples: Number of vertical samples to take
        bbox: Optional (x0, y0, x1, y1) crop bounds
        already_silhouette: If True, treat image as a binary silhouette
        smoothing_window: Median filter window size for width smoothing

    Returns:
        List of (height, radius) tuples where:
            - height: Normalized 0 (bottom) to 1 (top)
            - radius: Normalized 0 (thinnest) to 1 (widest)
    """
    if image is None or image.size == 0:
        raise ValueError("Image is empty or None")
    if num_samples <= 0:
        raise ValueError("num_samples must be >= 1")

    if already_silhouette:
        silhouette_bool = image.astype(bool)
    else:
        silhouette_bool = extract_binary_silhouette(image)

    if not silhouette_bool.any():
        raise ValueError("Silhouette mask is empty")

    if bbox is None:
        bbox_obj = bbox_from_mask(silhouette_bool)
        bbox = (bbox_obj.x0, bbox_obj.y0, bbox_obj.x1, bbox_obj.y1)

    x0, y0, x1, y1 = bbox
    silhouette = silhouette_bool[y0:y1, x0:x1]

    height, width = silhouette.shape

    if height < 2 or width < 2:
        raise ValueError(f"Image too small: {silhouette.shape}")

    # Initialize profile storage
    widths = []

    # Sample at regular vertical intervals (from bottom to top of image)
    # Note: Image coordinates have y=0 at top, but we treat bottom as z=0
    if num_samples == 1:
        sample_positions = np.array([0.5], dtype=np.float64)
    else:
        sample_positions = np.linspace(0.0, 1.0, num_samples, dtype=np.float64)

    row_indices = (height - 1 - sample_positions * (height - 1)).astype(int)
    rows = silhouette[row_indices, :]
    row_mask = rows.astype(bool)
    any_filled = row_mask.any(axis=1)

    left_edge = np.argmax(row_mask, axis=1)
    right_edge = width - 1 - np.argmax(row_mask[:, ::-1], axis=1)
    measured_widths = right_edge - left_edge + 1

    widths = np.where(any_filled, measured_widths.astype(float), np.nan)

    # Convert to numpy array for processing
    widths = np.array(widths)

    # Handle missing data (NaN values) with interpolation
    if np.isnan(widths).any():
        valid_indices = ~np.isnan(widths)
        if valid_indices.sum() >= 2:
            # Interpolate missing values from valid ones
            valid_positions = np.where(valid_indices)[0]
            valid_widths = widths[valid_indices]

            interp_func = interp1d(
                valid_positions, valid_widths, kind="linear", fill_value="extrapolate"
            )

            invalid_positions = np.where(~valid_indices)[0]
            widths[invalid_positions] = interp_func(invalid_positions)
        else:
            # Not enough valid data - use uniform profile
            widths = np.full(num_samples, width * 0.8)

    # Clamp widths to non-negative (extrapolation can produce negative values)
    widths = np.maximum(widths, 0)

    # Apply median filter to reduce noise (size=3 preserves more detail than size=5)
    widths = median_filter(widths, size=max(int(smoothing_window), 1))

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
        height_normalized = 0.5 if num_samples == 1 else i / (num_samples - 1)
        radius_normalized = normalized_widths[i]
        profile.append((height_normalized, radius_normalized))

    return profile


def smooth_profile(
    profile: List[Tuple[float, float]], window_size: int = 5
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
    if not all(heights[i] <= heights[i + 1] for i in range(len(heights) - 1)):
        return False

    # Check that all radii are in valid range [0, 1]
    radii = [r for h, r in profile]
    if not all(0 <= r <= 1 for r in radii):
        return False

    # Check that heights span [0, 1]
    if heights[0] != 0.0 or heights[-1] != 1.0:
        return False

    return True
