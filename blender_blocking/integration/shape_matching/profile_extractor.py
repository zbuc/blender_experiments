"""
Vertical profile extraction from reference images.

This module extracts width-at-height profiles from images by converting them
to filled silhouettes first, enabling accurate 3D mesh reconstruction.
"""

import numpy as np
import cv2
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d
from typing import List, Tuple


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
    num_samples: int = 100
) -> List[Tuple[float, float]]:
    """
    Extract vertical profile from image.

    Converts image to filled silhouette, then scans vertically to measure
    width at each height. Returns normalized (height, radius) tuples.

    Args:
        image: Input image (can be original image or edge-detected)
        num_samples: Number of vertical samples to take

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
    if np.isnan(widths).any():
        valid_indices = ~np.isnan(widths)
        if valid_indices.sum() >= 2:
            # Interpolate missing values from valid ones
            valid_positions = np.where(valid_indices)[0]
            valid_widths = widths[valid_indices]

            interp_func = interp1d(
                valid_positions,
                valid_widths,
                kind='linear',
                fill_value='extrapolate'
            )

            invalid_positions = np.where(~valid_indices)[0]
            widths[invalid_positions] = interp_func(invalid_positions)
        else:
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


def calculate_profile_confidence(profile: List[Tuple[float, float]], image: np.ndarray = None) -> float:
    """
    Calculate a confidence score for a profile based on consistency and coverage.

    Args:
        profile: List of (height, radius) tuples
        image: Optional source image for additional metrics

    Returns:
        Confidence score between 0 and 1
    """
    if not profile or len(profile) < 3:
        return 0.0

    radii = np.array([r for h, r in profile])

    # Metric 1: Smoothness (lower variance = higher confidence)
    # Calculate local changes
    changes = np.abs(np.diff(radii))
    smoothness = 1.0 - np.clip(np.mean(changes) / 0.5, 0, 1)  # Normalize by expected max change

    # Metric 2: Coverage (how much of the profile has non-zero radii)
    coverage = np.sum(radii > 0.01) / len(radii)

    # Metric 3: Silhouette fill quality (if image provided)
    fill_quality = 1.0
    if image is not None:
        try:
            silhouette = extract_silhouette_from_image(image)
            # Calculate ratio of filled pixels to bounding box
            filled_pixels = np.sum(silhouette > 127)
            total_pixels = silhouette.shape[0] * silhouette.shape[1]
            fill_quality = np.clip(filled_pixels / (total_pixels * 0.5), 0, 1)  # Expect ~50% fill
        except:
            fill_quality = 1.0

    # Combine metrics (weighted average)
    confidence = 0.4 * smoothness + 0.3 * coverage + 0.3 * fill_quality

    return float(np.clip(confidence, 0, 1))


def fuse_profiles(
    front_profile: List[Tuple[float, float]],
    side_profile: List[Tuple[float, float]],
    fusion_strategy: str = "equal",
    front_weight: float = 0.5,
    side_weight: float = 0.5,
    front_image: np.ndarray = None,
    side_image: np.ndarray = None
) -> List[Tuple[float, float]]:
    """
    Fuse front and side vertical profiles using various strategies.

    Args:
        front_profile: Profile from front view (height, radius) tuples
        side_profile: Profile from side view (height, radius) tuples
        fusion_strategy: Strategy for fusion - "equal", "front_heavy", "side_heavy", "adaptive", "custom"
        front_weight: Weight for front profile (used if strategy is "custom")
        side_weight: Weight for side profile (used if strategy is "custom")
        front_image: Optional front image for adaptive weighting
        side_image: Optional side image for adaptive weighting

    Returns:
        Fused profile as list of (height, radius) tuples

    Raises:
        ValueError: If profiles are incompatible or invalid
    """
    if not front_profile or not side_profile:
        raise ValueError("Both front and side profiles must be provided")

    if len(front_profile) != len(side_profile):
        raise ValueError(f"Profiles must have same length: {len(front_profile)} vs {len(side_profile)}")

    # Extract heights and radii
    front_heights = np.array([h for h, r in front_profile])
    front_radii = np.array([r for h, r in front_profile])
    side_heights = np.array([h for h, r in side_profile])
    side_radii = np.array([r for h, r in side_profile])

    # Verify heights match
    if not np.allclose(front_heights, side_heights):
        raise ValueError("Profile heights must match between front and side views")

    # Determine weights based on strategy
    if fusion_strategy == "equal":
        # E1: Equal weights (0.5, 0.5)
        w_front = 0.5
        w_side = 0.5
    elif fusion_strategy == "front_heavy":
        # E2: Front-heavy (0.6, 0.4)
        w_front = 0.6
        w_side = 0.4
    elif fusion_strategy == "side_heavy":
        # E3: Side-heavy (0.4, 0.6)
        w_front = 0.4
        w_side = 0.6
    elif fusion_strategy == "adaptive":
        # E4: Adaptive weights based on profile confidence
        front_conf = calculate_profile_confidence(front_profile, front_image)
        side_conf = calculate_profile_confidence(side_profile, side_image)

        total_conf = front_conf + side_conf
        if total_conf > 0:
            w_front = front_conf / total_conf
            w_side = side_conf / total_conf
        else:
            # Fallback to equal if both confidences are 0
            w_front = 0.5
            w_side = 0.5
    elif fusion_strategy == "custom":
        # Custom weights provided by user
        total = front_weight + side_weight
        if total == 0:
            raise ValueError("Weights must sum to non-zero value")
        w_front = front_weight / total
        w_side = side_weight / total
    else:
        raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

    # Fuse radii using weighted average
    fused_radii = w_front * front_radii + w_side * side_radii

    # Normalize to [0, 1] range
    max_radius = np.max(fused_radii)
    if max_radius > 0:
        fused_radii = fused_radii / max_radius
    else:
        fused_radii = np.ones_like(fused_radii) * 0.5

    # Create fused profile
    fused_profile = [(h, r) for h, r in zip(front_heights, fused_radii)]

    return fused_profile
