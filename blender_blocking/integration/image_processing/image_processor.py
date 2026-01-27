"""Image processing utilities for edge detection and normalization."""

from __future__ import annotations

import numpy as np
import cv2


def _to_gray_uint8(image: np.ndarray, prefer_alpha: bool = True) -> np.ndarray:
    """
    Convert an image to grayscale uint8, preferring alpha when available.

    Args:
        image: Input image array
        prefer_alpha: Whether to use alpha channel when present and meaningful

    Returns:
        Grayscale uint8 image
    """
    if image is None:
        raise ValueError("image is required")

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3:
        if image.shape[2] == 4:
            alpha = image[:, :, 3]
            if prefer_alpha and np.ptp(alpha) > 0:
                gray = alpha
            else:
                gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"Unsupported channel count: {image.shape[2]}")
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if gray.dtype != np.uint8:
        max_val = float(np.max(gray)) if gray.size else 0.0
        if max_val <= 1.0:
            gray = (gray.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
        else:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to 0-255 range.

    Args:
        image: Input image array

    Returns:
        Normalized image
    """
    image = _to_gray_uint8(image, prefer_alpha=True)

    # Normalize to 0-255
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def extract_edges(
    image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150
) -> np.ndarray:
    """
    Extract edges from image using Canny edge detection.

    Args:
        image: Input image array
        low_threshold: Lower threshold for Canny
        high_threshold: Upper threshold for Canny

    Returns:
        Edge map as binary image
    """
    gray = _to_gray_uint8(image, prefer_alpha=True)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # Detect edges
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    return edges


def process_image(
    image: np.ndarray, extract_edges_flag: bool = True, normalize_flag: bool = True
) -> np.ndarray:
    """
    Process image with normalization and optional edge extraction.

    Args:
        image: Input image array
        extract_edges_flag: Whether to extract edges
        normalize_flag: Whether to normalize image

    Returns:
        Processed image
    """
    result = image.copy()

    if normalize_flag:
        result = normalize_image(result)

    if extract_edges_flag:
        result = extract_edges(result)

    return result
