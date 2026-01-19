"""Image processing utilities for edge detection and normalization."""

import numpy as np
import cv2
from typing import Tuple


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to 0-255 range.

    Args:
        image: Input image array

    Returns:
        Normalized image
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert to grayscale if color
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Normalize to 0-255
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def extract_edges(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150
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
    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # Detect edges
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    return edges


def process_image(
    image: np.ndarray,
    extract_edges_flag: bool = True,
    normalize_flag: bool = True
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
