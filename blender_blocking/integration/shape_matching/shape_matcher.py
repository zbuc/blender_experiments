"""Shape matching and comparison utilities."""

import numpy as np
import cv2
from typing import List, Tuple


def match_shapes(
    contour1: np.ndarray,
    contour2: np.ndarray,
    method: int = cv2.CONTOURS_MATCH_I2
) -> float:
    """
    Compare two contours using shape matching.

    Args:
        contour1: First contour
        contour2: Second contour
        method: OpenCV matching method

    Returns:
        Similarity score (lower is better, 0 is perfect match)
    """
    return cv2.matchShapes(contour1, contour2, method, 0.0)


def compare_silhouettes(
    image1: np.ndarray,
    image2: np.ndarray
) -> Tuple[float, Dict]:
    """
    Compare silhouettes from two images.

    Args:
        image1: First binary image
        image2: Second binary image

    Returns:
        Tuple of (similarity_score, comparison_details)
    """
    # Ensure images are same size
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Calculate intersection and union
    intersection = np.logical_and(image1 > 0, image2 > 0).sum()
    union = np.logical_or(image1 > 0, image2 > 0).sum()

    # IoU (Intersection over Union)
    iou = intersection / union if union > 0 else 0

    # Pixel-wise difference
    difference = np.abs(image1.astype(float) - image2.astype(float)).mean()

    details = {
        'iou': iou,
        'intersection': intersection,
        'union': union,
        'pixel_difference': difference
    }

    return iou, details
