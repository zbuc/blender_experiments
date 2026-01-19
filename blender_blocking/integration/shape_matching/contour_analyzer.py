"""Contour detection and shape analysis utilities."""

import numpy as np
import cv2
from typing import List, Dict, Tuple


def find_contours(edge_image: np.ndarray) -> List[np.ndarray]:
    """
    Find contours in an edge-detected image.

    Args:
        edge_image: Binary edge image

    Returns:
        List of contours
    """
    contours, _ = cv2.findContours(
        edge_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def analyze_shape(contour: np.ndarray) -> Dict[str, any]:
    """
    Analyze a contour to extract shape properties.

    Args:
        contour: Contour points

    Returns:
        Dictionary of shape properties (area, perimeter, centroid, etc.)
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Calculate moments for centroid
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    # Fit bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)

    # Calculate circularity (4π * area / perimeter²)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

    return {
        'area': area,
        'perimeter': perimeter,
        'centroid': (cx, cy),
        'bounding_box': (x, y, w, h),
        'circularity': circularity,
        'aspect_ratio': w / h if h > 0 else 0
    }
