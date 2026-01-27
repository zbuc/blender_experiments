"""Contour detection and shape analysis utilities."""

from __future__ import annotations

import numpy as np
import cv2
from typing import Any, Dict, List, Tuple, Optional, Union


def find_contours(
    edge_image: np.ndarray,
    *,
    mode: str = "external",
    return_hierarchy: bool = False,
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], Optional[np.ndarray]]]:
    """
    Find contours in an edge-detected image.

    Args:
        edge_image: Binary edge image
        mode: Retrieval mode ("external", "ccomp", "tree", "hierarchy")
        return_hierarchy: Whether to return the contour hierarchy

    Returns:
        List of contours, and optionally the hierarchy
    """
    if edge_image is None or edge_image.size == 0:
        print("Warning: Empty edge image provided to find_contours")
        return ([], None) if return_hierarchy else []

    if edge_image.ndim == 3:
        if edge_image.shape[2] == 4:
            edge_image = cv2.cvtColor(edge_image, cv2.COLOR_RGBA2GRAY)
        else:
            edge_image = cv2.cvtColor(edge_image, cv2.COLOR_RGB2GRAY)

    mode_map = {
        "external": cv2.RETR_EXTERNAL,
        "ccomp": cv2.RETR_CCOMP,
        "tree": cv2.RETR_TREE,
        "hierarchy": cv2.RETR_CCOMP,
    }
    if mode not in mode_map:
        raise ValueError(f"Unknown contour mode: {mode}")

    contours, hierarchy = cv2.findContours(
        edge_image, mode_map[mode], cv2.CHAIN_APPROX_SIMPLE
    )
    if return_hierarchy:
        return contours, hierarchy
    return contours


def analyze_shape(contour: np.ndarray) -> Dict[str, Any]:
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
        "area": area,
        "perimeter": perimeter,
        "centroid": (cx, cy),
        "bounding_box": (x, y, w, h),
        "circularity": circularity,
        "aspect_ratio": w / h if h > 0 else 0,
    }
