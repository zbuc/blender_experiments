"""Shape matching and comparison utilities."""

from __future__ import annotations

import numpy as np
import cv2
from typing import Dict, Tuple, Union

from validation.silhouette_iou import (
    canonicalize_mask_cached,
    compute_mask_iou,
    mask_from_image_array,
)


def match_shapes(
    contour1: np.ndarray, contour2: np.ndarray, method: int = cv2.CONTOURS_MATCH_I2
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
    if contour1 is None or contour2 is None:
        raise ValueError("Contours must not be None")
    if len(contour1) < 3 or len(contour2) < 3:
        raise ValueError("Contours must have at least 3 points to match")
    return cv2.matchShapes(contour1, contour2, method, 0.0)


def compare_silhouettes(
    image1: np.ndarray,
    image2: np.ndarray,
    *,
    output_size: int = 256,
    padding_frac: float = 0.1,
    anchor: str = "bottom_center",
) -> Tuple[float, Dict[str, Union[float, int]]]:
    """
    Compare silhouettes from two images.

    Args:
        image1: First binary image
        image2: Second binary image

    Returns:
        Tuple of (similarity_score, comparison_details)
    """
    if image1 is None or image2 is None or image1.size == 0 or image2.size == 0:
        raise ValueError("Images must be non-empty for silhouette comparison")

    mask1 = mask_from_image_array(image1)
    mask2 = mask_from_image_array(image2)

    canon1 = canonicalize_mask_cached(
        mask1, output_size=output_size, padding_frac=padding_frac, anchor=anchor
    )
    canon2 = canonicalize_mask_cached(
        mask2, output_size=output_size, padding_frac=padding_frac, anchor=anchor
    )

    result = compute_mask_iou(canon1, canon2)
    difference = float(np.abs(canon1.astype(float) - canon2.astype(float)).mean())

    details: Dict[str, Union[float, int, str]] = {
        "iou": result.iou,
        "intersection": result.intersection,
        "union": result.union,
        "pixel_difference": difference,
    }
    if result.warnings:
        details["warning"] = "; ".join(result.warnings)

    return result.iou, details
