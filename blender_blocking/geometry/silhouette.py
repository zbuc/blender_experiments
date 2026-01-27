"""Canonical silhouette extraction utilities (pure Python)."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from geometry.profile_models import BBox2D


def _ensure_uint8(gray: np.ndarray) -> np.ndarray:
    if gray.dtype == np.uint8:
        return gray
    max_val = float(np.max(gray)) if gray.size else 0.0
    if max_val <= 1.0:
        gray = (gray.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def _gray_from_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return _ensure_uint8(image)
    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if image.shape[2] == 4:
        rgb = image[:, :, :3]
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    raise ValueError(f"Unsupported channel count: {image.shape[2]}")


def _auto_invert(gray: np.ndarray) -> bool:
    height, width = gray.shape
    border = np.concatenate(
        [
            gray[0, :],
            gray[-1, :],
            gray[:, 0],
            gray[:, -1],
        ]
    )
    center = gray[
        max(0, height // 4) : max(height // 4 + 1, 3 * height // 4),
        max(0, width // 4) : max(width // 4 + 1, 3 * width // 4),
    ]
    bg_mean = float(np.mean(border)) if border.size else 0.0
    center_mean = float(np.mean(center)) if center.size else bg_mean
    return center_mean < bg_mean


def _kernel_for(size: int) -> Optional[np.ndarray]:
    if size <= 0:
        return None
    k = max(3, int(size))
    if k % 2 == 0:
        k += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def extract_binary_silhouette(
    image: np.ndarray,
    *,
    prefer_alpha: bool = True,
    alpha_threshold: int = 127,
    gray_threshold: Optional[int] = None,
    invert_policy: str = "auto",
    morph_close_px: int = 0,
    morph_open_px: int = 0,
    fill_holes: bool = True,
    largest_component_only: bool = True,
) -> np.ndarray:
    """
    Extract a binary silhouette mask from an image.

    Returns:
        Boolean mask with True for foreground pixels.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if image.ndim == 3 and image.shape[2] == 4 and prefer_alpha:
        alpha = image[:, :, 3]
        if np.ptp(alpha) > 0:
            mask = alpha > alpha_threshold
        else:
            gray = _gray_from_image(image)
            mask = None
    else:
        gray = _gray_from_image(image)
        mask = None

    if mask is None:
        gray = _ensure_uint8(gray)
        if invert_policy == "auto":
            invert = _auto_invert(gray)
        elif invert_policy == "invert":
            invert = True
        elif invert_policy == "no_invert":
            invert = False
        else:
            raise ValueError(f"Unknown invert_policy: {invert_policy}")

        if gray_threshold is None:
            threshold_flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
            _, thresh = cv2.threshold(gray, 0, 255, threshold_flag + cv2.THRESH_OTSU)
        else:
            threshold_flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
            _, thresh = cv2.threshold(gray, gray_threshold, 255, threshold_flag)

        mask_uint8 = thresh
    else:
        mask_uint8 = (mask.astype(np.uint8) * 255).astype(np.uint8)

    close_kernel = _kernel_for(morph_close_px)
    if close_kernel is not None:
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, close_kernel)

    open_kernel = _kernel_for(morph_open_px)
    if open_kernel is not None:
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, open_kernel)

    if largest_component_only:
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return np.zeros(mask_uint8.shape, dtype=bool)
        largest = max(contours, key=cv2.contourArea)
        filtered = np.zeros_like(mask_uint8)
        cv2.drawContours(filtered, [largest], -1, 255, -1)
        mask_uint8 = filtered

    if fill_holes:
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filled = np.zeros_like(mask_uint8)
        if contours:
            cv2.drawContours(filled, contours, -1, 255, -1)
        mask_uint8 = filled

    return mask_uint8 > 0


def bbox_from_mask(mask: np.ndarray) -> BBox2D:
    """Compute a tight bounding box from a boolean mask."""
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("Silhouette mask is empty")
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return BBox2D(x0=x0, y0=y0, x1=x1, y1=y1)
