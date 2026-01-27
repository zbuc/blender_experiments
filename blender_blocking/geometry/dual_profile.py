"""Dual-profile extraction utilities for silhouette-based reconstruction."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import median_filter

from geometry.profile_models import (
    BBox2D,
    EllipticalProfileU,
    PixelScale,
    VerticalWidthProfilePx,
)
from geometry.silhouette import bbox_from_mask


def _sample_t_values(num_samples: int, sample_policy: str) -> np.ndarray:
    if num_samples <= 0:
        raise ValueError("num_samples must be >= 1")
    if num_samples == 1:
        t_values = np.array([0.5], dtype=np.float32)
    elif sample_policy == "endpoints":
        t_values = np.linspace(0.0, 1.0, num_samples, dtype=np.float32)
    elif sample_policy == "cell_centers":
        t_values = (np.arange(num_samples, dtype=np.float32) + 0.5) / float(num_samples)
    else:
        raise ValueError(f"Unknown sample_policy: {sample_policy}")

    return t_values


def _sample_rows(height: int, t_values: np.ndarray) -> np.ndarray:
    rows = (height - 1) - (t_values * (height - 1))
    return rows


def _fill_missing(values: np.ndarray, valid: np.ndarray, strategy: str) -> np.ndarray:
    if valid.all():
        return values

    if not valid.any():
        if strategy == "constant":
            return np.zeros_like(values)
        raise ValueError("No valid samples available for interpolation")

    indices = np.arange(values.size)
    valid_indices = indices[valid]
    valid_values = values[valid]

    if strategy == "interp_linear":
        filled = np.interp(indices, valid_indices, valid_values)
        return filled

    if strategy == "interp_nearest":
        positions = np.searchsorted(valid_indices, indices)
        left = valid_indices[np.clip(positions - 1, 0, valid_indices.size - 1)]
        right = valid_indices[np.clip(positions, 0, valid_indices.size - 1)]
        left_dist = np.abs(indices - left)
        right_dist = np.abs(right - indices)
        nearest = np.where(right_dist < left_dist, right, left)
        return values[nearest]

    if strategy == "constant":
        filled = values.copy()
        filled[~valid] = 0.0
        return filled

    raise ValueError(f"Unknown fill_strategy: {strategy}")


def extract_vertical_width_profile_px(
    mask: np.ndarray,
    *,
    bbox: Optional[BBox2D] = None,
    num_samples: int = 100,
    sample_policy: str = "endpoints",
    fill_strategy: str = "interp_linear",
    smoothing_window: int = 3,
) -> VerticalWidthProfilePx:
    """Extract a vertical width profile from a binary mask."""
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    mask_bool = mask.astype(bool)

    if bbox is None:
        bbox = bbox_from_mask(mask_bool)

    cropped = mask_bool[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1]
    height, width = cropped.shape

    t_values = _sample_t_values(num_samples, sample_policy)
    row_positions = _sample_rows(height, t_values)
    rows = np.clip(np.round(row_positions).astype(int), 0, height - 1)

    row_has = cropped.any(axis=1)
    left_idx = np.argmax(cropped, axis=1)
    right_idx = width - 1 - np.argmax(cropped[:, ::-1], axis=1)

    sample_has = row_has[rows]
    heights_t = t_values.astype(np.float32)
    left_x = np.where(sample_has, bbox.x0 + left_idx[rows], np.nan).astype(np.float32)
    right_x = np.where(sample_has, bbox.x0 + right_idx[rows], np.nan).astype(np.float32)
    valid = sample_has.copy()

    left_x = _fill_missing(left_x, valid, fill_strategy)
    right_x = _fill_missing(right_x, valid, fill_strategy)

    center_x = (left_x + right_x) / 2.0
    width_px = right_x - left_x + 1.0
    width_px = np.maximum(width_px, 0.0)

    if smoothing_window > 1:
        width_px = median_filter(width_px, size=int(smoothing_window))
        half = (width_px - 1.0) / 2.0
        left_x = center_x - half
        right_x = center_x + half
        width_px = right_x - left_x + 1.0
    return VerticalWidthProfilePx(
        heights_t=heights_t,
        left_x=left_x,
        right_x=right_x,
        width_px=width_px,
        center_x=center_x,
        valid=valid,
        bbox=bbox,
        source_view="",
    )


def build_elliptical_profile_from_views(
    front_mask: Optional[np.ndarray],
    side_mask: Optional[np.ndarray],
    scale: PixelScale,
    *,
    num_samples: int = 100,
    z0: float = 0.0,
    height_strategy: str = "front",
    fallback_policy: str = "circular",
    min_radius_u: float = 0.0,
    sample_policy: str = "endpoints",
    fill_strategy: str = "interp_linear",
    smoothing_window: int = 3,
    enable_offsets: bool = False,
) -> EllipticalProfileU:
    """Build an elliptical profile from front/side silhouettes."""
    if front_mask is None and side_mask is None:
        raise ValueError("At least one of front_mask or side_mask must be provided")

    front_profile = None
    side_profile = None

    if front_mask is not None:
        front_profile = extract_vertical_width_profile_px(
            front_mask,
            num_samples=num_samples,
            sample_policy=sample_policy,
            fill_strategy=fill_strategy,
            smoothing_window=smoothing_window,
        )
        front_profile = VerticalWidthProfilePx(
            **{**front_profile.__dict__, "source_view": "front"}
        )

    if side_mask is not None:
        side_profile = extract_vertical_width_profile_px(
            side_mask,
            num_samples=num_samples,
            sample_policy=sample_policy,
            fill_strategy=fill_strategy,
            smoothing_window=smoothing_window,
        )
        side_profile = VerticalWidthProfilePx(
            **{**side_profile.__dict__, "source_view": "side"}
        )

    heights_t = None
    rx = None
    ry = None
    cx = None
    cy = None

    if front_profile is not None:
        heights_t = np.asarray(front_profile.heights_t, dtype=np.float32)
        rx = (
            np.asarray(front_profile.width_px, dtype=np.float32)
            * scale.unit_per_px
            / 2.0
        )

    if side_profile is not None:
        if heights_t is None:
            heights_t = np.asarray(side_profile.heights_t, dtype=np.float32)
        ry = (
            np.asarray(side_profile.width_px, dtype=np.float32)
            * scale.unit_per_px
            / 2.0
        )

    if front_profile is None or side_profile is None:
        if fallback_policy == "circular":
            if rx is None and ry is not None:
                rx = ry.copy()
            if ry is None and rx is not None:
                ry = rx.copy()
        elif fallback_policy == "error":
            raise ValueError(
                "Both front and side views are required for this fallback policy"
            )
        else:
            raise ValueError(f"Unknown fallback_policy: {fallback_policy}")

    if rx is None or ry is None:
        raise ValueError("Failed to construct radii for profile")

    rx = np.maximum(rx, min_radius_u)
    ry = np.maximum(ry, min_radius_u)

    front_height_px = front_profile.bbox.h if front_profile is not None else None
    side_height_px = side_profile.bbox.h if side_profile is not None else None

    if height_strategy == "front" and front_height_px is not None:
        world_height = front_height_px * scale.unit_per_px
    elif height_strategy == "side" and side_height_px is not None:
        world_height = side_height_px * scale.unit_per_px
    elif height_strategy == "max":
        world_height = max(
            (front_height_px or 0) * scale.unit_per_px,
            (side_height_px or 0) * scale.unit_per_px,
        )
    elif height_strategy == "mean":
        heights = [h for h in [front_height_px, side_height_px] if h is not None]
        world_height = (sum(heights) / len(heights)) * scale.unit_per_px
    else:
        height_source = (
            front_height_px if front_height_px is not None else side_height_px
        )
        world_height = (height_source or 0) * scale.unit_per_px

    if world_height <= 0:
        raise ValueError("world_height must be positive")

    if enable_offsets:
        if front_profile is not None:
            center = (front_profile.bbox.x0 + front_profile.bbox.x1) / 2.0
            cx = (np.asarray(front_profile.center_x) - center) * scale.unit_per_px
        if side_profile is not None:
            center = (side_profile.bbox.x0 + side_profile.bbox.x1) / 2.0
            cy = (np.asarray(side_profile.center_x) - center) * scale.unit_per_px

    return EllipticalProfileU(
        heights_t=heights_t,
        rx=rx,
        ry=ry,
        world_height=world_height,
        z0=z0,
        cx=cx,
        cy=cy,
        meta=None,
    )
