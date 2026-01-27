"""Utilities for contour normalization, resampling, and scaling."""

from __future__ import annotations

import numpy as np
import cv2
from typing import Tuple

from geometry.contour_models import ContourTemplate


def detect_flat_object(
    max_rx: float,
    max_ry: float,
    world_height: float,
    flatness_threshold: float = 0.2,
) -> bool:
    """
    Detect if object is flat (disk-like) vs. volumetric (box-like).

    Flat objects have small height relative to XY extent, meaning their
    front/side projections don't match their true cross-sections.

    Args:
        max_rx: Maximum X radius from front profile (world units)
        max_ry: Maximum Y radius from side profile (world units)
        world_height: Object height from profile (world units)
        flatness_threshold: Height/XY ratio threshold for flat detection

    Returns:
        True if object is flat (disk-like), False if volumetric (box-like)

    Examples:
        - Cube (2×2×2): height=2, max_xy=1, ratio=2.0 → False (volumetric)
        - Flat star (4×4×0.2): height=0.2, max_xy=2, ratio=0.1 → True (flat)
        - Bottle (1×1×3): height=3, max_xy=0.5, ratio=6.0 → False (volumetric)
    """
    max_xy = max(max_rx, max_ry)

    # Avoid division by zero
    if max_xy == 0:
        return False

    # Compute height to XY ratio
    height_to_xy_ratio = world_height / max_xy

    # Object is flat if height is much smaller than XY extent
    return height_to_xy_ratio < flatness_threshold


def compute_contour_native_scale(
    bbox_w_px: int,
    bbox_h_px: int,
    unit_scale: float,
    z_scale_factor: float = 1.0,
) -> Tuple[float, float]:
    """
    Compute scale factors based on contour's native pixel dimensions.

    Used for flat objects where top-view defines true XY extent.

    Args:
        bbox_w_px: Contour bounding box width in pixels
        bbox_h_px: Contour bounding box height in pixels
        unit_scale: Pixel to world unit conversion (e.g., 0.01)
        z_scale_factor: Height-based scaling factor (0-1 range)

    Returns:
        (scale_x, scale_y) in world units (half-widths for compatibility)

    Example:
        bbox = (0, 0, 400, 400)  # 400×400px star
        unit_scale = 0.01
        z_scale_factor = 0.8  # 80% of max height

        → scale_x = (400 * 0.01 * 0.8) / 2 = 1.6
        → scale_y = (400 * 0.01 * 0.8) / 2 = 1.6

        Contour normalized to [-0.5, 0.5] scaled by 2*1.6 = 3.2 → 3.2×3.2 world units
    """
    # Convert pixel dimensions to world units
    w_world = bbox_w_px * unit_scale
    h_world = bbox_h_px * unit_scale

    # Apply height-based scaling uniformly
    w_scaled = w_world * z_scale_factor
    h_scaled = h_world * z_scale_factor

    # Return as half-widths (radii) for compatibility with existing code
    # (contour scaling multiplies by 2, so we need to divide by 2 here)
    return w_scaled / 2.0, h_scaled / 2.0


def normalize_contour(
    contour: np.ndarray,
    bbox: Tuple[int, int, int, int] = None,
) -> np.ndarray:
    """
    Normalize contour to unit square [-0.5, 0.5] centered at origin.

    Args:
        contour: OpenCV contour (N, 1, 2) or (N, 2)
        bbox: Optional pre-computed bounding box (x, y, w, h)

    Returns:
        Normalized contour (N, 2) in range [-0.5, 0.5]
    """
    # Reshape to (N, 2) if needed
    if contour.ndim == 3 and contour.shape[1] == 1:
        contour = contour.squeeze(1)

    contour = contour.astype(np.float32)

    # Get bounding box
    if bbox is None:
        x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
    else:
        x, y, w, h = bbox

    # Center at origin
    cx = x + w / 2.0
    cy = y + h / 2.0
    centered = contour - np.array([[cx, cy]], dtype=np.float32)

    # Scale to unit square [-0.5, 0.5]
    scale = max(w, h)
    if scale > 0:
        normalized = centered / scale
    else:
        normalized = centered

    return normalized


def resample_contour_uniform(
    contour: np.ndarray,
    num_points: int,
) -> np.ndarray:
    """
    Resample contour to exactly num_points uniformly spaced by arc length.

    This ensures consistent vertex correspondence between loft slices.

    Args:
        contour: Input contour (N, 2)
        num_points: Target number of vertices

    Returns:
        Resampled contour (num_points, 2)
    """
    if len(contour) < 2:
        raise ValueError("Contour must have at least 2 points")

    # Ensure closed contour for arc length calculation
    contour_closed = np.vstack([contour, contour[0:1]])

    # Calculate cumulative arc length
    distances = np.sqrt(
        np.sum(np.diff(contour_closed, axis=0)**2, axis=1)
    )
    cumulative = np.concatenate([[0], np.cumsum(distances)])
    perimeter = cumulative[-1]

    if perimeter == 0:
        # Degenerate contour, return duplicated first point
        return np.tile(contour[0:1], (num_points, 1))

    # Target arc lengths for new points
    target_lengths = np.linspace(0, perimeter, num_points, endpoint=False)

    # Interpolate x and y coordinates
    resampled = np.zeros((num_points, 2), dtype=np.float32)
    resampled[:, 0] = np.interp(
        target_lengths,
        cumulative,
        contour_closed[:, 0]
    )
    resampled[:, 1] = np.interp(
        target_lengths,
        cumulative,
        contour_closed[:, 1]
    )

    return resampled


def create_contour_template(
    contour: np.ndarray,
    num_vertices: int,
    source_view: str = "top",
) -> ContourTemplate:
    """
    Create a normalized, resampled contour template.

    Args:
        contour: Raw contour from image processing (N, 1, 2) or (N, 2)
        num_vertices: Number of vertices for resampling
        source_view: View name (for metadata)

    Returns:
        ContourTemplate ready for loft mesh generation
    """
    # Get bounding box for metadata
    if contour.ndim == 3:
        contour_2d = contour.squeeze(1)
    else:
        contour_2d = contour

    x, y, w, h = cv2.boundingRect(contour_2d.astype(np.int32))

    # Normalize and resample
    normalized = normalize_contour(contour, bbox=(x, y, w, h))
    resampled = resample_contour_uniform(normalized, num_vertices)

    return ContourTemplate(
        points=resampled,
        num_vertices=num_vertices,
        source_view=source_view,
        original_bbox=(x, y, w, h),
    )


def scale_contour_2d(
    normalized_contour: np.ndarray,
    scale_x: float,
    scale_y: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
    original_bbox: Tuple[int, int, int, int] = None,
    unit_scale: float = None,
) -> np.ndarray:
    """
    Scale normalized contour by profile factors.

    Args:
        normalized_contour: Contour in unit square [-0.5, 0.5]
        scale_x: Target width (rx from front view, represents half-width)
        scale_y: Target depth (ry from side view, represents half-width)
        center_x: X offset in world units
        center_y: Y offset in world units
        original_bbox: Original pixel bounding box (x, y, w, h) before normalization
        unit_scale: Pixel to world unit conversion (e.g., 0.01)

    Returns:
        Scaled contour in world coordinates (N, 2)
    """
    scaled = normalized_contour.copy()

    # Scale from [-0.5, 0.5] to actual dimensions
    # Multiply by 2 because rx/ry are radii and we need diameter
    scaled[:, 0] = scaled[:, 0] * scale_x * 2.0 + center_x
    scaled[:, 1] = scaled[:, 1] * scale_y * 2.0 + center_y

    return scaled
