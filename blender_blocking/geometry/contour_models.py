"""Data models for contour-based loft reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class ContourTemplate:
    """Normalized 2D contour template for loft cross-sections.

    Contour points are normalized to unit square [-0.5, 0.5] with
    consistent vertex count for ring bridging.
    """

    points: np.ndarray  # (N, 2) array of normalized (x, y) coordinates
    num_vertices: int
    source_view: str = "top"
    original_bbox: Optional[tuple] = None  # (x, y, w, h) in pixels

    def __post_init__(self):
        if self.points.shape[1] != 2:
            raise ValueError("ContourTemplate points must be (N, 2)")
        if len(self.points) != self.num_vertices:
            raise ValueError(
                f"Point count {len(self.points)} != num_vertices {self.num_vertices}"
            )


@dataclass(frozen=True)
class ContourSlice:
    """Single loft slice using contour template with scale factors.

    The contour template is scaled by (scale_x, scale_y) at height z.
    Scaling mode determines whether profile or native contour dimensions are used.
    """

    z: float
    scale_x: float  # Width scale factor (from front view profile)
    scale_y: float  # Depth scale factor (from side view profile)
    cx: float = 0.0  # Center X offset
    cy: float = 0.0  # Center Y offset
    scale_mode: str = "profile"  # "profile" or "contour_native"
    z_scale_factor: float = 1.0  # For height-based uniform scaling

    def __post_init__(self):
        if self.scale_x < 0 or self.scale_y < 0:
            raise ValueError("Scale factors must be non-negative")
        if self.scale_mode not in ("profile", "contour_native"):
            raise ValueError(f"Invalid scale_mode: {self.scale_mode}")
        if not (0.0 <= self.z_scale_factor <= 1.0):
            raise ValueError(f"z_scale_factor must be in [0, 1], got {self.z_scale_factor}")
