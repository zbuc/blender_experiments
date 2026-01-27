"""Canonical geometry data contracts for profile-based reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence


@dataclass(frozen=True)
class BBox2D:
    """Axis-aligned 2D bounding box using exclusive max bounds."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        """Width in pixels."""
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        """Height in pixels."""
        return self.y1 - self.y0


@dataclass(frozen=True)
class PixelScale:
    """Pixel-to-world scaling helper."""

    unit_per_px: float

    @staticmethod
    def from_target_height(
        target_height_units: float, silhouette_height_px: int
    ) -> "PixelScale":
        """Create a scale from a desired target height in world units."""
        if silhouette_height_px <= 0:
            raise ValueError("silhouette_height_px must be > 0")
        return PixelScale(unit_per_px=target_height_units / float(silhouette_height_px))


@dataclass(frozen=True)
class VerticalWidthProfilePx:
    """Per-row width measurements extracted from a binary silhouette."""

    heights_t: Sequence[float]
    left_x: Sequence[float]
    right_x: Sequence[float]
    width_px: Sequence[float]
    center_x: Sequence[float]
    valid: Sequence[bool]
    bbox: BBox2D
    source_view: str


@dataclass(frozen=True)
class EllipticalProfileU:
    """Elliptical profile in world units sampled along normalized height."""

    heights_t: Sequence[float]
    rx: Sequence[float]
    ry: Sequence[float]
    world_height: float
    z0: float = 0.0
    cx: Optional[Sequence[float]] = None
    cy: Optional[Sequence[float]] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class EllipticalSlice:
    """Single loft slice in world units."""

    z: float
    rx: float
    ry: float
    cx: Optional[float] = None
    cy: Optional[float] = None
