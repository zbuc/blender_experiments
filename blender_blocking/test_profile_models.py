"""Tests for geometry profile model contracts (pure Python)."""

from __future__ import annotations

import unittest

from geometry.profile_models import (
    BBox2D,
    EllipticalProfileU,
    EllipticalSlice,
    PixelScale,
    VerticalWidthProfilePx,
)


class TestProfileModels(unittest.TestCase):
    def test_bbox_dimensions(self) -> None:
        bbox = BBox2D(x0=2, y0=3, x1=7, y1=11)
        self.assertEqual(bbox.w, 5)
        self.assertEqual(bbox.h, 8)

    def test_pixel_scale_from_target_height(self) -> None:
        scale = PixelScale.from_target_height(2.0, 100)
        self.assertAlmostEqual(scale.unit_per_px, 0.02)

    def test_pixel_scale_invalid_height(self) -> None:
        with self.assertRaises(ValueError):
            PixelScale.from_target_height(1.0, 0)

    def test_profile_construction(self) -> None:
        bbox = BBox2D(x0=0, y0=0, x1=10, y1=20)
        profile = VerticalWidthProfilePx(
            heights_t=[0.0, 0.5, 1.0],
            left_x=[1.0, 2.0, 3.0],
            right_x=[5.0, 6.0, 7.0],
            width_px=[4.0, 4.0, 4.0],
            center_x=[3.0, 4.0, 5.0],
            valid=[True, True, True],
            bbox=bbox,
            source_view="front",
        )
        self.assertEqual(profile.bbox.w, 10)
        self.assertEqual(profile.source_view, "front")

        elliptical = EllipticalProfileU(
            heights_t=[0.0, 1.0],
            rx=[0.5, 0.5],
            ry=[0.25, 0.25],
            world_height=2.0,
            z0=-1.0,
            meta={"source": "test"},
        )
        self.assertEqual(elliptical.world_height, 2.0)
        self.assertEqual(elliptical.meta["source"], "test")

        slice_u = EllipticalSlice(z=0.0, rx=0.5, ry=0.25, cx=0.1, cy=-0.1)
        self.assertAlmostEqual(slice_u.rx, 0.5)


if __name__ == "__main__":
    unittest.main()
