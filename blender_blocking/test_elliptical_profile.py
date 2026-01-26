"""Tests for elliptical profile construction."""

from __future__ import annotations

import unittest

import numpy as np

from geometry.dual_profile import build_elliptical_profile_from_views
from geometry.profile_models import PixelScale


class TestEllipticalProfile(unittest.TestCase):
    def test_front_side_map_to_rx_ry(self) -> None:
        front = np.zeros((10, 10), dtype=np.uint8)
        front[:, 3:7] = 255  # width 4

        side = np.zeros((10, 10), dtype=np.uint8)
        side[:, 4:6] = 255  # width 2

        profile = build_elliptical_profile_from_views(
            front,
            side,
            PixelScale(unit_per_px=0.1),
            num_samples=4,
            height_strategy="front",
        )

        self.assertTrue(np.allclose(profile.rx, 0.2))
        self.assertTrue(np.allclose(profile.ry, 0.1))

    def test_single_view_fallback_circular(self) -> None:
        front = np.zeros((10, 10), dtype=np.uint8)
        front[:, 2:8] = 255  # width 6

        profile = build_elliptical_profile_from_views(
            front,
            None,
            PixelScale(unit_per_px=0.5),
            num_samples=3,
            fallback_policy="circular",
        )

        self.assertTrue(np.allclose(profile.rx, profile.ry))


if __name__ == "__main__":
    unittest.main()
