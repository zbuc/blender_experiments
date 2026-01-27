"""Tests for dual-profile width sampling."""

from __future__ import annotations

import unittest

import numpy as np

from geometry.dual_profile import extract_vertical_width_profile_px


class TestProfileSampling(unittest.TestCase):
    def test_rectangle_mask_constant_width(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[4:16, 6:12] = 255

        profile = extract_vertical_width_profile_px(
            mask,
            num_samples=5,
            sample_policy="endpoints",
            fill_strategy="interp_linear",
            smoothing_window=1,
        )

        widths = np.round(profile.width_px, 6)
        self.assertTrue(np.allclose(widths, widths[0]))
        self.assertAlmostEqual(widths[0], 6.0, places=6)

    def test_interp_nearest_fill(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[0, 2:8] = 255
        mask[-1, 4:6] = 255

        profile = extract_vertical_width_profile_px(
            mask,
            num_samples=5,
            sample_policy="endpoints",
            fill_strategy="interp_nearest",
            smoothing_window=1,
        )

        widths = np.round(profile.width_px, 6)
        expected = np.array([2.0, 2.0, 2.0, 6.0, 6.0])
        self.assertTrue(np.allclose(widths, expected))


if __name__ == "__main__":
    unittest.main()
