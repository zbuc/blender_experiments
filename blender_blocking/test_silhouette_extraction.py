"""Tests for canonical silhouette extraction and profiles."""

from __future__ import annotations

import unittest

import numpy as np

from geometry.silhouette import bbox_from_mask, extract_binary_silhouette
from integration.shape_matching.profile_extractor import extract_vertical_profile


class TestSilhouetteExtraction(unittest.TestCase):
    def test_auto_invert_consistency(self) -> None:
        image_dark = np.full((64, 64), 255, dtype=np.uint8)
        image_dark[16:48, 24:40] = 0

        image_light = np.zeros((64, 64), dtype=np.uint8)
        image_light[16:48, 24:40] = 255

        mask_dark = extract_binary_silhouette(image_dark)
        mask_light = extract_binary_silhouette(image_light)

        self.assertTrue(np.array_equal(mask_dark, mask_light))

    def test_rgba_alpha_bbox(self) -> None:
        image = np.zeros((32, 40, 4), dtype=np.uint8)
        image[5:20, 10:30, 3] = 255

        mask = extract_binary_silhouette(image)
        bbox = bbox_from_mask(mask)

        self.assertEqual((bbox.x0, bbox.y0, bbox.x1, bbox.y1), (10, 5, 30, 20))

    def test_profile_padding_invariance(self) -> None:
        base = np.zeros((32, 32), dtype=np.uint8)
        base[8:24, 10:22] = 255

        padded = np.zeros((64, 64), dtype=np.uint8)
        padded[16:32, 26:38] = 255

        profile_base = extract_vertical_profile(
            base, num_samples=9, already_silhouette=True
        )
        profile_padded = extract_vertical_profile(
            padded, num_samples=9, already_silhouette=True
        )

        radii_base = np.array([r for _, r in profile_base])
        radii_padded = np.array([r for _, r in profile_padded])

        self.assertTrue(np.allclose(radii_base, radii_padded, atol=1e-6))

    def test_empty_silhouette_raises(self) -> None:
        empty = np.zeros((16, 16), dtype=np.uint8)
        with self.assertRaises(ValueError):
            extract_vertical_profile(empty, num_samples=5, already_silhouette=True)

    def test_profile_constant_width(self) -> None:
        base = np.zeros((20, 20), dtype=np.uint8)
        base[5:15, 7:13] = 255

        profile = extract_vertical_profile(base, num_samples=5, already_silhouette=True)
        radii = np.array([r for _, r in profile])
        self.assertTrue(np.allclose(radii, 1.0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
