"""Tests for silhouette and contour matching utilities (pure Python)."""

from __future__ import annotations

import unittest

import numpy as np

from integration.shape_matching.shape_matcher import compare_silhouettes, match_shapes


class TestShapeMatcher(unittest.TestCase):
    def test_match_shapes_rejects_invalid(self) -> None:
        contour = np.array([[[0, 0]], [[1, 1]]], dtype=np.int32)
        with self.assertRaises(ValueError):
            match_shapes(contour, contour)
        with self.assertRaises(ValueError):
            match_shapes(None, contour)  # type: ignore[arg-type]

    def test_match_shapes_basic(self) -> None:
        contour = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32)
        score = match_shapes(contour, contour)
        self.assertGreaterEqual(score, 0.0)

    def test_compare_silhouettes_identical(self) -> None:
        image = np.full((32, 32), 255, dtype=np.uint8)
        image[8:24, 10:22] = 0
        iou, details = compare_silhouettes(image, image, output_size=32)
        self.assertAlmostEqual(iou, 1.0, places=6)
        self.assertIn("iou", details)

    def test_compare_silhouettes_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            compare_silhouettes(np.array([]), np.array([]))


if __name__ == "__main__":
    unittest.main()
