"""Tests for contour analyzer utilities (pure Python)."""

from __future__ import annotations

import unittest

import numpy as np

from integration.shape_matching.contour_analyzer import analyze_shape, find_contours


class TestContourAnalyzer(unittest.TestCase):
    def test_find_contours_empty(self) -> None:
        empty = np.zeros((0, 0), dtype=np.uint8)
        contours = find_contours(empty)
        self.assertEqual(contours, [])

    def test_find_contours_rgb(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[8:24, 10:22, :] = 255
        contours = find_contours(image)
        self.assertTrue(len(contours) >= 1)

    def test_analyze_shape_basic(self) -> None:
        contour = np.array([[[0, 0]], [[10, 0]], [[10, 5]], [[0, 5]]], dtype=np.int32)
        stats = analyze_shape(contour)
        self.assertGreater(stats["area"], 0)
        self.assertGreater(stats["perimeter"], 0)
        self.assertGreater(stats["aspect_ratio"], 0)


if __name__ == "__main__":
    unittest.main()
