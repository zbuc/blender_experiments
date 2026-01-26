"""Tests for slice shape matching metrics (pure Python)."""

from __future__ import annotations

import unittest

import numpy as np

from shape_matching.slice_shape_matcher import SliceBasedShapeMatcher


class TestSliceShapeMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.matcher = SliceBasedShapeMatcher(num_slices=2)

    def test_normalize_features_range(self) -> None:
        features = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],
            ]
        )
        normalized = self.matcher._normalize_features(features)
        expected = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        np.testing.assert_allclose(normalized, expected)

    def test_cosine_similarity_identity(self) -> None:
        features = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        sim = self.matcher._cosine_similarity(features, features)
        self.assertAlmostEqual(sim, 1.0, places=6)

    def test_correlation_constant(self) -> None:
        arr1 = np.array([1.0, 1.0, 1.0])
        arr2 = np.array([1.0, 2.0, 3.0])
        corr = self.matcher._correlation(arr1, arr2)
        self.assertEqual(corr, 0.0)


if __name__ == "__main__":
    unittest.main()
