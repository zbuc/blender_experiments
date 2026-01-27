"""Tests for profile combination utilities."""

from __future__ import annotations

import unittest

import numpy as np

from integration.shape_matching.mesh_profile_extractor import combine_profiles


class TestProfileCombination(unittest.TestCase):
    def test_combine_profiles_methods(self) -> None:
        profiles = [
            [(0.0, 1.0), (1.0, 2.0)],
            [(0.0, 3.0), (1.0, 4.0)],
        ]

        mean_profile = combine_profiles(profiles, method="mean")
        median_profile = combine_profiles(profiles, method="median")
        min_profile = combine_profiles(profiles, method="min")
        max_profile = combine_profiles(profiles, method="max")

        self.assertEqual([h for h, _ in mean_profile], [0.0, 1.0])
        self.assertTrue(np.allclose([r for _, r in mean_profile], [2.0, 3.0]))
        self.assertTrue(np.allclose([r for _, r in median_profile], [2.0, 3.0]))
        self.assertTrue(np.allclose([r for _, r in min_profile], [1.0, 2.0]))
        self.assertTrue(np.allclose([r for _, r in max_profile], [3.0, 4.0]))

    def test_combine_profiles_length_mismatch(self) -> None:
        profiles = [
            [(0.0, 1.0), (1.0, 2.0)],
            [(0.0, 3.0)],
        ]
        with self.assertRaises(ValueError):
            combine_profiles(profiles, method="median")


if __name__ == "__main__":
    unittest.main()
