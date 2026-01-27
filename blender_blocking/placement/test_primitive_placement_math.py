"""Math-only tests for SliceAnalyzer (pure Python)."""

from __future__ import annotations

import unittest

from placement.primitive_placement import SliceAnalyzer


class TestPrimitivePlacementMath(unittest.TestCase):
    def test_num_slices_one_uses_mid_height(self) -> None:
        analyzer = SliceAnalyzer(
            bounds_min=(0, 0, 0), bounds_max=(2, 2, 4), num_slices=1
        )
        slices = analyzer.get_all_slice_data()
        self.assertEqual(len(slices), 1)
        self.assertAlmostEqual(slices[0]["center"].z, 2.0, places=6)

    def test_overlap_ratio_sets_scale_z(self) -> None:
        analyzer = SliceAnalyzer(
            bounds_min=(0, 0, 0),
            bounds_max=(2, 2, 4),
            num_slices=4,
            z_overlap_ratio=2.0,
        )
        slices = analyzer.get_all_slice_data()
        self.assertAlmostEqual(slices[0]["scale"].z, 2.0, places=6)

    def test_min_radius_ratio(self) -> None:
        analyzer = SliceAnalyzer(
            bounds_min=(0, 0, 0),
            bounds_max=(4, 2, 1),
            num_slices=2,
            min_radius_ratio=0.1,
        )
        self.assertAlmostEqual(analyzer.min_radius, 0.4, places=6)

    def test_invalid_num_slices(self) -> None:
        with self.assertRaises(ValueError):
            SliceAnalyzer(bounds_min=(0, 0, 0), bounds_max=(1, 1, 1), num_slices=0)

    def test_profile_interpolation(self) -> None:
        analyzer = SliceAnalyzer(
            bounds_min=(0, 0, 0),
            bounds_max=(2, 2, 2),
            num_slices=2,
            vertical_profile=[(0.0, 0.2), (0.5, 0.6), (1.0, 1.0)],
        )
        self.assertAlmostEqual(analyzer._interpolate_profile(0.0), 0.2, places=6)
        self.assertAlmostEqual(analyzer._interpolate_profile(0.5), 0.6, places=6)
        self.assertAlmostEqual(analyzer._interpolate_profile(1.0), 1.0, places=6)
        self.assertAlmostEqual(analyzer._interpolate_profile(0.25), 0.4, places=6)


if __name__ == "__main__":
    unittest.main()
