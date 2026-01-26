"""Blender-only tests for SliceBasedShapeMatcher geometry handling."""

from __future__ import annotations

import math
import random
import unittest

try:
    from mathutils import Vector
    import bmesh

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

from shape_matching.slice_shape_matcher import SliceProfile, SliceBasedShapeMatcher


@unittest.skipUnless(BLENDER_AVAILABLE, "Requires Blender mathutils")
class TestSliceShapeMatcher(unittest.TestCase):
    def test_circle_profile_area_perimeter(self) -> None:
        radius = 1.0
        points = []
        for i in range(64):
            theta = (2.0 * math.pi * i) / 64.0
            points.append(
                Vector((radius * math.cos(theta), radius * math.sin(theta), 0.0))
            )

        profile = SliceProfile(points, plane_height=0.0)
        self.assertAlmostEqual(profile.area, math.pi * radius**2, delta=0.2)
        self.assertAlmostEqual(profile.perimeter, 2 * math.pi * radius, delta=0.2)

    def test_non_convex_profile_stays_sane(self) -> None:
        points = [
            Vector((0.0, 0.0, 0.0)),
            Vector((2.0, 0.0, 0.0)),
            Vector((2.0, 2.0, 0.0)),
            Vector((1.0, 1.0, 0.0)),
            Vector((0.0, 2.0, 0.0)),
            Vector((0.0, 0.0, 0.0)),
        ]
        random.seed(42)
        random.shuffle(points)

        profile = SliceProfile(points, plane_height=0.0)
        self.assertGreater(profile.area, 0.0)
        self.assertGreater(profile.perimeter, 0.0)

    def test_num_slices_validation(self) -> None:
        with self.assertRaises(ValueError):
            SliceBasedShapeMatcher(num_slices=0)

    def test_slice_at_plane_cube(self) -> None:
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=2.0)

        matcher = SliceBasedShapeMatcher(num_slices=2)
        points = matcher._slice_at_plane(bm, axis_idx=2, plane_pos=0.0)
        bm.free()

        self.assertEqual(len(points), 4)


if __name__ == "__main__":
    unittest.main()
