"""Tests for elliptical slice sampling policies."""

from __future__ import annotations

import unittest

import numpy as np

from geometry.profile_models import EllipticalProfileU
from geometry.slicing import sample_elliptical_slices


class TestSliceSampling(unittest.TestCase):
    def test_endpoints_vs_cell_centers(self) -> None:
        profile = EllipticalProfileU(
            heights_t=np.array([0.0, 1.0], dtype=np.float32),
            rx=np.array([1.0, 1.0], dtype=np.float32),
            ry=np.array([2.0, 2.0], dtype=np.float32),
            world_height=10.0,
            z0=0.0,
        )

        endpoints = sample_elliptical_slices(
            profile, num_slices=2, sampling="endpoints"
        )
        centers = sample_elliptical_slices(
            profile, num_slices=2, sampling="cell_centers"
        )

        self.assertEqual([s.z for s in endpoints], [0.0, 10.0])
        self.assertEqual([s.z for s in centers], [2.5, 7.5])


if __name__ == "__main__":
    unittest.main()
