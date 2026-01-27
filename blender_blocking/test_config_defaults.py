"""Tests for config defaults and validation (pure Python)."""

from __future__ import annotations

import unittest

from config import BlockingConfig


class TestConfigDefaults(unittest.TestCase):
    def test_defaults_match_legacy(self) -> None:
        cfg = BlockingConfig()
        self.assertEqual(cfg.reconstruction.unit_scale, 0.01)
        self.assertEqual(cfg.reconstruction.num_slices, 10)
        self.assertEqual(cfg.mesh_join.mode, "boolean")
        self.assertEqual(cfg.render_silhouette.resolution, (512, 512))
        self.assertEqual(cfg.canonicalize.output_size, 256)

    def test_validate(self) -> None:
        cfg = BlockingConfig()
        cfg.validate()


if __name__ == "__main__":
    unittest.main()
