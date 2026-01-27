"""Tests for manifest schema integrity (pure Python)."""

from __future__ import annotations

import unittest

from utils.generation_context import GenerationContext
from utils.manifest import build_manifest


class TestManifestSchema(unittest.TestCase):
    def test_manifest_contains_required_keys(self) -> None:
        ctx = GenerationContext()
        manifest = build_manifest(ctx, outputs={}, warnings=[], errors=[])
        for key in (
            "manifest_version",
            "run_id",
            "created_utc",
            "context",
            "stages",
            "outputs",
            "warnings",
            "errors",
        ):
            self.assertIn(key, manifest)

    def test_run_id_consistency(self) -> None:
        ctx = GenerationContext(seed=42)
        manifest = build_manifest(ctx)
        self.assertEqual(manifest["run_id"], ctx.run_id)


if __name__ == "__main__":
    unittest.main()
