"""Tests for GenerationContext utilities (pure Python)."""

from __future__ import annotations

import json
import random
import unittest

from config import BlockingConfig
from utils.generation_context import GenerationContext


class TestGenerationContext(unittest.TestCase):
    def test_apply_seed_determinism(self) -> None:
        ctx1 = GenerationContext(seed=123)
        ctx2 = GenerationContext(seed=123)

        ctx1.apply_seed()
        values1 = [random.random() for _ in range(3)]

        ctx2.apply_seed()
        values2 = [random.random() for _ in range(3)]

        self.assertEqual(values1, values2)

        try:
            import numpy as np
        except ImportError:
            return

        ctx1.apply_seed()
        arr1 = np.random.rand(3)
        ctx2.apply_seed()
        arr2 = np.random.rand(3)
        self.assertTrue(np.array_equal(arr1, arr2))

    def test_to_dict_json_safe(self) -> None:
        ctx = GenerationContext(seed=7)
        ctx.config = BlockingConfig()
        payload = ctx.to_dict()
        json.dumps(payload)

    def test_time_block_records(self) -> None:
        ctx = GenerationContext()
        with ctx.time_block("stage_a"):
            pass
        self.assertEqual(len(ctx.stages), 1)
        self.assertEqual(ctx.stages[0].stage, "stage_a")
        self.assertGreaterEqual(ctx.stages[0].elapsed_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
