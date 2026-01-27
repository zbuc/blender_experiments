"""Tests for canonical silhouette IoU utilities."""

from __future__ import annotations

import unittest

import numpy as np

from validation.silhouette_iou import (
    canonicalize_mask,
    canonicalize_mask_cached,
    compute_mask_iou,
    get_canonicalize_cache_stats,
    mask_from_image_array,
    reset_canonicalize_cache,
)


class TestSilhouetteIoU(unittest.TestCase):
    def test_resolution_invariance(self) -> None:
        mask_small = np.zeros((32, 32), dtype=bool)
        mask_small[8:24, 12:20] = True

        mask_large = np.zeros((64, 64), dtype=bool)
        mask_large[16:48, 24:40] = True

        canon_small = canonicalize_mask(mask_small, output_size=64, padding_frac=0.1)
        canon_large = canonicalize_mask(mask_large, output_size=64, padding_frac=0.1)

        result = compute_mask_iou(canon_small, canon_large)
        self.assertGreater(result.iou, 0.9)

    def test_padding_invariance(self) -> None:
        base = np.zeros((24, 24), dtype=bool)
        base[6:18, 8:16] = True

        padded = np.zeros((48, 48), dtype=bool)
        padded[18:30, 20:28] = True

        canon_base = canonicalize_mask(base, output_size=64, padding_frac=0.1)
        canon_padded = canonicalize_mask(padded, output_size=64, padding_frac=0.1)

        result = compute_mask_iou(canon_base, canon_padded)
        self.assertAlmostEqual(result.iou, 1.0, places=6)

    def test_bottom_center_anchor(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[8:14, 8:12] = True

        canon_bottom = canonicalize_mask(
            mask, output_size=32, padding_frac=0.0, anchor="bottom_center"
        )
        canon_center = canonicalize_mask(
            mask, output_size=32, padding_frac=0.0, anchor="center"
        )

        bottom_max = np.where(canon_bottom)[0].max()
        center_max = np.where(canon_center)[0].max()

        self.assertEqual(bottom_max, 31)
        self.assertLessEqual(center_max, bottom_max)

    def test_empty_masks_warning(self) -> None:
        empty = np.zeros((16, 16), dtype=bool)
        result = compute_mask_iou(empty, empty)
        self.assertEqual(result.iou, 0.0)
        self.assertTrue(result.warnings)

    def test_mask_from_image_array_prefers_alpha(self) -> None:
        image = np.zeros((8, 8, 4), dtype=np.uint8)
        image[:, :, :3] = 255
        image[2:6, 3:5, 3] = 255
        mask = mask_from_image_array(image)
        self.assertTrue(mask[3, 4])
        self.assertFalse(mask[0, 0])

    def test_canonicalize_invalid_output(self) -> None:
        mask = np.zeros((4, 4), dtype=bool)
        with self.assertRaises(ValueError):
            canonicalize_mask(mask, output_size=0)

    def test_canonicalize_cache_hits(self) -> None:
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:12, 5:10] = True

        reset_canonicalize_cache()
        first = canonicalize_mask_cached(mask, output_size=32, padding_frac=0.1)
        stats_after_first = get_canonicalize_cache_stats()

        second = canonicalize_mask_cached(mask, output_size=32, padding_frac=0.1)
        stats_after_second = get_canonicalize_cache_stats()

        np.testing.assert_array_equal(first, second)
        self.assertEqual(stats_after_first["hits"], 0)
        self.assertEqual(stats_after_first["misses"], 1)
        self.assertEqual(stats_after_second["hits"], 1)


if __name__ == "__main__":
    unittest.main()
