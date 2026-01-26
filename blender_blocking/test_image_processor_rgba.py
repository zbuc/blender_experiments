"""Tests for RGBA-safe image processing."""

from __future__ import annotations

import unittest

import numpy as np

from integration.image_processing.image_processor import process_image


class TestImageProcessorRGBA(unittest.TestCase):
    def test_process_rgba_uses_alpha(self) -> None:
        image = np.zeros((32, 32, 4), dtype=np.uint8)
        image[:, :, 3] = 0
        image[8:24, 10:22, 3] = 255

        processed = process_image(image, extract_edges_flag=True, normalize_flag=True)

        self.assertEqual(processed.dtype, np.uint8)
        self.assertEqual(processed.ndim, 2)
        self.assertGreater(processed.sum(), 0)


if __name__ == "__main__":
    unittest.main()
