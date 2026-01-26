"""Blender-only test for the loft workflow path."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import bpy

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

try:
    from PIL import Image, ImageDraw

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from config import BlockingConfig
from main_integration import BlockingWorkflow


def _create_simple_silhouette(path: Path, view: str) -> None:
    width = 256
    height = 256
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)

    if view in {"front", "side"}:
        draw.rectangle([96, 40, 160, 220], fill=0)
    else:
        draw.ellipse([96, 96, 160, 160], fill=0)

    img.save(path)


@unittest.skipUnless(BLENDER_AVAILABLE and PIL_AVAILABLE, "Requires Blender and PIL")
class TestLoftWorkflow(unittest.TestCase):
    def test_loft_workflow_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            front_path = temp_path / "front.png"
            side_path = temp_path / "side.png"

            _create_simple_silhouette(front_path, "front")
            _create_simple_silhouette(side_path, "side")

            config = BlockingConfig()
            config.reconstruction.reconstruction_mode = "loft_profile"

            workflow = BlockingWorkflow(
                front_path=str(front_path),
                side_path=str(side_path),
                config=config,
            )

            result = workflow.run_full_workflow(num_slices=8)

            self.assertIsNotNone(result)
            self.assertEqual(result.type, "MESH")
            self.assertGreater(len(result.data.vertices), 0)


if __name__ == "__main__":
    unittest.main()
