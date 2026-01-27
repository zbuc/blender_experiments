"""Blender-only tests for silhouette rendering helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Add parent directory to path for blender_blocking imports
sys.path.insert(0, str(Path(__file__).parent))

# Add ~/blender_python_packages for user-installed dependencies
sys.path.insert(0, str(Path.home() / "blender_python_packages"))

try:
    import bpy
    import numpy as np
    from mathutils import Vector

    BLENDER_AVAILABLE = True
except ImportError:
    bpy = None
    np = None
    Vector = None
    BLENDER_AVAILABLE = False

from integration.blender_ops.camera_framing import (
    compute_bounds_world,
    configure_ortho_camera_for_view,
)
from integration.blender_ops.render_utils import render_orthogonal_views
from integration.blender_ops.silhouette_render import (
    ensure_silhouette_material,
    render_silhouette_frame,
    silhouette_session,
)
from utils.blender_version import get_eevee_engine_name


def _load_mask(image_path: Path) -> np.ndarray:
    image = bpy.data.images.load(str(image_path))
    width, height = image.size
    channels = image.channels
    pixels = np.array(image.pixels[:])
    if channels >= 3:
        pixels = pixels.reshape((height, width, channels))
        gray = pixels[:, :, :3].mean(axis=2)
    else:
        gray = pixels.reshape((height, width))
    mask = gray < 0.5
    bpy.data.images.remove(image)
    return mask


@unittest.skipUnless(BLENDER_AVAILABLE, "Requires Blender bpy")
class TestSilhouetteRendering(unittest.TestCase):
    def tearDown(self) -> None:
        if not BLENDER_AVAILABLE:
            return
        for obj in list(bpy.context.scene.objects):
            if obj.type == "MESH" and obj.name.startswith("TestRender_"):
                bpy.data.objects.remove(obj, do_unlink=True)

    def test_render_isolation(self) -> None:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
        target = bpy.context.active_object
        target.name = "TestRender_Target"

        bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0.75, 0, 0))
        extra = bpy.context.active_object
        extra.name = "TestRender_Extra"

        bounds_min, bounds_max = compute_bounds_world([target, extra])
        output_dir = Path(__file__).parent / "test_output" / "render_isolation"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Test 1: Render only target (hide_non_targets=True)
        with silhouette_session(
            target_objects=[target],
            resolution=(128, 128),
            color_mode="RGBA",
            transparent_bg=False,
            hide_non_targets=True,
            engine=get_eevee_engine_name(),
        ) as session:
            configure_ortho_camera_for_view(
                session.camera,
                "front",
                bounds_min,
                bounds_max,
                margin_frac=0.08,
                resolution=(128, 128),
            )
            hidden_path = output_dir / "hidden.png"
            render_silhouette_frame(session, hidden_path)

        # Test 2: Render both objects (include both as targets)
        with silhouette_session(
            target_objects=[target, extra],
            resolution=(128, 128),
            color_mode="RGBA",
            transparent_bg=False,
            hide_non_targets=True,
            engine=get_eevee_engine_name(),
        ) as session:
            configure_ortho_camera_for_view(
                session.camera,
                "front",
                bounds_min,
                bounds_max,
                margin_frac=0.08,
                resolution=(128, 128),
            )
            visible_path = output_dir / "visible.png"
            render_silhouette_frame(session, visible_path)

        hidden_mask = _load_mask(hidden_path)
        visible_mask = _load_mask(visible_path)

        hidden_area = float(hidden_mask.sum())
        visible_area = float(visible_mask.sum())

        self.assertGreater(hidden_area, 0.0)
        self.assertGreater(visible_area, hidden_area * 1.1)

    def test_framing_margin(self) -> None:
        bpy.ops.mesh.primitive_cylinder_add(radius=0.4, depth=2.0, location=(0, 0, 0))
        obj = bpy.context.active_object
        obj.name = "TestRender_Cylinder"

        output_dir = Path(__file__).parent / "test_output" / "render_framing"
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = render_orthogonal_views(
            str(output_dir),
            views=["front"],
            target_objects=[obj],
            margin_frac=0.08,
            resolution=(128, 128),
            color_mode="RGBA",
            transparent_bg=False,
            force_material=True,
        )
        front_path = Path(outputs["front"])
        mask = _load_mask(front_path)

        ys, xs = np.where(mask)
        self.assertGreater(ys.min(), 0)
        self.assertGreater(xs.min(), 0)
        self.assertLess(ys.max(), mask.shape[0] - 1)
        self.assertLess(xs.max(), mask.shape[1] - 1)


if __name__ == "__main__":
    unittest.main()
