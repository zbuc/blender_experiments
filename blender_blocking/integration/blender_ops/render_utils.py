"""Rendering utilities for Blender."""

from __future__ import annotations

from pathlib import Path
import re
import uuid
from typing import Callable, Dict, List, Optional, Tuple

try:
    import bpy
    import mathutils

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

from integration.blender_ops.camera_framing import (
    compute_bounds_world,
    configure_ortho_camera_for_view,
)
from integration.blender_ops.silhouette_render import (
    collect_target_objects,
    render_silhouette_frame,
    silhouette_session,
)


def render_orthogonal_views(
    output_dir: str,
    views: List[str] = ["front", "side", "top"],
    *,
    target_objects: Optional[List[object]] = None,
    fit_to_bounds: bool = True,
    margin_frac: float = 0.08,
    resolution: Tuple[int, int] = (512, 512),
    color_mode: str = "RGBA",
    transparent_bg: bool = True,
    force_material: bool = False,
    background_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    silhouette_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    camera_distance_factor: float = 2.0,
    party_mode: bool = False,
    filename_prefix: Optional[str] = None,
    start_index: int = 1,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Dict[str, str]:
    """
    Render orthogonal views of the scene.

    Args:
        output_dir: Directory to save renders
        views: List of views to render

    Returns:
        Dictionary mapping view names to output file paths
    """
    if not BLENDER_AVAILABLE:
        print("Warning: Blender API not available")
        return {}

    output_paths = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    target_objects = collect_target_objects(scene, target_objects)

    if not target_objects:
        print("Warning: No renderable mesh objects found.")
        return {}

    if fit_to_bounds and target_objects:
        bounds_min, bounds_max = compute_bounds_world(target_objects)
    else:
        bounds_min = mathutils.Vector((0.0, 0.0, 0.0))
        bounds_max = mathutils.Vector((1.0, 1.0, 1.0))

    def _unique_path(base_path: Path) -> Path:
        if not base_path.exists():
            return base_path
        stem = base_path.stem
        suffix = base_path.suffix
        match = re.match(r"^(.*?)(?:_(\d+))?$", stem)
        if match:
            base_stem = match.group(1)
            start_at = int(match.group(2)) if match.group(2) else 1
        else:
            base_stem = stem
            start_at = 1
        for idx in range(start_at + 1, 10000):
            candidate = base_path.with_name(f"{base_stem}_{idx}{suffix}")
            if not candidate.exists():
                return candidate
        return base_path.with_name(f"{base_stem}_{uuid.uuid4().hex[:8]}{suffix}")

    try:
        with silhouette_session(
            scene=scene,
            target_objects=target_objects,
            resolution=resolution,
            color_mode=color_mode,
            transparent_bg=transparent_bg,
            background_color=background_color,
            silhouette_color=silhouette_color,
            force_material=force_material,
            hide_non_targets=True,
            party_mode=party_mode,
        ) as session:
            for view in views:
                if view not in {"front", "side", "top"}:
                    continue

                configure_ortho_camera_for_view(
                    session.camera,
                    view,
                    bounds_min,
                    bounds_max,
                    margin_frac=margin_frac,
                    resolution=resolution,
                    distance_factor=camera_distance_factor,
                )

                if filename_prefix:
                    stem = f"{filename_prefix}{view}_{start_index}"
                else:
                    stem = view
                output_file = _unique_path(output_path / f"{stem}.png")
                render_silhouette_frame(session, output_file)

                output_paths[view] = str(output_file)
                if progress_callback is not None:
                    progress_callback(1)
    finally:
        pass

    return output_paths


def save_render(output_path: str) -> None:
    """
    Render and save the current scene.

    Args:
        output_path: Path to save the render
    """
    if not BLENDER_AVAILABLE:
        print("Warning: Blender API not available")
        return

    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
