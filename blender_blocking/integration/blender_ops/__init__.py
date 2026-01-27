"""Blender ops helpers for scene setup, mesh generation, and rendering."""

from .mesh_generator import create_mesh_from_contours, extrude_profile
from .scene_setup import setup_scene, add_camera, add_lighting
from .render_utils import render_orthogonal_views, save_render
from .profile_loft_mesh import create_loft_mesh_from_slices
from .camera_framing import compute_bounds_world, configure_ortho_camera_for_view
from .ops_fastpath import suppress_view_layer_updates
from .silhouette_render import (
    collect_target_objects,
    render_silhouette_frame,
    set_camera_orbit,
    set_camera_top,
    silhouette_session,
)

__all__ = [
    "create_mesh_from_contours",
    "extrude_profile",
    "setup_scene",
    "add_camera",
    "add_lighting",
    "render_orthogonal_views",
    "save_render",
    "create_loft_mesh_from_slices",
    "compute_bounds_world",
    "configure_ortho_camera_for_view",
    "suppress_view_layer_updates",
    "collect_target_objects",
    "render_silhouette_frame",
    "set_camera_orbit",
    "set_camera_top",
    "silhouette_session",
]
