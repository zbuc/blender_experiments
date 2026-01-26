"""
Multi-view reconstruction module.

Provides Visual Hull reconstruction from N camera views for higher accuracy
shape reconstruction compared to 3-view orthogonal baseline.

Main classes:
- MultiViewVisualHull: Voxel-based visual hull reconstruction
- CameraView: Single camera view with silhouette projection

Functions:
- load_multi_view_turntable: Load turntable image sequence
"""

from .visual_hull import MultiViewVisualHull, CameraView, load_multi_view_turntable

__all__ = [
    "MultiViewVisualHull",
    "CameraView",
    "load_multi_view_turntable",
]
