"""
Multi-view reconstruction module.

Extends current 3-view pipeline to support 8-12 views for higher IoU accuracy.

Based on research: blender_experiments/research/MULTI_VIEW_RECONSTRUCTION_RESEARCH.md
"""

from .visual_hull import MultiViewVisualHull, CameraView, load_multi_view_turntable

__all__ = [
    'MultiViewVisualHull',
    'CameraView',
    'load_multi_view_turntable'
]
