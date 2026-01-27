"""Pure-Python validation utilities (no bpy imports)."""

from .silhouette_iou import (
    IoUResult,
    canonicalize_mask,
    compute_mask_iou,
    mask_from_image_array,
)

__all__ = [
    "IoUResult",
    "canonicalize_mask",
    "compute_mask_iou",
    "mask_from_image_array",
]
