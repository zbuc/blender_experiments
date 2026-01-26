"""Canonical silhouette IoU utilities."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from geometry.silhouette import extract_binary_silhouette
from config import SilhouetteExtractConfig

_CANONICALIZE_CACHE_MAX = 128
_CANONICALIZE_CACHE: "OrderedDict[Tuple[object, ...], np.ndarray]" = OrderedDict()
_CANONICALIZE_CACHE_HITS = 0
_CANONICALIZE_CACHE_MISSES = 0


@dataclass(frozen=True)
class IoUResult:
    """Result of an IoU comparison."""

    iou: float
    intersection: int
    union: int
    warnings: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        """Return a serializable dict of result details."""
        return {
            "iou": self.iou,
            "intersection": self.intersection,
            "union": self.union,
            "warnings": list(self.warnings),
        }


def mask_from_image_array(
    image: np.ndarray, *, extract_config: Optional[SilhouetteExtractConfig] = None
) -> np.ndarray:
    """Convert an image array into a boolean silhouette mask."""
    image = np.asarray(image)
    if extract_config is None:
        extract_config = SilhouetteExtractConfig(
            prefer_alpha=True,
            alpha_threshold=127,
            gray_threshold=None,
            invert_policy="auto",
            morph_close_px=0,
            morph_open_px=0,
            fill_holes=False,
            largest_component_only=False,
        )
    mask = extract_binary_silhouette(image, **extract_config.to_dict())
    return mask


def canonicalize_mask(
    mask: np.ndarray,
    *,
    output_size: int = 256,
    padding_frac: float = 0.1,
    anchor: str = "bottom_center",
    morph_close_px: int = 0,
) -> np.ndarray:
    """Canonicalize a boolean mask into a fixed-size canvas."""
    if output_size < 1:
        raise ValueError("output_size must be >= 1")

    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return np.zeros((output_size, output_size), dtype=bool)

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    cropped = mask[y0:y1, x0:x1]
    crop_h, crop_w = cropped.shape

    pad = int(round(max(crop_h, crop_w) * padding_frac))
    padded = np.pad(
        cropped, ((pad, pad), (pad, pad)), mode="constant", constant_values=False
    )

    padded_h, padded_w = padded.shape
    scale = output_size / float(max(padded_h, padded_w))
    new_w = max(1, int(round(padded_w * scale)))
    new_h = max(1, int(round(padded_h * scale)))

    resized = cv2.resize(
        padded.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST
    )
    resized = resized.astype(bool)

    canvas = np.zeros((output_size, output_size), dtype=bool)

    if anchor == "bottom_center":
        x_start = (output_size - new_w) // 2
        y_start = output_size - new_h
    elif anchor == "center":
        x_start = (output_size - new_w) // 2
        y_start = (output_size - new_h) // 2
    else:
        raise ValueError(f"Unknown anchor: {anchor}")

    x_start = max(0, min(output_size - new_w, x_start))
    y_start = max(0, min(output_size - new_h, y_start))
    canvas[y_start : y_start + new_h, x_start : x_start + new_w] = resized

    if morph_close_px > 0:
        k = max(3, int(morph_close_px))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        canvas = cv2.morphologyEx(
            canvas.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)

    return canvas


def _hash_mask(mask: np.ndarray) -> str:
    """Return a stable hash for a boolean mask."""
    mask_bool = np.ascontiguousarray(mask.astype(bool))
    return hashlib.sha1(mask_bool.tobytes()).hexdigest()


def _make_cache_key(
    mask: np.ndarray,
    *,
    output_size: int,
    padding_frac: float,
    anchor: str,
    morph_close_px: int,
) -> Tuple[object, ...]:
    """Build a cache key from mask content and canonicalization params."""
    mask_bool = np.asarray(mask).astype(bool, copy=False)
    return (
        _hash_mask(mask_bool),
        mask_bool.shape,
        int(output_size),
        float(padding_frac),
        str(anchor),
        int(morph_close_px),
    )


def canonicalize_mask_cached(
    mask: np.ndarray,
    *,
    output_size: int = 256,
    padding_frac: float = 0.1,
    anchor: str = "bottom_center",
    morph_close_px: int = 0,
) -> np.ndarray:
    """Canonicalize a mask using a small LRU cache."""
    global _CANONICALIZE_CACHE_HITS
    global _CANONICALIZE_CACHE_MISSES

    key = _make_cache_key(
        mask,
        output_size=output_size,
        padding_frac=padding_frac,
        anchor=anchor,
        morph_close_px=morph_close_px,
    )

    cached = _CANONICALIZE_CACHE.get(key)
    if cached is not None:
        _CANONICALIZE_CACHE_HITS += 1
        _CANONICALIZE_CACHE.move_to_end(key)
        return cached

    _CANONICALIZE_CACHE_MISSES += 1
    result = canonicalize_mask(
        mask,
        output_size=output_size,
        padding_frac=padding_frac,
        anchor=anchor,
        morph_close_px=morph_close_px,
    )
    _CANONICALIZE_CACHE[key] = result
    if len(_CANONICALIZE_CACHE) > _CANONICALIZE_CACHE_MAX:
        _CANONICALIZE_CACHE.popitem(last=False)
    return result


def reset_canonicalize_cache() -> None:
    """Clear the canonicalization cache and counters."""
    global _CANONICALIZE_CACHE_HITS
    global _CANONICALIZE_CACHE_MISSES

    _CANONICALIZE_CACHE.clear()
    _CANONICALIZE_CACHE_HITS = 0
    _CANONICALIZE_CACHE_MISSES = 0


def get_canonicalize_cache_stats() -> Dict[str, int]:
    """Return cache hit/miss counters and current size."""
    return {
        "hits": _CANONICALIZE_CACHE_HITS,
        "misses": _CANONICALIZE_CACHE_MISSES,
        "size": len(_CANONICALIZE_CACHE),
        "capacity": _CANONICALIZE_CACHE_MAX,
    }


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> IoUResult:
    """Compute intersection-over-union between two boolean masks."""
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())

    warnings: List[str] = []
    if union == 0:
        warnings.append("Both masks are empty")
        return IoUResult(iou=0.0, intersection=0, union=0, warnings=tuple(warnings))

    if mask_a.sum() == 0 or mask_b.sum() == 0:
        warnings.append("One mask is empty")

    iou = float(intersection) / float(union)
    return IoUResult(
        iou=iou, intersection=intersection, union=union, warnings=tuple(warnings)
    )
