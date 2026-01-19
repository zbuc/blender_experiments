"""Image loading utilities for orthogonal reference images."""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, Tuple


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file.

    Args:
        image_path: Path to the image file

    Returns:
        Image as numpy array
    """
    img = Image.open(image_path)
    return np.array(img)


def load_orthogonal_views(
    front_path: Optional[str] = None,
    side_path: Optional[str] = None,
    top_path: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Load orthogonal reference images (front, side, top views).

    Args:
        front_path: Path to front view image
        side_path: Path to side view image
        top_path: Path to top view image

    Returns:
        Dictionary mapping view names to image arrays
    """
    views = {}

    if front_path:
        views['front'] = load_image(front_path)
    if side_path:
        views['side'] = load_image(side_path)
    if top_path:
        views['top'] = load_image(top_path)

    return views
