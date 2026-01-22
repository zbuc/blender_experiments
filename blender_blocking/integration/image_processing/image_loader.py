"""Image loading utilities for orthogonal and multi-view reference images."""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import re

# Try PIL/Pillow, fall back to alternatives
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    # Try alternative image loading (opencv, imageio, etc.)
    try:
        import cv2
        HAS_CV2 = True
    except ImportError:
        HAS_CV2 = False


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file.

    Args:
        image_path: Path to the image file

    Returns:
        Image as numpy array
    """
    if HAS_PIL:
        img = Image.open(image_path)
        return np.array(img)
    elif HAS_CV2:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        # OpenCV loads as BGR, convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    else:
        raise ImportError("No image loading library available. Install PIL/Pillow or OpenCV.")


def load_orthogonal_views(
    front_path: Optional[str] = None,
    side_path: Optional[str] = None,
    top_path: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Load orthogonal reference images (front, side, top views).

    Legacy 3-view mode (backward compatible).

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


def load_multi_view_turntable(
    directory: str,
    pattern: str = "view_{angle:03d}.png",
    angles: Optional[List[float]] = None,
    include_top: bool = True,
    top_filename: str = "top.png"
) -> Dict[str, Tuple[np.ndarray, float, str]]:
    """
    Load multi-view turntable sequence.

    Supports flexible naming conventions:
    - Angle-based: view_000.png, view_030.png, view_060.png, ...
    - Numeric: 0.png, 30.png, 60.png, 90.png, ...
    - Auto-detection: Scans directory for numbered files

    Args:
        directory: Directory containing view images
        pattern: Filename pattern with {angle} placeholder
                 Examples: "view_{angle:03d}.png", "{angle}.png", "lateral_{angle:03d}.jpg"
        angles: List of rotation angles (degrees). If None, auto-detect from filenames
        include_top: Whether to load top view
        top_filename: Filename for top view (if include_top=True)

    Returns:
        Dictionary mapping 'lateral_N' -> (image, angle, view_type) and 'top' -> (image, 0.0, 'top')
        where N is the view index
    """
    directory = Path(directory)
    views = {}

    if angles is None:
        # Auto-detect angles from directory
        angles = _auto_detect_angles(directory, pattern)
        if not angles:
            raise ValueError(f"No turntable views found in {directory} with pattern {pattern}")

    # Load lateral views
    for i, angle in enumerate(angles):
        # Generate filename from pattern
        filename = pattern.replace("{angle:03d}", f"{int(angle):03d}")
        filename = filename.replace("{angle}", str(int(angle)))

        filepath = directory / filename

        if filepath.exists():
            img = load_image(str(filepath))
            views[f'lateral_{i}'] = (img, float(angle), 'lateral')
        else:
            print(f"Warning: Expected view file not found: {filepath}")

    # Load top view
    if include_top:
        top_path = directory / top_filename
        if top_path.exists():
            img = load_image(str(top_path))
            views['top'] = (img, 0.0, 'top')
        else:
            print(f"Warning: Top view not found: {top_path}")

    return views


def _auto_detect_angles(directory: Path, pattern: str) -> List[float]:
    """
    Auto-detect rotation angles from filenames in directory.

    Args:
        directory: Directory to scan
        pattern: Filename pattern (may contain {angle} or {angle:03d})

    Returns:
        List of detected angles, sorted
    """
    # Convert pattern to regex
    # Replace {angle:03d} or {angle} with regex group
    regex_pattern = pattern.replace("{angle:03d}", r"(\d{3})")
    regex_pattern = regex_pattern.replace("{angle}", r"(\d+)")
    regex_pattern = regex_pattern.replace(".", r"\.")  # Escape dots
    regex_pattern = f"^{regex_pattern}$"

    angles = []

    # Scan directory
    for filepath in directory.glob("*"):
        if not filepath.is_file():
            continue

        match = re.match(regex_pattern, filepath.name)
        if match:
            angle_str = match.group(1)
            angles.append(float(angle_str))

    return sorted(angles)


def load_multi_view_auto(
    directory: str,
    num_views: Optional[int] = None,
    include_top: bool = True
) -> Dict[str, Tuple[np.ndarray, float, str]]:
    """
    Convenience function to auto-detect and load multi-view sequence.

    Tries common naming conventions in order:
    1. view_000.png, view_030.png, ...
    2. 0.png, 30.png, 60.png, ...
    3. lateral_000.png, lateral_030.png, ...

    Args:
        directory: Directory containing images
        num_views: Expected number of lateral views (for validation, optional)
        include_top: Whether to include top view

    Returns:
        Dictionary mapping view names to (image, angle, view_type) tuples
    """
    directory = Path(directory)

    # Try common patterns
    patterns = [
        "view_{angle:03d}.png",
        "{angle}.png",
        "lateral_{angle:03d}.png",
        "view_{angle:03d}.jpg",
        "{angle}.jpg",
    ]

    for pattern in patterns:
        try:
            views = load_multi_view_turntable(
                str(directory),
                pattern=pattern,
                include_top=include_top
            )

            if views:
                num_lateral = sum(1 for k in views if k.startswith('lateral_'))
                print(f"Auto-detected {num_lateral} lateral views using pattern: {pattern}")

                if num_views and num_lateral != num_views:
                    print(f"Warning: Expected {num_views} views, found {num_lateral}")

                return views

        except (ValueError, FileNotFoundError):
            continue

    raise ValueError(f"Could not auto-detect multi-view sequence in {directory}")


def validate_view_coverage(
    views: Dict[str, Tuple[np.ndarray, float, str]],
    expected_spacing: float = 30.0
) -> bool:
    """
    Validate that lateral views are evenly distributed around 360°.

    Args:
        views: Dictionary of views from load_multi_view_turntable()
        expected_spacing: Expected angular spacing (degrees)

    Returns:
        True if views are well-distributed, False otherwise
    """
    lateral_angles = [
        angle for name, (img, angle, vtype) in views.items()
        if vtype == 'lateral'
    ]

    if len(lateral_angles) < 3:
        return False  # Need at least 3 views

    lateral_angles = sorted(lateral_angles)

    # Check spacing
    for i in range(len(lateral_angles)):
        next_i = (i + 1) % len(lateral_angles)
        spacing = (lateral_angles[next_i] - lateral_angles[i]) % 360

        if abs(spacing - expected_spacing) > 5.0:  # 5° tolerance
            print(f"Warning: Irregular spacing between {lateral_angles[i]}° and {lateral_angles[next_i]}°: {spacing:.1f}°")
            return False

    return True
