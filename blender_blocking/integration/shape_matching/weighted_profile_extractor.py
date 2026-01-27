"""
Weighted Profile Extraction for Phase 2

Extracts profiles at multiple angles and selects best orthogonal pairs
for robust radius estimation.

Based on researcher's analysis (hq-oqsh):
- Extract 12 profiles at even angles (0-360°)
- Select profiles at 0°, 90°, 180°, 270°
- Average opposite views: [0°, 180°] for front, [90°, 270°] for side
- This provides orthogonal profiles that SliceAnalyzer expects
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from types import ModuleType

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import using importlib to bypass cv2 dependency
import importlib.util


def import_module_from_path(module_name: str, file_path: Path) -> ModuleType:
    """Import module from file path bypassing __init__.py imports."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import mesh_profile_extractor
profile_extractor = import_module_from_path(
    "mesh_profile_extractor", Path(__file__).parent / "mesh_profile_extractor.py"
)


def extract_weighted_profiles(
    mesh_obj: Any,
    num_heights: int = 20,
    bounds_min: Optional[Any] = None,
    bounds_max: Optional[Any] = None,
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Extract weighted orthogonal profiles using best-angle selection.

    Addresses coordinate system issue from Phase 2 failure analysis:
    - SliceAnalyzer expects orthogonal profiles (front/side at 0°/90°)
    - Previous approach used arbitrary angles (30°, 60°, etc.)
    - This caused coordinate confusion and primitive amplification

    Algorithm:
    1. Extract 12 profiles at even angles (0° to 330° in 30° steps)
    2. Select profiles at cardinal angles: 0°, 90°, 180°, 270°
    3. Average opposite views to reduce noise:
       - Front: Average [0°, 180°]
       - Side: Average [90°, 270°]

    Args:
        mesh_obj: Blender mesh object
        num_heights: Number of height samples per profile (default 20)
        bounds_min: Optional minimum bounds (Vector or None)
        bounds_max: Optional maximum bounds (Vector or None)

    Returns:
        Dictionary with:
        - 'front': Averaged front profile (0° + 180°)
        - 'side': Averaged side profile (90° + 270°)
        - 'all': All 12 extracted profiles (for debugging)
    """
    # Extract 12 profiles at even angles (0-330° in 30° steps)
    print(f"Extracting 12 profiles at cardinal and intermediate angles...")
    all_profiles = profile_extractor.extract_multi_angle_profiles(
        mesh_obj,
        num_angles=12,
        num_heights=num_heights,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )
    if len(all_profiles) < 12:
        raise ValueError(
            f"Expected 12 profiles for weighted extraction, got {len(all_profiles)}"
        )

    # Map angle indices to degrees for 12-angle extraction
    # Index 0 = 0°, Index 1 = 30°, Index 2 = 60°, ...
    angle_to_index = {
        0: 0,  # 0° - front
        30: 1,
        60: 2,
        90: 3,  # 90° - side (right)
        120: 4,
        150: 5,
        180: 6,  # 180° - back (opposite of front)
        210: 7,
        240: 8,
        270: 9,  # 270° - side (left, opposite of 90°)
        300: 10,
        330: 11,
    }

    # Select cardinal angle profiles
    front_0 = all_profiles[angle_to_index[0]]  # 0° front view
    side_90 = all_profiles[angle_to_index[90]]  # 90° right side
    back_180 = all_profiles[angle_to_index[180]]  # 180° back view
    side_270 = all_profiles[angle_to_index[270]]  # 270° left side

    print(f"✓ Extracted profiles at cardinal angles (0°, 90°, 180°, 270°)")

    # Average opposite views to reduce noise and create robust orthogonal profiles
    # Front profile: Average of front (0°) and back (180°) views
    front_profile = profile_extractor.combine_profiles(
        [front_0, back_180], method="mean"
    )

    # Side profile: Average of right (90°) and left (270°) views
    side_profile = profile_extractor.combine_profiles(
        [side_90, side_270], method="mean"
    )

    # EMPIRICAL 2X SCALING CORRECTION (Option 3, per researcher hq-wtmf)
    # Root cause: extract_profile_at_angle() produces radii 2x too small
    # Empirical fix: Multiply all extracted radii by 2.0 before feeding to SliceAnalyzer
    print(f"\n✓ Applying empirical 2x scaling correction...")

    # Extract raw radii for reporting
    front_radii_raw = [r for _, r in front_profile]
    side_radii_raw = [r for _, r in side_profile]

    print(f"  Raw front profile: radius {min(front_radii_raw):.3f} to {max(front_radii_raw):.3f}")
    print(f"  Raw side profile: radius {min(side_radii_raw):.3f} to {max(side_radii_raw):.3f}")

    # Apply 2x correction
    front_profile = [(h, r * 2.0) for h, r in front_profile]
    side_profile = [(h, r * 2.0) for h, r in side_profile]

    # Extract corrected radii for reporting
    front_radii = [r for _, r in front_profile]
    side_radii = [r for _, r in side_profile]

    print(f"✓ Averaged opposite views:")
    print(
        f"  Front profile (0° + 180°): radius {min(front_radii):.3f} to {max(front_radii):.3f}"
    )
    print(
        f"  Side profile (90° + 270°): radius {min(side_radii):.3f} to {max(side_radii):.3f}"
    )

    return {
        "front": front_profile,
        "side": side_profile,
        "all": all_profiles,  # Keep all profiles for debugging
    }


def combine_orthogonal_profiles(
    front_profile: List[Tuple[float, float]],
    side_profile: List[Tuple[float, float]],
    method: str = "max",
) -> List[Tuple[float, float]]:
    """
    Combine front and side profiles into single profile for SliceAnalyzer.

    SliceAnalyzer expects a single vertical profile. We need to combine
    the orthogonal front/side profiles into one.

    Methods:
    - 'max': Use maximum radius at each height (conservative, captures full extent)
    - 'mean': Average radii (balanced)
    - 'geometric_mean': Geometric mean of radii (good for elliptical cross-sections)

    Args:
        front_profile: Front profile (height, radius) pairs
        side_profile: Side profile (height, radius) pairs
        method: Combination method ('max', 'mean', 'geometric_mean')

    Returns:
        Combined profile as (height, radius) pairs
    """
    if len(front_profile) != len(side_profile):
        raise ValueError(
            f"Profile length mismatch: front={len(front_profile)}, side={len(side_profile)}"
        )

    combined = []
    for (h_front, r_front), (h_side, r_side) in zip(front_profile, side_profile):
        # Heights should match (both normalized 0-1)
        if abs(h_front - h_side) > 0.01:
            raise ValueError(f"Height mismatch at index: {h_front} vs {h_side}")

        # Combine radii based on method
        if method == "max":
            radius = max(r_front, r_side)
        elif method == "mean":
            radius = (r_front + r_side) / 2.0
        elif method == "geometric_mean":
            radius = np.sqrt(r_front * r_side)
        else:
            raise ValueError(f"Unknown method: {method}")

        combined.append((h_front, radius))

    return combined
