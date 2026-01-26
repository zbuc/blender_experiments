"""
Multi-angle vertical profile extraction from 3D meshes.

Extracts vertical profiles at multiple angles from a Blender mesh object,
enabling multi-view primitive fitting for improved reconstruction accuracy.

Usage:
    profiles = extract_multi_angle_profiles(mesh_obj, num_angles=12, num_heights=20)
    combined = combine_profiles(profiles, method='median')
"""

from __future__ import annotations

try:
    import bpy
    import bmesh
    from mathutils import Vector, Matrix

    BLENDER_AVAILABLE = True
except ImportError:
    bpy = None
    bmesh = None
    Vector = None
    Matrix = None
    BLENDER_AVAILABLE = False
import math
import numpy as np
from typing import Dict, List, Optional, Tuple


def extract_profile_at_angle(
    mesh_obj: bpy.types.Object,
    angle_degrees: float,
    num_samples: int = 20,
    bounds_min: Optional[Vector] = None,
    bounds_max: Optional[Vector] = None,
) -> List[Tuple[float, float]]:
    """
    Extract vertical profile from mesh at a specific angle.

    Slices the mesh horizontally at multiple heights and measures the
    radius of the cross-section at each height, viewed from the given angle.

    Args:
        mesh_obj: Blender mesh object to analyze
        angle_degrees: Viewing angle in degrees (0 = front/X-axis, 90 = side/Y-axis)
        num_samples: Number of height samples to take
        bounds_min: Optional minimum bounds (auto-detected if None)
        bounds_max: Optional maximum bounds (auto-detected if None)

    Returns:
        List of (height_normalized, radius_normalized) tuples
        where height is 0 (bottom) to 1 (top)
        and radius is 0 to 1 (normalized to max dimension)
    """
    if not BLENDER_AVAILABLE:
        raise RuntimeError("Blender API not available for profile extraction")
    if num_samples <= 0:
        raise ValueError("num_samples must be >= 1")

    # Get mesh bounds
    if bounds_min is None or bounds_max is None:
        bbox = [mesh_obj.matrix_world @ Vector(corner) for corner in mesh_obj.bound_box]
        calc_min = Vector(
            (min(v.x for v in bbox), min(v.y for v in bbox), min(v.z for v in bbox))
        )
        calc_max = Vector(
            (max(v.x for v in bbox), max(v.y for v in bbox), max(v.z for v in bbox))
        )
        bounds_min = bounds_min or calc_min
        bounds_max = bounds_max or calc_max

    z_min, z_max = bounds_min.z, bounds_max.z
    z_range = z_max - z_min

    if z_range <= 0:
        return [(0.5, 0.0)]  # Degenerate case

    # Calculate rotation for viewing angle
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Max radius for normalization (half of max XY dimension)
    max_radius = max(bounds_max.x - bounds_min.x, bounds_max.y - bounds_min.y) / 2.0

    if num_samples > 1:
        t_values = np.linspace(0.0, 1.0, num_samples)
    else:
        t_values = np.array([0.5])
    z_worlds = bounds_min.z + t_values * z_range

    center_xy = (
        (bounds_min.x + bounds_max.x) / 2.0,
        (bounds_min.y + bounds_max.y) / 2.0,
    )
    max_distance = max(bounds_max.x - bounds_min.x, bounds_max.y - bounds_min.y) * 1.5
    fallback_dirs = [
        Vector(
            (
                math.cos(angle_rad + (i * 2 * math.pi / 16)),
                math.sin(angle_rad + (i * 2 * math.pi / 16)),
                0,
            )
        )
        for i in range(16)
    ]

    profile = []
    for t, z_world in zip(t_values, z_worlds):
        radius = _measure_radius_at_height(
            mesh_obj,
            float(z_world),
            angle_rad,
            cos_a,
            sin_a,
            bounds_min,
            bounds_max,
            center_xy=center_xy,
            max_distance=max_distance,
            fallback_dirs=fallback_dirs,
        )

        radius_normalized = radius / max_radius if max_radius > 0 else 0.0
        profile.append((float(t), float(radius_normalized)))

    return profile


def _measure_radius_at_height(
    mesh_obj: bpy.types.Object,
    z_height: float,
    angle_rad: float,
    cos_a: float,
    sin_a: float,
    bounds_min: Vector,
    bounds_max: Vector,
    *,
    center_xy: Optional[Tuple[float, float]] = None,
    max_distance: Optional[float] = None,
    fallback_dirs: Optional[List[Vector]] = None,
) -> float:
    """
    Measure radius of mesh cross-section at given height and angle.

    Uses raycasting to find the mesh boundary in the viewing direction.

    Args:
        mesh_obj: Blender mesh object
        z_height: Z coordinate of horizontal slice
        angle_rad: Viewing angle in radians
        cos_a: Cosine of viewing angle (precomputed)
        sin_a: Sine of viewing angle (precomputed)
        bounds_min: Minimum bounds
        bounds_max: Maximum bounds

    Returns:
        Measured radius at this height
    """
    if not BLENDER_AVAILABLE:
        raise RuntimeError("Blender API not available for profile extraction")
    # Center point at this height
    if center_xy is None:
        center_x = (bounds_min.x + bounds_max.x) / 2.0
        center_y = (bounds_min.y + bounds_max.y) / 2.0
    else:
        center_x, center_y = center_xy
    center = Vector((center_x, center_y, z_height))

    # Direction perpendicular to viewing angle (for radius measurement)
    # If viewing from angle θ, we measure radius perpendicular to that
    perp_direction = Vector((cos_a, sin_a, 0))

    # Cast ray outward from center to find edge
    if max_distance is None:
        max_distance = (
            max(bounds_max.x - bounds_min.x, bounds_max.y - bounds_min.y) * 1.5
        )  # Extra margin

    # Cast ray in both directions (positive and negative perpendicular)
    radius_pos = _raycast_distance(mesh_obj, center, perp_direction, max_distance)
    radius_neg = _raycast_distance(mesh_obj, center, -perp_direction, max_distance)

    # Use the maximum of the two (accounting for off-center objects)
    radius = max(radius_pos, radius_neg)

    # Fallback: if no hit, try sampling around the circle
    if radius < 0.001:
        radius = _sample_circular_radius(
            mesh_obj,
            center,
            angle_rad,
            max_distance,
            directions=fallback_dirs,
        )

    return radius


def _raycast_distance(
    mesh_obj: bpy.types.Object, origin: Vector, direction: Vector, max_distance: float
) -> float:
    """Cast ray and return distance to first hit."""
    if not BLENDER_AVAILABLE:
        raise RuntimeError("Blender API not available for raycast")
    from integration.blender_ops.raycast_utils import ray_cast_world

    result, location, normal, index = ray_cast_world(
        mesh_obj, origin, direction, max_distance
    )

    if result:
        return (location - origin).length
    return 0.0


def _sample_circular_radius(
    mesh_obj: bpy.types.Object,
    center: Vector,
    base_angle: float,
    max_distance: float,
    *,
    directions: Optional[List[Vector]] = None,
) -> float:
    """
    Sample radius by casting rays in multiple directions around a circle.

    Fallback method when single raycast doesn't hit.
    """
    if not BLENDER_AVAILABLE:
        raise RuntimeError("Blender API not available for radius sampling")
    num_rays = 16
    max_radius = 0.0

    if directions is None:
        directions = [
            Vector(
                (
                    math.cos(base_angle + (i * 2 * math.pi / num_rays)),
                    math.sin(base_angle + (i * 2 * math.pi / num_rays)),
                    0,
                )
            )
            for i in range(num_rays)
        ]

    for direction in directions:
        distance = _raycast_distance(mesh_obj, center, direction, max_distance)
        max_radius = max(max_radius, distance)

    return max_radius


def extract_multi_angle_profiles(
    mesh_obj: bpy.types.Object,
    num_angles: int = 12,
    num_heights: int = 20,
    bounds_min: Optional[Vector] = None,
    bounds_max: Optional[Vector] = None,
) -> List[List[Tuple[float, float]]]:
    """
    Extract vertical profiles at multiple angles around the object.

    Args:
        mesh_obj: Blender mesh object to analyze
        num_angles: Number of viewing angles (evenly spaced 0-360°)
        num_heights: Number of height samples per profile
        bounds_min: Optional minimum bounds
        bounds_max: Optional maximum bounds

    Returns:
        List of profiles, each profile is a list of (height, radius) tuples
    """
    if not BLENDER_AVAILABLE:
        raise RuntimeError("Blender API not available for profile extraction")
    if num_angles <= 0:
        raise ValueError("num_angles must be >= 1")
    if num_heights <= 0:
        raise ValueError("num_heights must be >= 1")

    # Compute bounds once for all angles
    if bounds_min is None or bounds_max is None:
        bbox = [mesh_obj.matrix_world @ Vector(corner) for corner in mesh_obj.bound_box]
        calc_min = Vector(
            (min(v.x for v in bbox), min(v.y for v in bbox), min(v.z for v in bbox))
        )
        calc_max = Vector(
            (max(v.x for v in bbox), max(v.y for v in bbox), max(v.z for v in bbox))
        )
        bounds_min = bounds_min or calc_min
        bounds_max = bounds_max or calc_max

    z_min, z_max = bounds_min.z, bounds_max.z
    z_range = z_max - z_min
    if z_range <= 0:
        return [[(0.5, 0.0)] for _ in range(num_angles)]

    if num_heights > 1:
        t_values = np.linspace(0.0, 1.0, num_heights)
    else:
        t_values = np.array([0.5])
    z_worlds = bounds_min.z + t_values * z_range

    center_xy = (
        (bounds_min.x + bounds_max.x) / 2.0,
        (bounds_min.y + bounds_max.y) / 2.0,
    )
    max_radius = max(bounds_max.x - bounds_min.x, bounds_max.y - bounds_min.y) / 2.0
    max_distance = max(bounds_max.x - bounds_min.x, bounds_max.y - bounds_min.y) * 1.5

    profiles = []
    angle_step = 360.0 / num_angles

    for i in range(num_angles):
        angle = i * angle_step
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        fallback_dirs = [
            Vector(
                (
                    math.cos(angle_rad + (j * 2 * math.pi / 16)),
                    math.sin(angle_rad + (j * 2 * math.pi / 16)),
                    0,
                )
            )
            for j in range(16)
        ]

        profile = []
        for t, z_world in zip(t_values, z_worlds):
            radius = _measure_radius_at_height(
                mesh_obj,
                float(z_world),
                angle_rad,
                cos_a,
                sin_a,
                bounds_min,
                bounds_max,
                center_xy=center_xy,
                max_distance=max_distance,
                fallback_dirs=fallback_dirs,
            )
            radius_normalized = radius / max_radius if max_radius > 0 else 0.0
            profile.append((float(t), float(radius_normalized)))
        profiles.append(profile)

    return profiles


def combine_profiles(
    profiles: List[List[Tuple[float, float]]], method: str = "median"
) -> List[Tuple[float, float]]:
    """
    Combine multiple profiles into a single averaged profile.

    Args:
        profiles: List of profiles (each is list of (height, radius) tuples)
        method: Combination method ('mean', 'median', 'min', 'max')

    Returns:
        Combined profile as list of (height, radius) tuples
    """
    if not profiles:
        return []

    if len(profiles) == 1:
        return profiles[0]

    # Assume all profiles have same number of samples
    lengths = {len(profile) for profile in profiles}
    if len(lengths) != 1:
        raise ValueError("Profiles must all have the same number of samples")

    values = np.asarray(profiles, dtype=np.float64)
    heights = values[0, :, 0]
    radii = values[:, :, 1]

    if method == "mean":
        combined_radii = np.mean(radii, axis=0)
    elif method == "median":
        combined_radii = np.median(radii, axis=0)
    elif method == "min":
        combined_radii = np.min(radii, axis=0)
    elif method == "max":
        combined_radii = np.max(radii, axis=0)
    else:
        raise ValueError(f"Unknown combination method: {method}")

    return [(float(h), float(r)) for h, r in zip(heights, combined_radii)]


def visualize_profiles(
    profiles: List[List[Tuple[float, float]]],
    combined: Optional[List[Tuple[float, float]]] = None,
    output_path: Optional[str] = None,
) -> None:
    """
    Visualize multiple profiles and their combination.

    Args:
        profiles: List of profiles to visualize
        combined: Optional combined profile to overlay
        output_path: Optional path to save plot (displays if None)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available for visualization")
        return

    if not profiles:
        print("Warning: no profiles provided for visualization")
        return

    plt.figure(figsize=(10, 6))

    # Plot individual profiles with transparency
    for i, profile in enumerate(profiles):
        heights = [h for h, r in profile]
        radii = [r for h, r in profile]
        angle = i * (360.0 / len(profiles))
        plt.plot(
            radii,
            heights,
            alpha=0.3,
            color="blue",
            label=f"{angle}°" if i < 3 else None,
        )

    # Plot combined profile
    if combined:
        heights = [h for h, r in combined]
        radii = [r for h, r in combined]
        plt.plot(radii, heights, linewidth=2, color="red", label="Combined")

    plt.xlabel("Radius (normalized)")
    plt.ylabel("Height (normalized)")
    plt.title(f"Multi-Angle Profiles ({len(profiles)} angles)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path)
        print(f"Saved profile visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def test_multi_angle_extraction() -> (
    Tuple[List[List[Tuple[float, float]]], List[Tuple[float, float]]]
):
    """
    Test multi-angle profile extraction on a simple cylinder.
    """
    if not BLENDER_AVAILABLE:
        raise RuntimeError("Blender API not available for profile extraction")
    # Create test cylinder
    bpy.ops.mesh.primitive_cylinder_add(radius=1.0, depth=2.0, location=(0, 0, 1.0))
    cylinder = bpy.context.active_object

    print("\nTesting multi-angle profile extraction...")

    # Extract profiles at 12 angles
    profiles = extract_multi_angle_profiles(cylinder, num_angles=12, num_heights=10)

    print(f"Extracted {len(profiles)} profiles")
    for i, profile in enumerate(profiles):
        angle = i * 30
        print(f"  Profile at {angle}°: {len(profile)} samples")
        radii = [r for h, r in profile]
        print(f"    Radius range: {min(radii):.3f} to {max(radii):.3f}")

    # Combine profiles
    combined = combine_profiles(profiles, method="median")
    print(f"\nCombined profile (median): {len(combined)} samples")
    radii = [r for h, r in combined]
    print(f"  Radius range: {min(radii):.3f} to {max(radii):.3f}")

    # For a perfect cylinder, all radii should be similar (~1.0 normalized)
    radius_std = np.std(radii)
    print(f"  Radius std dev: {radius_std:.4f} (should be low for cylinder)")

    # Validate transformed mesh matches base profile (translation + Z-rotation)
    bpy.ops.mesh.primitive_cylinder_add(
        radius=1.0, depth=2.0, location=(1.5, -0.5, 0.5)
    )
    transformed = bpy.context.active_object
    transformed.rotation_euler = (0.0, 0.0, math.radians(45))

    transformed_profiles = extract_multi_angle_profiles(
        transformed, num_angles=12, num_heights=10
    )
    transformed_combined = combine_profiles(transformed_profiles, method="median")

    base_radii = [r for h, r in combined]
    transformed_radii = [r for h, r in transformed_combined]

    if len(base_radii) == len(transformed_radii):
        diff = np.abs(np.array(base_radii) - np.array(transformed_radii))
        print(f"  Transformed profile max diff: {diff.max():.4f}")
        assert np.allclose(base_radii, transformed_radii, atol=1e-3)

    # Invalid sampling counts should raise clear errors
    try:
        extract_multi_angle_profiles(cylinder, num_angles=0, num_heights=10)
    except ValueError:
        print("✓ num_angles=0 rejected")
    else:
        raise AssertionError("Expected ValueError for num_angles=0")

    try:
        extract_multi_angle_profiles(cylinder, num_angles=12, num_heights=0)
    except ValueError:
        print("✓ num_heights=0 rejected")
    else:
        raise AssertionError("Expected ValueError for num_heights=0")

    try:
        extract_profile_at_angle(cylinder, angle_degrees=0.0, num_samples=0)
    except ValueError:
        print("✓ num_samples=0 rejected")
    else:
        raise AssertionError("Expected ValueError for num_samples=0")

    return profiles, combined


if __name__ == "__main__":
    # Run test when executed in Blender
    test_multi_angle_extraction()
