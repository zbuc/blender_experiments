"""Vertex-level mesh refinement to match reference silhouettes.

This module provides vertex repositioning to improve 3D reconstruction accuracy
by matching silhouette boundaries extracted from reference images.

QA Iteration 3: Vertex-level refinement after subdivision.
"""

import math

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False


def interpolate_profile(profile, z_normalized):
    """
    Interpolate radius from vertical profile at given normalized height.

    Args:
        profile: List of (height, radius) tuples sorted by height
        z_normalized: Height value normalized to 0-1 range

    Returns:
        Interpolated radius value
    """
    if not profile:
        return 0.0

    # Clamp to profile range
    z_normalized = max(0.0, min(1.0, z_normalized))

    # Find bracketing points
    for i in range(len(profile) - 1):
        h1, r1 = profile[i]
        h2, r2 = profile[i + 1]

        if h1 <= z_normalized <= h2:
            # Linear interpolation
            if h2 == h1:
                return r1
            t = (z_normalized - h1) / (h2 - h1)
            return r1 + t * (r2 - r1)

    # If we're at or past the end, return last radius
    if z_normalized >= profile[-1][0]:
        return profile[-1][1]

    # If we're before the start, return first radius
    return profile[0][1]


def refine_mesh_to_silhouettes(
    mesh_obj,
    front_silhouette=None,
    side_silhouette=None,
    subdivision_levels=1
):
    """
    Refine mesh vertices to match reference silhouette boundaries.

    Two-phase approach:
    1. Subdivide mesh to increase vertex resolution
    2. Reposition vertices radially to match silhouette profiles

    Args:
        mesh_obj: Blender mesh object to refine
        front_silhouette: Front view image (numpy array) for profile extraction
        side_silhouette: Side view image (numpy array) for profile extraction
        subdivision_levels: Number of subdivision levels (1=4x vertices, 2=16x vertices)

    Returns:
        Refined mesh object
    """
    if not BLENDER_AVAILABLE:
        print("Warning: Blender API not available, skipping vertex refinement")
        return mesh_obj

    if front_silhouette is None and side_silhouette is None:
        print("Warning: No silhouettes provided, skipping vertex refinement")
        return mesh_obj

    print(f"\n=== Vertex-Level Refinement (Iteration 3) ===")
    print(f"Subdivision levels: {subdivision_levels}")

    # Phase 1: Apply subdivision to increase vertex count
    print(f"\nPhase 1: Subdivision")
    print(f"  Initial vertices: {len(mesh_obj.data.vertices)}")

    # Add subdivision modifier
    subsurf = mesh_obj.modifiers.new(name="Subdivision", type='SUBSURF')
    subsurf.levels = subdivision_levels
    subsurf.render_levels = subdivision_levels
    subsurf.subdivision_type = 'CATMULL_CLARK'

    # Apply modifier
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.modifier_apply(modifier=subsurf.name)

    print(f"  After subdivision: {len(mesh_obj.data.vertices)} vertices")

    # Phase 2: Extract vertical profiles from silhouettes
    print(f"\nPhase 2: Profile Extraction")

    from .profile_extractor import extract_vertical_profile

    profiles = {}
    if front_silhouette is not None:
        try:
            front_profile = extract_vertical_profile(front_silhouette, num_samples=100)
            profiles['front'] = front_profile
            radii = [r for h, r in front_profile]
            print(f"  Front profile: {len(front_profile)} samples, radius range {min(radii):.3f}-{max(radii):.3f}")
        except Exception as e:
            print(f"  Warning: Could not extract front profile: {e}")

    if side_silhouette is not None:
        try:
            side_profile = extract_vertical_profile(side_silhouette, num_samples=100)
            profiles['side'] = side_profile
            radii = [r for h, r in side_profile]
            print(f"  Side profile: {len(side_profile)} samples, radius range {min(radii):.3f}-{max(radii):.3f}")
        except Exception as e:
            print(f"  Warning: Could not extract side profile: {e}")

    if not profiles:
        print("  No profiles extracted, skipping vertex repositioning")
        return mesh_obj

    # Phase 3: Reposition vertices to match profiles
    print(f"\nPhase 3: Vertex Repositioning (Per-Axis)")
    if 'front' in profiles and 'side' in profiles:
        print(f"  Using directional profiles: Front→X-axis, Side→Y-axis")
    elif 'front' in profiles:
        print(f"  Using front profile for both X and Y axes (circular)")
    elif 'side' in profiles:
        print(f"  Using side profile for both X and Y axes (circular)")

    # Get mesh bounds for Z normalization
    mesh = mesh_obj.data
    vertices = mesh.vertices

    if len(vertices) == 0:
        print("  Warning: Mesh has no vertices")
        return mesh_obj

    # Calculate Z bounds
    z_coords = [v.co.z for v in vertices]
    z_min = min(z_coords)
    z_max = max(z_coords)
    z_range = z_max - z_min

    if z_range == 0:
        print("  Warning: Mesh has zero height")
        return mesh_obj

    print(f"  Z range: {z_min:.3f} to {z_max:.3f} (height: {z_range:.3f})")

    # Calculate maximum radial distance for uniform scaling reference
    # We'll use this as a common scale for both X and Y profiles
    max_radius = max(math.sqrt(v.co.x**2 + v.co.y**2) for v in vertices)
    print(f"  Max mesh radius (for profile scaling): {max_radius:.3f}")

    # Reposition each vertex
    adjusted_count = 0

    for vertex in vertices:
        # Get vertex position
        x, y, z = vertex.co

        # Normalize Z to 0-1 range
        z_normalized = (z - z_min) / z_range

        # Current radius from origin
        current_radius = math.sqrt(x*x + y*y)

        # Handle center vertices (zero radius)
        if current_radius < 0.001:
            continue

        # Apply profiles directionally:
        # - Front view (X-Z plane) controls X-axis width
        # - Side view (Y-Z plane) controls Y-axis width
        # Profile values are 0-1 normalized, need to scale to mesh dimensions

        # Calculate current X and Y distances from center (not combined radius)
        current_x_dist = abs(x)
        current_y_dist = abs(y)

        # Get target X distance from front profile
        x_scale_factor = 1.0
        if 'front' in profiles and current_x_dist > 0.001:
            front_radius_normalized = interpolate_profile(profiles['front'], z_normalized)
            # Front profile controls X-axis: scale by max_radius (common reference)
            target_x_dist = front_radius_normalized * max_radius
            x_scale_factor = target_x_dist / current_x_dist

        # Get target Y distance from side profile
        y_scale_factor = 1.0
        if 'side' in profiles and current_y_dist > 0.001:
            side_radius_normalized = interpolate_profile(profiles['side'], z_normalized)
            # Side profile controls Y-axis: scale by max_radius (common reference)
            target_y_dist = side_radius_normalized * max_radius
            y_scale_factor = target_y_dist / current_y_dist

        # If only one profile available, use it for both axes (fallback to circular)
        if 'front' in profiles and 'side' not in profiles:
            # Only front available: use it for both X and Y
            y_scale_factor = x_scale_factor
        elif 'side' in profiles and 'front' not in profiles:
            # Only side available: use it for both X and Y
            x_scale_factor = y_scale_factor

        # Apply directional adjustments (X, Y separately, preserve Z)
        vertex.co.x = x * x_scale_factor
        vertex.co.y = y * y_scale_factor
        # vertex.co.z stays unchanged

        adjusted_count += 1

    # Update mesh
    mesh.update()

    print(f"  Adjusted {adjusted_count} vertices")
    print(f"=== Refinement Complete ===\n")

    return mesh_obj
