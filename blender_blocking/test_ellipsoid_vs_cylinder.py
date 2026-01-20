"""
Test script comparing ellipsoid vs cylinder primitive selection.

This script creates two blockout meshes side-by-side:
1. Using CYLINDER primitives (current approach)
2. Using ELLIPSOID primitives (experimental approach)

The goal is to evaluate which primitive type produces better results
for reducing stepping artifacts in the stacked reconstruction.

Usage:
    Run in Blender:
    blender --background --python test_ellipsoid_vs_cylinder.py

    Or with a reference image:
    blender --background --python test_ellipsoid_vs_cylinder.py -- --front path/to/front.png
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import bpy
from mathutils import Vector
from placement.primitive_placement import SliceAnalyzer, PrimitivePlacer, MeshJoiner, clear_scene
from integration.blender_ops.scene_setup import add_camera, add_lighting


def create_comparison_blockouts(bounds_min, bounds_max, num_slices=12, vertical_profile=None, spacing=10.0):
    """
    Create two blockout meshes side-by-side for comparison.

    Args:
        bounds_min: Minimum bounds (x, y, z) for the shape
        bounds_max: Maximum bounds (x, y, z) for the shape
        num_slices: Number of vertical slices to use
        vertical_profile: Optional vertical profile data from image analysis
        spacing: Distance between the two meshes

    Returns:
        Tuple of (cylinder_mesh, ellipsoid_mesh)
    """
    print("="*70)
    print("ELLIPSOID vs CYLINDER COMPARISON TEST")
    print("="*70)

    # Clear scene
    clear_scene()

    # Create analyzer for slice data
    print(f"\nAnalyzing {num_slices} slices...")
    analyzer = SliceAnalyzer(
        bounds_min, bounds_max,
        num_slices=num_slices,
        vertical_profile=vertical_profile
    )
    slice_data = analyzer.get_all_slice_data()

    print(f"  Slice data generated: {len(slice_data)} slices")

    # ========================================================================
    # Create CYLINDER-based blockout (left side)
    # ========================================================================
    print("\n" + "-"*70)
    print("Creating CYLINDER-based blockout (LEFT)")
    print("-"*70)

    placer_cylinder = PrimitivePlacer()

    # Offset slice data for left placement
    slice_data_left = []
    for data in slice_data:
        data_copy = data.copy()
        data_copy['center'] = data['center'].copy()
        data_copy['center'].x -= spacing / 2
        slice_data_left.append(data_copy)

    objects_cylinder = placer_cylinder.place_primitives_from_slices(
        slice_data_left,
        primitive_type='CYLINDER'
    )
    print(f"  Placed {len(objects_cylinder)} cylinder primitives")

    # Join cylinders
    print("  Joining cylinder primitives...")
    joiner_cylinder = MeshJoiner()
    cylinder_mesh = joiner_cylinder.join_with_boolean_union(
        objects_cylinder,
        target_name="Cylinder_Blockout"
    )
    print(f"  ✓ Created: {cylinder_mesh.name}")

    # ========================================================================
    # Create ELLIPSOID-based blockout (right side)
    # ========================================================================
    print("\n" + "-"*70)
    print("Creating ELLIPSOID-based blockout (RIGHT)")
    print("-"*70)

    placer_ellipsoid = PrimitivePlacer()

    # Offset slice data for right placement
    slice_data_right = []
    for data in slice_data:
        data_copy = data.copy()
        data_copy['center'] = data['center'].copy()
        data_copy['center'].x += spacing / 2
        slice_data_right.append(data_copy)

    objects_ellipsoid = placer_ellipsoid.place_primitives_from_slices(
        slice_data_right,
        primitive_type='ELLIPSOID'
    )
    print(f"  Placed {len(objects_ellipsoid)} ellipsoid primitives")

    # Join ellipsoids
    print("  Joining ellipsoid primitives...")
    joiner_ellipsoid = MeshJoiner()
    ellipsoid_mesh = joiner_ellipsoid.join_with_boolean_union(
        objects_ellipsoid,
        target_name="Ellipsoid_Blockout"
    )
    print(f"  ✓ Created: {ellipsoid_mesh.name}")

    # ========================================================================
    # Setup scene for viewing
    # ========================================================================
    print("\n" + "-"*70)
    print("Setting up scene")
    print("-"*70)

    # Add camera and lighting
    camera = add_camera()
    # Position camera to view both objects
    camera.location = (0, -20, 5)
    camera.rotation_euler = (1.2, 0, 0)

    add_lighting()

    print("  ✓ Camera and lighting configured")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print(f"\nLeft  (CYLINDER):  {cylinder_mesh.name}")
    print(f"  - Vertices: {len(cylinder_mesh.data.vertices)}")
    print(f"  - Faces: {len(cylinder_mesh.data.polygons)}")
    print(f"\nRight (ELLIPSOID): {ellipsoid_mesh.name}")
    print(f"  - Vertices: {len(ellipsoid_mesh.data.vertices)}")
    print(f"  - Faces: {len(ellipsoid_mesh.data.polygons)}")
    print("\n" + "="*70)
    print("\nVISUAL COMPARISON:")
    print("  - LEFT:  Cylinder-based (current approach)")
    print("  - RIGHT: Ellipsoid-based (experimental)")
    print("\nLook for:")
    print("  • Stepping/banding artifacts")
    print("  • Surface smoothness")
    print("  • Silhouette quality")
    print("  • Boolean operation quality")
    print("="*70 + "\n")

    return cylinder_mesh, ellipsoid_mesh


def test_with_procedural_shape():
    """
    Test with a procedurally defined shape (no reference image).
    Uses a simple tapered cylinder profile.
    """
    print("\nTEST MODE: Procedural shape (tapered cylinder)")
    print("-"*70)

    # Define bounds
    bounds_min = (-2, -2, 0)
    bounds_max = (2, 2, 6)

    # Define a simple vertical profile (tapered cylinder)
    # Format: list of (height_normalized, radius_factor) tuples
    vertical_profile = [
        (0.0, 1.0),   # Bottom: full radius
        (0.2, 0.95),
        (0.4, 0.85),
        (0.6, 0.75),
        (0.8, 0.6),
        (1.0, 0.4),   # Top: narrow
    ]

    print("  Profile defined: Tapered from base to tip")
    print(f"  Bounds: {bounds_min} to {bounds_max}")

    return create_comparison_blockouts(
        bounds_min,
        bounds_max,
        num_slices=12,
        vertical_profile=vertical_profile,
        spacing=10.0
    )


def test_with_reference_image(image_path):
    """
    Test with a reference image for realistic shape.

    Args:
        image_path: Path to front or side view reference image
    """
    print(f"\nTEST MODE: Reference image - {image_path}")
    print("-"*70)

    from integration.shape_matching.profile_extractor import (
        extract_vertical_profile,
        extract_silhouette_from_image
    )
    from integration.image_processing.image_loader import load_image
    import cv2

    # Load image
    image = load_image(image_path)
    print(f"  Loaded image: {image.shape}")

    # Extract silhouette and profile
    silhouette = extract_silhouette_from_image(image)
    vertical_profile = extract_vertical_profile(image, num_samples=12)

    print(f"  Extracted vertical profile: {len(vertical_profile)} samples")

    # Calculate bounds from silhouette
    contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        print(f"  Silhouette bbox: x={x}, y={y}, w={w}, h={h}")

        scale = 0.01  # 1 pixel = 0.01 units
        width = w * scale
        height = h * scale

        bounds_min = (-width/2, -width/2, 0)
        bounds_max = (width/2, width/2, height)
        print(f"  Calculated bounds: {bounds_min} to {bounds_max}")
    else:
        print("  Warning: No silhouette found, using default bounds")
        bounds_min = (-2, -2, 0)
        bounds_max = (2, 2, 6)

    return create_comparison_blockouts(
        bounds_min,
        bounds_max,
        num_slices=12,
        vertical_profile=vertical_profile,
        spacing=10.0
    )


def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(
        description="Compare ellipsoid vs cylinder primitive selection"
    )
    parser.add_argument(
        '--front',
        type=str,
        help='Path to front view reference image'
    )
    parser.add_argument(
        '--slices',
        type=int,
        default=12,
        help='Number of vertical slices (default: 12)'
    )

    # Parse args (skip Blender's args)
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        argv = []

    args = parser.parse_args(argv)

    # Run test
    if args.front:
        test_with_reference_image(args.front)
    else:
        test_with_procedural_shape()

    print("\n✓ Test complete. Review the meshes in Blender viewport.")
    print("  To save: File > Save As... or use Blender's render tools\n")


if __name__ == "__main__":
    main()
