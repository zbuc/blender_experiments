"""
Example workflow: Multi-view (8-12 views) 3D reconstruction.

Demonstrates how to use the multi-view Visual Hull system for higher accuracy
reconstruction compared to the 3-view baseline.

Expected IoU:
- 3-view baseline: 0.875
- 8-view: 0.87-0.90
- 12-view: 0.88-0.92 (target)

Usage:
    # From within Blender:
    python example_multiview_workflow.py

    # Or run in Blender:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python example_multiview_workflow.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Add ~/blender_python_packages for Pillow (if in Blender)
sys.path.insert(0, str(Path.home() / "blender_python_packages"))

try:
    import bpy
    from mathutils import Vector
except ImportError:
    print("Warning: Running outside Blender. Some features unavailable.")
    bpy = None

import numpy as np
from integration.image_processing.image_loader import (
    load_multi_view_auto,
    load_multi_view_turntable,
    validate_view_coverage,
)
from integration.multi_view.visual_hull import MultiViewVisualHull, CameraView


def example_12view_synthetic() -> Tuple[MultiViewVisualHull, np.ndarray]:
    """
    Example 1: 12-view reconstruction with synthetic silhouettes.

    Demonstrates the complete pipeline with generated test data.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: 12-View Reconstruction (Synthetic)")
    print("=" * 70)

    # Create Visual Hull instance
    hull = MultiViewVisualHull(
        resolution=64,  # Lower resolution for quick demo
        bounds_min=np.array([-1.5, -1.5, -1.5]),
        bounds_max=np.array([1.5, 1.5, 1.5]),
    )

    print("\n1. Generating synthetic cylinder silhouettes...")

    # Generate 12 lateral views (30° spacing)
    angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

    for angle in angles:
        # Simple rectangular silhouette (cylinder from side)
        silhouette = np.zeros((128, 128), dtype=bool)
        silhouette[:, 40:88] = True  # Vertical rectangle

        hull.add_view_from_silhouette(
            silhouette, angle=float(angle), view_type="lateral"
        )

    # Add top view (circular silhouette)
    top_silhouette = np.zeros((128, 128), dtype=bool)
    center = 64
    radius = 24
    y, x = np.ogrid[:128, :128]
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius**2
    top_silhouette[mask] = True

    hull.add_view_from_silhouette(top_silhouette, view_type="top")

    print(f"   ✓ Added {len(hull.views)} views (12 lateral + 1 top)")

    # Reconstruct
    print("\n2. Running Visual Hull reconstruction...")
    voxel_grid = hull.reconstruct(verbose=True)

    # Get statistics
    stats = hull.get_stats()
    print("\n3. Reconstruction Statistics:")
    print(f"   Views: {stats['num_views']}")
    print(f"   Resolution: {stats['resolution']}³")
    print(f"   Occupied voxels: {stats['occupied_voxels']:,}")
    print(f"   Occupancy: {stats['occupancy']*100:.2f}%")

    # Extract surface points
    print("\n4. Extracting surface...")
    points = hull.extract_mesh_points(surface_only=True)
    print(f"   ✓ Extracted {len(points):,} surface points")

    print("\n✓ Example 1 complete")
    return hull, points


def example_load_from_directory() -> None:
    """
    Example 2: Load multi-view images from directory.

    Demonstrates auto-detection of turntable sequence.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Load Multi-View from Directory (Demo)")
    print("=" * 70)

    # This is a demonstration of the API
    # In practice, you would have actual image files

    print("\n1. Directory structure expected:")
    print("   my_scans/")
    print("     view_000.png   (0° lateral view)")
    print("     view_030.png   (30° lateral view)")
    print("     view_060.png   (60° lateral view)")
    print("     ...")
    print("     view_330.png   (330° lateral view)")
    print("     top.png        (top view)")

    print("\n2. Loading views:")
    print("   views = load_multi_view_auto('my_scans/', num_views=12)")

    print("\n3. View structure:")
    print("   views = {")
    print("     'lateral_0': (image_array, 0.0, 'lateral'),")
    print("     'lateral_1': (image_array, 30.0, 'lateral'),")
    print("     ...")
    print("     'top': (image_array, 0.0, 'top')")
    print("   }")

    print("\n4. Validate coverage:")
    print("   is_valid = validate_view_coverage(views, expected_spacing=30.0)")

    print("\n✓ Example 2 complete (demonstration only)")


def example_create_blender_mesh() -> None:
    """
    Example 3: Create Blender mesh from multi-view reconstruction.

    Demonstrates integration with Blender for mesh creation.
    """
    if bpy is None:
        print("\n" + "=" * 70)
        print("EXAMPLE 3: Blender Mesh Creation (SKIPPED - not in Blender)")
        print("=" * 70)
        print("\nRun this script in Blender to see mesh creation.")
        return

    print("\n" + "=" * 70)
    print("EXAMPLE 3: Create Blender Mesh from Multi-View")
    print("=" * 70)

    # Run synthetic reconstruction
    hull, points = example_12view_synthetic()

    print("\n5. Creating Blender mesh...")

    # Clear existing meshes
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Create mesh from points
    mesh = bpy.data.meshes.new("MultiView_Hull")
    obj = bpy.data.objects.new("MultiView_Hull", mesh)

    # Link to scene
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj

    # Create vertices
    mesh.from_pydata(points.tolist(), [], [])
    mesh.update()

    print(f"   ✓ Created Blender object: {obj.name}")
    print(f"   ✓ Vertices: {len(mesh.vertices)}")

    print("\n✓ Example 3 complete - mesh created in Blender")


def example_comparison_3view_vs_12view() -> None:
    """
    Example 4: Compare 3-view baseline vs 12-view.

    Demonstrates expected IoU improvement.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: 3-View vs 12-View Comparison")
    print("=" * 70)

    print("\n3-View Baseline:")
    print("  Views: front (0°), side (90°), top")
    print("  Total: 3 views")
    print("  Expected IoU: ~0.875")
    print("  Processing time: ~30s")

    print("\n12-View Enhanced:")
    print("  Views: lateral at 0°, 30°, 60°, ..., 330° + top")
    print("  Total: 13 views")
    print("  Expected IoU: 0.88-0.92")
    print("  Processing time: ~60-80s")

    print("\nImprovement:")
    print("  IoU gain: +0.005 to +0.045 (0.5% to 4.5%)")
    print("  Time cost: 2-2.7x slower")
    print("  Trade-off: Acceptable for high-quality scans")

    print("\n✓ Example 4 complete")


def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 70)
    print("MULTI-VIEW RECONSTRUCTION - EXAMPLE WORKFLOWS")
    print("=" * 70)
    print("\nBased on research: MULTI_VIEW_RECONSTRUCTION_RESEARCH.md")
    print("Target: 0.88-0.92 IoU with 12 views (vs 0.875 with 3 views)")

    # Run examples
    example_12view_synthetic()
    example_load_from_directory()
    example_comparison_3view_vs_12view()

    if bpy:
        example_create_blender_mesh()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Capture turntable images (12 views at 30° spacing)")
    print("  2. Use load_multi_view_auto() to load images")
    print("  3. Run MultiViewVisualHull.reconstruct()")
    print("  4. Extract mesh and import into Blender")
    print("  5. Validate IoU improvement over 3-view baseline")


if __name__ == "__main__":
    main()
