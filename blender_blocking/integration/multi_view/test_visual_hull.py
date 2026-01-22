"""
Test suite for Multi-View Visual Hull reconstruction.

Tests:
1. Basic instantiation
2. Single view reconstruction
3. 3-view reconstruction (baseline)
4. 8-view reconstruction
5. 12-view reconstruction
6. Camera projection accuracy
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from integration.multi_view.visual_hull import MultiViewVisualHull, CameraView


def generate_cylinder_silhouette(
    width: int = 128,
    height: int = 128,
    radius_ratio: float = 0.4,
    angle: float = 0.0
) -> np.ndarray:
    """
    Generate synthetic cylinder silhouette at given angle.

    For turntable views, cylinder appears as rectangle (constant width).

    Args:
        width: Image width
        height: Image height
        radius_ratio: Cylinder radius as fraction of image size
        angle: Viewing angle (for lateral views, doesn't affect cylinder)

    Returns:
        Binary silhouette (HxW)
    """
    silhouette = np.zeros((height, width), dtype=bool)

    # Cylinder appears as vertical rectangle in lateral view
    radius_pixels = int(width * radius_ratio)
    center_x = width // 2

    # Fill rectangle
    x_min = center_x - radius_pixels
    x_max = center_x + radius_pixels

    silhouette[:, max(0, x_min):min(width, x_max)] = True

    return silhouette


def generate_top_cylinder_silhouette(
    width: int = 128,
    height: int = 128,
    radius_ratio: float = 0.4
) -> np.ndarray:
    """
    Generate synthetic cylinder silhouette from top view (circular).

    Args:
        width: Image width
        height: Image height
        radius_ratio: Cylinder radius as fraction of image size

    Returns:
        Binary silhouette (HxW)
    """
    silhouette = np.zeros((height, width), dtype=bool)

    center_x = width // 2
    center_y = height // 2
    radius = int(width * radius_ratio)

    # Create circle
    y, x = np.ogrid[:height, :width]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    silhouette[mask] = True

    return silhouette


def test_basic_instantiation():
    """Test basic Visual Hull creation."""
    print("\n" + "="*70)
    print("TEST: Basic Instantiation")
    print("="*70)

    hull = MultiViewVisualHull(resolution=32)

    print(f"  ✓ Created Visual Hull")
    print(f"    Resolution: {hull.resolution}")
    print(f"    Bounds: {hull.bounds_min} to {hull.bounds_max}")

    assert hull.resolution == 32
    assert len(hull.views) == 0

    print("\n✓ Basic instantiation test PASSED")
    return True


def test_camera_view():
    """Test CameraView creation and projection."""
    print("\n" + "="*70)
    print("TEST: Camera View Creation")
    print("="*70)

    # Create simple silhouette (circle)
    silhouette = generate_top_cylinder_silhouette(64, 64, radius_ratio=0.3)

    view = CameraView(
        silhouette=silhouette,
        angle=0.0,
        view_type='top'
    )

    print(f"  ✓ Created camera view")
    print(f"    Angle: {view.angle}°")
    print(f"    Type: {view.view_type}")
    print(f"    Size: {view.width}x{view.height}")

    # Test projection
    test_point = np.array([0.0, 0.0, 0.0])  # Center point
    u, v = view.project_point(test_point, (np.array([-1, -1, -1]), np.array([1, 1, 1])))

    print(f"  ✓ Projection test: (0,0,0) -> ({u}, {v})")
    print(f"    Expected near: ({view.width//2}, {view.height//2})")

    # Should project near center
    assert abs(u - view.width//2) < 5
    assert abs(v - view.height//2) < 5

    print("\n✓ Camera view test PASSED")
    return True


def test_single_view_reconstruction():
    """Test reconstruction with single view (should give cone)."""
    print("\n" + "="*70)
    print("TEST: Single View Reconstruction")
    print("="*70)

    hull = MultiViewVisualHull(resolution=32)

    # Add single lateral view (cylinder from side)
    silhouette = generate_cylinder_silhouette(64, 64, radius_ratio=0.3)
    hull.add_view_from_silhouette(silhouette, angle=0.0, view_type='lateral')

    print(f"  Added 1 view")

    # Reconstruct
    voxel_grid = hull.reconstruct(verbose=False)

    print(f"  ✓ Reconstruction complete")

    stats = hull.get_stats()
    print(f"    Occupied voxels: {stats['occupied_voxels']:,}")
    print(f"    Occupancy: {stats['occupancy']*100:.2f}%")

    assert stats['num_views'] == 1
    assert stats['occupied_voxels'] > 0

    print("\n✓ Single view test PASSED")
    return True


def test_three_view_reconstruction():
    """Test reconstruction with 3 orthogonal views (baseline)."""
    print("\n" + "="*70)
    print("TEST: 3-View Reconstruction (Baseline)")
    print("="*70)

    hull = MultiViewVisualHull(resolution=32)

    # Add 3 orthogonal lateral views (0°, 90°, 180°)
    for angle in [0, 90, 180]:
        silhouette = generate_cylinder_silhouette(64, 64, radius_ratio=0.3)
        hull.add_view_from_silhouette(silhouette, angle=float(angle), view_type='lateral')

    # Add top view
    top_silhouette = generate_top_cylinder_silhouette(64, 64, radius_ratio=0.3)
    hull.add_view_from_silhouette(top_silhouette, view_type='top')

    print(f"  Added 4 views (3 lateral + 1 top)")

    # Reconstruct
    voxel_grid = hull.reconstruct(verbose=True)

    stats = hull.get_stats()
    print(f"\n  Final statistics:")
    print(f"    Occupied voxels: {stats['occupied_voxels']:,}")
    print(f"    Occupancy: {stats['occupancy']*100:.2f}%")

    assert stats['num_views'] == 4
    assert stats['occupied_voxels'] > 0

    # Extract surface points
    points = hull.extract_mesh_points(surface_only=True)
    print(f"    Surface points: {len(points):,}")

    print("\n✓ 3-view reconstruction test PASSED")
    return True


def test_eight_view_reconstruction():
    """Test reconstruction with 8 views (45° spacing)."""
    print("\n" + "="*70)
    print("TEST: 8-View Reconstruction")
    print("="*70)

    hull = MultiViewVisualHull(resolution=32)

    # Add 8 lateral views (45° spacing)
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    for angle in angles:
        silhouette = generate_cylinder_silhouette(64, 64, radius_ratio=0.3)
        hull.add_view_from_silhouette(silhouette, angle=float(angle), view_type='lateral')

    # Add top view
    top_silhouette = generate_top_cylinder_silhouette(64, 64, radius_ratio=0.3)
    hull.add_view_from_silhouette(top_silhouette, view_type='top')

    print(f"  Added 9 views (8 lateral at 45° + 1 top)")

    # Reconstruct
    voxel_grid = hull.reconstruct(verbose=True)

    stats = hull.get_stats()
    print(f"\n  Final statistics:")
    print(f"    Occupied voxels: {stats['occupied_voxels']:,}")
    print(f"    Occupancy: {stats['occupancy']*100:.2f}%")

    assert stats['num_views'] == 9
    assert stats['occupied_voxels'] > 0

    print("\n✓ 8-view reconstruction test PASSED")
    return True


def test_twelve_view_reconstruction():
    """Test reconstruction with 12 views (30° spacing) - target configuration."""
    print("\n" + "="*70)
    print("TEST: 12-View Reconstruction (Target Configuration)")
    print("="*70)

    hull = MultiViewVisualHull(resolution=32)

    # Add 12 lateral views (30° spacing)
    angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    for angle in angles:
        silhouette = generate_cylinder_silhouette(64, 64, radius_ratio=0.3)
        hull.add_view_from_silhouette(silhouette, angle=float(angle), view_type='lateral')

    # Add top view
    top_silhouette = generate_top_cylinder_silhouette(64, 64, radius_ratio=0.3)
    hull.add_view_from_silhouette(top_silhouette, view_type='top')

    print(f"  Added 13 views (12 lateral at 30° + 1 top)")

    # Reconstruct
    voxel_grid = hull.reconstruct(verbose=True)

    stats = hull.get_stats()
    print(f"\n  Final statistics:")
    print(f"    Occupied voxels: {stats['occupied_voxels']:,}")
    print(f"    Occupancy: {stats['occupancy']*100:.2f}%")

    assert stats['num_views'] == 13
    assert stats['occupied_voxels'] > 0

    # Extract points
    points = hull.extract_mesh_points(surface_only=True)
    print(f"    Surface points: {len(points):,}")

    print("\n✓ 12-view reconstruction test PASSED")
    return True


def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*70)
    print("MULTI-VIEW VISUAL HULL TEST SUITE")
    print("="*70)

    tests = [
        ("Basic Instantiation", test_basic_instantiation),
        ("Camera View", test_camera_view),
        ("Single View Reconstruction", test_single_view_reconstruction),
        ("3-View Reconstruction", test_three_view_reconstruction),
        ("8-View Reconstruction", test_eight_view_reconstruction),
        ("12-View Reconstruction", test_twelve_view_reconstruction),
    ]

    results = {}
    for name, test_func in tests:
        try:
            result = test_func()
            results[name] = result
        except Exception as e:
            print(f"\n❌ {name} FAILED with exception:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)

    for name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {name:.<50} {status}")

    print("="*70)
    print(f"\nResults: {passed} passed, {failed} failed")

    if failed > 0:
        print("\n❌ TESTS FAILED")
        return 1
    else:
        print("\n✓ ALL TESTS PASSED")
        return 0


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
