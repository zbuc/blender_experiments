"""
Test suite for Residual Primitive Fitting (ResFit) algorithm.

Tests:
1. Initialization from slices
2. Initialization from voxels
3. Primitive optimization
4. Residual error computation
5. Adaptive primitive addition
6. Full fitting workflow
"""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from placement.resfitting import ResidualFitter
from primitives.superfrustum import SuperFrustum


def generate_cylinder_points(
    radius: float = 1.0, height: float = 2.0, num_points: int = 1000
) -> np.ndarray:
    """Generate points on a cylinder surface for testing."""
    points = []

    # Side surface
    for _ in range(int(num_points * 0.8)):
        theta = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(-height / 2, height / 2)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append([x, y, z])

    # Top and bottom caps
    for _ in range(int(num_points * 0.2)):
        r = np.random.uniform(0, radius)
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.choice([-height / 2, height / 2])
        points.append([x, y, z])

    return np.array(points)


def generate_cone_points(
    radius_bottom: float = 1.5,
    radius_top: float = 0.0,
    height: float = 2.0,
    num_points: int = 1000,
) -> np.ndarray:
    """Generate points on a cone surface for testing."""
    points = []

    for _ in range(num_points):
        # Random height
        t = np.random.uniform(0, 1)
        z = -height / 2 + t * height

        # Radius at this height (linear interpolation)
        r = radius_bottom * (1 - t) + radius_top * t

        # Random angle
        theta = np.random.uniform(0, 2 * np.pi)

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, z])

    return np.array(points)


def test_initialization_from_slices() -> bool:
    """Test primitive initialization from slice data."""
    print("\n" + "=" * 70)
    print("TEST: Initialization from Slices")
    print("=" * 70)

    # Create mock slice data
    slice_data = []
    for i in range(10):
        z = i * 0.5
        slice_data.append(
            {"center": [0.0, 0.0, z], "radius": 1.0, "scale": [1.0, 1.0, 0.5]}
        )

    # Initialize fitter
    fitter = ResidualFitter()

    # Initialize primitives
    primitives = fitter.initialize_from_slices(slice_data, num_initial=3)

    print(f"  ✓ Initialized {len(primitives)} primitives from {len(slice_data)} slices")

    for i, sf in enumerate(primitives):
        print(f"    Primitive {i}: {sf}")

    assert len(primitives) == 3, "Should initialize 3 primitives"
    assert all(
        isinstance(sf, SuperFrustum) for sf in primitives
    ), "All should be SuperFrustum"

    print("\n✓ Initialization from slices test PASSED")
    return True


def test_initialization_from_voxels() -> bool:
    """Test primitive initialization from voxel grid."""
    print("\n" + "=" * 70)
    print("TEST: Initialization from Voxels")
    print("=" * 70)

    # Create simple voxel grid (cylinder-like shape)
    grid_size = 32
    voxel_grid = np.zeros((grid_size, grid_size, grid_size))

    # Fill cylinder-like region
    center = grid_size // 2
    radius = grid_size // 4

    for z in range(grid_size):
        for y in range(grid_size):
            for x in range(grid_size):
                dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
                if dist < radius:
                    voxel_grid[x, y, z] = 1.0

    # Initialize fitter
    fitter = ResidualFitter()

    # Initialize primitives
    primitives = fitter.initialize_from_voxels(voxel_grid, num_initial=4)

    print(f"  ✓ Initialized {len(primitives)} primitives from voxel grid")

    for i, sf in enumerate(primitives):
        print(f"    Primitive {i}: {sf}")

    assert len(primitives) > 0, "Should initialize at least one primitive"
    assert all(
        isinstance(sf, SuperFrustum) for sf in primitives
    ), "All should be SuperFrustum"

    print("\n✓ Initialization from voxels test PASSED")
    return True


def test_rejects_empty_target_points() -> bool:
    """Ensure empty target points raise a clear error."""
    print("\n" + "=" * 70)
    print("TEST: Reject Empty Target Points")
    print("=" * 70)

    fitter = ResidualFitter()
    empty_points = np.empty((0, 3))

    try:
        fitter.fit(empty_points, num_initial=1, verbose=False)
    except ValueError as exc:
        assert "target_points" in str(exc)
        print("\n✓ Empty target points rejected as expected")
        return True

    raise AssertionError("Expected ValueError for empty target_points")


def test_rejects_zero_num_initial() -> bool:
    """Ensure num_initial == 0 raises a clear error."""
    print("\n" + "=" * 70)
    print("TEST: Reject num_initial=0")
    print("=" * 70)

    fitter = ResidualFitter()
    slice_data = [{"center": [0.0, 0.0, 0.0], "radius": 1.0, "scale": [1.0, 1.0, 0.5]}]

    try:
        fitter.initialize_from_slices(slice_data, num_initial=0)
    except ValueError as exc:
        assert "num_initial" in str(exc)
        print("\n✓ num_initial=0 rejected as expected")
        return True

    raise AssertionError("Expected ValueError for num_initial=0")


def test_optimization() -> bool:
    """Test primitive optimization on simple cylinder."""
    print("\n" + "=" * 70)
    print("TEST: Primitive Optimization")
    print("=" * 70)

    # Generate cylinder points
    np.random.seed(1337)
    target_points = generate_cylinder_points(radius=1.5, height=3.0, num_points=500)
    print(f"  Generated {len(target_points)} target points (cylinder r=1.5, h=3.0)")

    # Create initial primitive (slightly off)
    initial_sf = SuperFrustum(
        position=(0.2, 0.2, 0.1),  # Offset position
        orientation=(0.0, 0.0),
        radius_bottom=1.0,  # Smaller radius
        radius_top=1.0,
        height=2.5,  # Shorter height
    )

    print(f"  Initial primitive: {initial_sf}")

    # Optimize
    initial_error = np.mean([abs(initial_sf.sdf(p)) for p in target_points[:100]])

    fitter = ResidualFitter(learning_rate=0.01, optimization_steps=30)
    optimized = fitter.optimize_primitives([initial_sf], target_points, steps=30)

    print(f"  Optimized primitive: {optimized[0]}")

    # Verify optimization improved the fit
    final_error = np.mean([abs(optimized[0].sdf(p)) for p in target_points[:100]])

    print(f"\n  Initial error: {initial_error:.6f}")
    print(f"  Final error: {final_error:.6f}")
    print(
        f"  Improvement: {((initial_error - final_error) / initial_error * 100):.1f}%"
    )

    assert final_error < initial_error, "Optimization should reduce error"

    print("\n✓ Optimization test PASSED")
    return True


def test_residual_computation() -> bool:
    """Test residual error computation."""
    print("\n" + "=" * 70)
    print("TEST: Residual Error Computation")
    print("=" * 70)

    # Generate cone points
    np.random.seed(1337)
    target_points = generate_cone_points(radius_bottom=2.0, height=3.0, num_points=500)

    # Create primitive that partially fits
    primitive = SuperFrustum(
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0),
        radius_bottom=1.5,  # Smaller than actual
        radius_top=0.0,
        height=2.5,  # Shorter
    )

    # Compute residual
    fitter = ResidualFitter()
    total_error, per_point_errors = fitter.compute_residual_error(
        [primitive], target_points
    )

    print(f"  Total error: {total_error:.6f}")
    print(f"  Min point error: {per_point_errors.min():.6f}")
    print(f"  Max point error: {per_point_errors.max():.6f}")
    print(f"  Median point error: {np.median(per_point_errors):.6f}")

    assert total_error > 0, "Should have non-zero error"
    assert len(per_point_errors) == len(
        target_points
    ), "Should have error for each point"

    print("\n✓ Residual computation test PASSED")
    return True


def test_primitive_addition() -> bool:
    """Test adaptive primitive addition in high-error regions."""
    print("\n" + "=" * 70)
    print("TEST: Adaptive Primitive Addition")
    print("=" * 70)

    # Generate two-segment shape (bottom cylinder + top cone)
    np.random.seed(1337)
    bottom_points = generate_cylinder_points(radius=1.5, height=2.0, num_points=300)
    bottom_points[:, 2] -= 2.0  # Shift down

    top_points = generate_cone_points(
        radius_bottom=1.5, radius_top=0.5, height=2.0, num_points=200
    )
    top_points[:, 2] += 1.0  # Shift up

    target_points = np.vstack([bottom_points, top_points])

    print(f"  Generated {len(target_points)} points (2-segment shape)")

    # Create primitive that only fits bottom
    primitive = SuperFrustum(
        position=(0.0, 0.0, -2.0),
        orientation=(0.0, 0.0),
        radius_bottom=1.5,
        radius_top=1.5,
        height=2.0,
    )

    # Compute residual
    fitter = ResidualFitter(max_primitives=5)
    total_error, per_point_errors = fitter.compute_residual_error(
        [primitive], target_points
    )

    print(f"  Initial error with 1 primitive: {total_error:.6f}")

    # Add primitive in high-error region
    new_primitive = fitter.add_primitive_at_error_region(
        [primitive], target_points, per_point_errors
    )

    assert new_primitive is not None, "Should add new primitive"
    print(f"  ✓ Added new primitive: {new_primitive}")

    # Verify new primitive reduces error
    new_error, _ = fitter.compute_residual_error(
        [primitive, new_primitive], target_points
    )

    print(f"  Error with 2 primitives: {new_error:.6f}")
    print(f"  Error reduction: {((total_error - new_error) / total_error * 100):.1f}%")

    assert new_error < total_error, "Adding primitive should reduce error"

    print("\n✓ Primitive addition test PASSED")
    return True


def test_full_fitting_workflow() -> bool:
    """Test complete ResFit workflow on a simple shape."""
    print("\n" + "=" * 70)
    print("TEST: Full ResFit Workflow")
    print("=" * 70)

    # Generate cone shape
    np.random.seed(1337)
    target_points = generate_cone_points(
        radius_bottom=2.0, radius_top=0.5, height=3.0, num_points=1000
    )

    print(f"  Target shape: cone (r_bottom=2.0, r_top=0.5, h=3.0)")
    print(f"  Target points: {len(target_points)}")

    # Run ResFit
    fitter = ResidualFitter(
        max_primitives=5,
        max_iterations=5,
        error_threshold=0.05,
        learning_rate=0.02,
        optimization_steps=20,
    )

    primitives = fitter.fit(target_points, num_initial=2, verbose=True)

    # Get history
    history = fitter.get_history()

    print(f"\n  Final results:")
    print(f"    Primitives: {history['num_primitives']}")
    print(f"    Iterations: {history['iterations']}")
    print(f"    Final error: {history['final_error']:.6f}")
    print(f"    Error history: {[f'{e:.6f}' for e in history['errors']]}")

    assert len(primitives) > 0, "Should fit at least one primitive"
    assert history["final_error"] is not None, "Should have final error"
    assert len(history["errors"]) > 0, "Should have error history"

    # Verify error decreased
    if len(history["errors"]) > 1:
        assert (
            history["errors"][-1] <= history["errors"][0]
        ), "Error should decrease or stay same"

    print("\n✓ Full fitting workflow test PASSED")
    return True


def run_all_tests() -> int:
    """Run all test cases."""
    print("\n" + "=" * 70)
    print("RESIDUAL PRIMITIVE FITTING (ResFit) TEST SUITE")
    print("=" * 70)

    tests = [
        ("Initialization from Slices", test_initialization_from_slices),
        ("Initialization from Voxels", test_initialization_from_voxels),
        ("Reject Empty Target Points", test_rejects_empty_target_points),
        ("Reject num_initial=0", test_rejects_zero_num_initial),
        ("Primitive Optimization", test_optimization),
        ("Residual Error Computation", test_residual_computation),
        ("Adaptive Primitive Addition", test_primitive_addition),
        ("Full ResFit Workflow", test_full_fitting_workflow),
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
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)

    for name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {name:.<50} {status}")

    print("=" * 70)
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
