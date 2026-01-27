"""
Quick test for Residual Primitive Fitting (ResFit) algorithm.

Simplified tests to verify basic functionality without expensive optimization.
"""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from placement.resfitting import ResidualFitter
from primitives.superfrustum import SuperFrustum


def test_basic_instantiation() -> bool:
    """Test basic ResidualFitter creation."""
    print("\n" + "=" * 70)
    print("TEST: Basic Instantiation")
    print("=" * 70)

    fitter = ResidualFitter(
        max_primitives=10,
        max_iterations=5,
        error_threshold=0.01,
        learning_rate=0.01,
        optimization_steps=10,
    )

    print(f"  ✓ Created fitter with:")
    print(f"    - max_primitives: {fitter.max_primitives}")
    print(f"    - max_iterations: {fitter.max_iterations}")
    print(f"    - error_threshold: {fitter.error_threshold}")
    print(f"    - learning_rate: {fitter.learning_rate}")
    print(f"    - optimization_steps: {fitter.optimization_steps}")

    assert fitter.max_primitives == 10
    assert len(fitter.primitives) == 0
    assert len(fitter.errors) == 0

    print("\n✓ Basic instantiation test PASSED")
    return True


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
            {
                "center": np.array([0.0, 0.0, z]),
                "radius": 1.0 - i * 0.05,  # Slightly tapered
                "scale": np.array([1.0, 1.0, 0.5]),
            }
        )

    fitter = ResidualFitter()
    primitives = fitter.initialize_from_slices(slice_data, num_initial=3)

    print(f"  ✓ Initialized {len(primitives)} primitives from {len(slice_data)} slices")
    assert len(primitives) == 3

    for i, sf in enumerate(primitives):
        print(
            f"    Primitive {i}: pos={sf.position}, r_bot={sf.radius_bottom:.2f}, r_top={sf.radius_top:.2f}"
        )

    print("\n✓ Initialization from slices test PASSED")
    return True


def test_residual_computation() -> bool:
    """Test residual error computation (fast, minimal points)."""
    print("\n" + "=" * 70)
    print("TEST: Residual Error Computation")
    print("=" * 70)

    # Create simple test points (cylinder surface)
    np.random.seed(1337)
    target_points = []
    for i in range(20):  # Small number for speed
        theta = i * 2 * np.pi / 20
        z = np.random.uniform(-1, 1)
        x = np.cos(theta)
        y = np.sin(theta)
        target_points.append([x, y, z])

    target_points = np.array(target_points)

    # Create primitive
    primitive = SuperFrustum(
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0),
        radius_bottom=1.0,
        radius_top=1.0,
        height=2.0,
    )

    # Compute residual
    fitter = ResidualFitter()
    total_error, per_point_errors = fitter.compute_residual_error(
        [primitive], target_points
    )

    print(f"  Total error: {total_error:.6f}")
    print(f"  Points tested: {len(target_points)}")
    print(
        f"  Error range: [{per_point_errors.min():.6f}, {per_point_errors.max():.6f}]"
    )

    assert total_error >= 0
    assert len(per_point_errors) == len(target_points)

    print("\n✓ Residual computation test PASSED")
    return True


def test_quick_optimization() -> bool:
    """Test quick optimization (few steps, few points)."""
    print("\n" + "=" * 70)
    print("TEST: Quick Optimization")
    print("=" * 70)

    # Small set of cylinder points
    np.random.seed(1337)
    target_points = []
    for i in range(30):
        theta = i * 2 * np.pi / 30
        z = np.random.uniform(-1.5, 1.5)
        x = 1.5 * np.cos(theta)
        y = 1.5 * np.sin(theta)
        target_points.append([x, y, z])

    target_points = np.array(target_points)

    # Slightly off primitive
    primitive = SuperFrustum(
        position=(0.1, 0.1, 0.0),
        orientation=(0.0, 0.0),
        radius_bottom=1.0,
        radius_top=1.0,
        height=2.5,
    )

    # Initial error
    initial_error = np.mean([abs(primitive.sdf(p)) for p in target_points[:10]])
    print(f"  Initial error (sample): {initial_error:.6f}")

    # Quick optimization
    fitter = ResidualFitter(learning_rate=0.05, optimization_steps=5)
    optimized = fitter.optimize_primitives([primitive], target_points, steps=5)

    # Final error
    final_error = np.mean([abs(optimized[0].sdf(p)) for p in target_points[:10]])
    print(f"  Final error (sample): {final_error:.6f}")

    print(
        f"  Optimized primitive: pos={optimized[0].position}, r={optimized[0].radius_bottom:.2f}"
    )

    # Just verify it runs, don't require improvement (5 steps might not be enough)
    assert len(optimized) == 1

    print("\n✓ Quick optimization test PASSED")
    return True


def test_vectorized_vs_scalar_step() -> bool:
    """Ensure vectorized and scalar paths stay consistent for one step."""
    print("\n" + "=" * 70)
    print("TEST: Vectorized vs Scalar Optimization (1 step)")
    print("=" * 70)

    rng = np.random.default_rng(2025)
    target_points = rng.normal(size=(50, 3))

    base_primitives = [
        SuperFrustum(
            position=(0.2, -0.1, 0.05),
            orientation=(0.1, 0.2),
            radius_bottom=1.1,
            radius_top=0.9,
            height=2.0,
        ),
        SuperFrustum(
            position=(-0.15, 0.2, -0.05),
            orientation=(0.3, 0.4),
            radius_bottom=0.8,
            radius_top=0.6,
            height=1.5,
        ),
    ]

    def copy_primitives(primitives: list[SuperFrustum]) -> list[SuperFrustum]:
        return [SuperFrustum.from_dict(sf.to_dict()) for sf in primitives]

    fitter = ResidualFitter(learning_rate=0.02, optimization_steps=1)
    vec_primitives = copy_primitives(base_primitives)
    scalar_primitives = copy_primitives(base_primitives)

    fitter.optimize_primitives(
        vec_primitives, target_points, steps=1, use_vectorized=True
    )
    fitter.optimize_primitives(
        scalar_primitives, target_points, steps=1, use_vectorized=False
    )

    for vec, scalar in zip(vec_primitives, scalar_primitives):
        np.testing.assert_allclose(vec.position, scalar.position, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(
            vec.orientation, scalar.orientation, rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            vec.radius_bottom, scalar.radius_bottom, rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            vec.radius_top, scalar.radius_top, rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(vec.height, scalar.height, rtol=1e-4, atol=1e-5)

    print("\n✓ Vectorized vs scalar step test PASSED")
    return True


def test_integration_check() -> bool:
    """Verify all components integrate without errors."""
    print("\n" + "=" * 70)
    print("TEST: Integration Check")
    print("=" * 70)

    # Create tiny point set
    target_points = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )

    # Very quick fit
    fitter = ResidualFitter(
        max_primitives=2,
        max_iterations=2,
        error_threshold=0.01,
        learning_rate=0.05,
        optimization_steps=3,
    )

    primitives = fitter.fit(target_points, num_initial=1, verbose=True)

    history = fitter.get_history()

    print(f"\n  Results:")
    print(f"    Primitives fitted: {len(primitives)}")
    print(f"    Iterations: {history['iterations']}")
    print(f"    Final error: {history['final_error']:.6f}")

    assert len(primitives) > 0
    assert history["iterations"] > 0

    print("\n✓ Integration check test PASSED")
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
    slice_data = [
        {"center": np.array([0.0, 0.0, 0.0]), "radius": 1.0, "scale": np.ones(3)}
    ]

    try:
        fitter.initialize_from_slices(slice_data, num_initial=0)
    except ValueError as exc:
        assert "num_initial" in str(exc)
        print("\n✓ num_initial=0 rejected as expected")
        return True

    raise AssertionError("Expected ValueError for num_initial=0")


def run_quick_tests() -> int:
    """Run all quick tests."""
    print("\n" + "=" * 70)
    print("RESFITTING QUICK TEST SUITE")
    print("=" * 70)

    tests = [
        ("Basic Instantiation", test_basic_instantiation),
        ("Initialization from Slices", test_initialization_from_slices),
        ("Residual Computation", test_residual_computation),
        ("Quick Optimization", test_quick_optimization),
        ("Vectorized vs Scalar Step", test_vectorized_vs_scalar_step),
        ("Integration Check", test_integration_check),
        ("Reject Empty Target Points", test_rejects_empty_target_points),
        ("Reject num_initial=0", test_rejects_zero_num_initial),
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
    exit_code = run_quick_tests()
    sys.exit(exit_code)
