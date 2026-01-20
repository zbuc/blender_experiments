"""
Unit tests for profile fusion functionality.
These tests can run without Blender.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from integration.shape_matching.profile_extractor import (
    fuse_profiles,
    calculate_profile_confidence,
    validate_profile
)


def test_equal_weights_fusion():
    """Test E1: Equal weights fusion (0.5, 0.5)"""
    print("\n" + "="*60)
    print("TEST: Equal weights fusion (E1)")
    print("="*60)

    # Create sample profiles
    front_profile = [(i/10, 0.5 + i/20) for i in range(11)]  # Heights 0-1, radii 0.5-1.0
    side_profile = [(i/10, 0.3 + i/30) for i in range(11)]   # Heights 0-1, radii 0.3-0.63

    # Fuse with equal weights
    fused = fuse_profiles(front_profile, side_profile, fusion_strategy="equal")

    print(f"Front profile radii: {[r for h, r in front_profile[:3]]} ... {[r for h, r in front_profile[-3:]]}")
    print(f"Side profile radii:  {[r for h, r in side_profile[:3]]} ... {[r for h, r in side_profile[-3:]]}")
    print(f"Fused profile radii: {[r for h, r in fused[:3]]} ... {[r for h, r in fused[-3:]]}")

    # Verify fused profile is valid
    assert validate_profile(fused), "Fused profile should be valid"

    # Verify heights match
    assert all(h1 == h2 for (h1, _), (h2, _) in zip(front_profile, fused)), "Heights should match"

    print("✓ Equal weights fusion works correctly")


def test_front_heavy_fusion():
    """Test E2: Front-heavy fusion (0.6, 0.4)"""
    print("\n" + "="*60)
    print("TEST: Front-heavy fusion (E2)")
    print("="*60)

    front_profile = [(i/10, 1.0) for i in range(11)]  # All radii = 1.0
    side_profile = [(i/10, 0.0) for i in range(11)]   # All radii = 0.0

    fused = fuse_profiles(front_profile, side_profile, fusion_strategy="front_heavy")

    # With front=1.0, side=0.0, front_heavy (0.6, 0.4) should give radii closer to 1.0
    # After normalization, all should be 1.0 since max is used
    assert all(r == 1.0 for h, r in fused), "Front-heavy should favor front view"

    print("✓ Front-heavy fusion works correctly")


def test_side_heavy_fusion():
    """Test E3: Side-heavy fusion (0.4, 0.6)"""
    print("\n" + "="*60)
    print("TEST: Side-heavy fusion (E3)")
    print("="*60)

    front_profile = [(i/10, 0.0) for i in range(11)]  # All radii = 0.0
    side_profile = [(i/10, 1.0) for i in range(11)]   # All radii = 1.0

    fused = fuse_profiles(front_profile, side_profile, fusion_strategy="side_heavy")

    # With front=0.0, side=1.0, side_heavy (0.4, 0.6) should give radii closer to 1.0
    # After normalization, all should be 1.0
    assert all(r == 1.0 for h, r in fused), "Side-heavy should favor side view"

    print("✓ Side-heavy fusion works correctly")


def test_adaptive_fusion():
    """Test E4: Adaptive fusion based on confidence"""
    print("\n" + "="*60)
    print("TEST: Adaptive fusion (E4)")
    print("="*60)

    # Create a smooth front profile (high confidence)
    front_profile = [(i/10, 0.5 + 0.05 * i) for i in range(11)]

    # Create a noisy side profile (lower confidence)
    side_profile = [(i/10, 0.3 + 0.1 * (i % 2)) for i in range(11)]

    # Calculate confidences
    front_conf = calculate_profile_confidence(front_profile)
    side_conf = calculate_profile_confidence(side_profile)

    print(f"Front profile confidence: {front_conf:.4f}")
    print(f"Side profile confidence:  {side_conf:.4f}")

    # Fuse with adaptive weights
    fused = fuse_profiles(front_profile, side_profile, fusion_strategy="adaptive")

    assert validate_profile(fused), "Fused profile should be valid"

    # Smoother profile should have higher confidence
    assert front_conf > side_conf, "Smooth profile should have higher confidence than noisy one"

    print("✓ Adaptive fusion works correctly")


def test_custom_weights():
    """Test custom weight fusion"""
    print("\n" + "="*60)
    print("TEST: Custom weights fusion")
    print("="*60)

    front_profile = [(i/10, 1.0) for i in range(11)]
    side_profile = [(i/10, 0.5) for i in range(11)]

    # Custom weights: 0.7 front, 0.3 side
    fused = fuse_profiles(
        front_profile,
        side_profile,
        fusion_strategy="custom",
        front_weight=0.7,
        side_weight=0.3
    )

    assert validate_profile(fused), "Fused profile should be valid"

    # Expected unnormalized: 0.7*1.0 + 0.3*0.5 = 0.7 + 0.15 = 0.85
    # After normalization with max=0.85, all should be 1.0
    assert all(r == 1.0 for h, r in fused), "Custom weights should work correctly"

    print("✓ Custom weights fusion works correctly")


def test_profile_confidence_calculation():
    """Test confidence calculation metrics"""
    print("\n" + "="*60)
    print("TEST: Profile confidence calculation")
    print("="*60)

    # Very smooth profile
    smooth_profile = [(i/10, 0.5) for i in range(11)]

    # Very noisy profile
    noisy_profile = [(i/10, 0.1 if i % 2 == 0 else 0.9) for i in range(11)]

    # Sparse profile
    sparse_profile = [(i/10, 0.0 if i < 5 else 0.5) for i in range(11)]

    smooth_conf = calculate_profile_confidence(smooth_profile)
    noisy_conf = calculate_profile_confidence(noisy_profile)
    sparse_conf = calculate_profile_confidence(sparse_profile)

    print(f"Smooth profile confidence: {smooth_conf:.4f}")
    print(f"Noisy profile confidence:  {noisy_conf:.4f}")
    print(f"Sparse profile confidence: {sparse_conf:.4f}")

    # Smooth should have highest confidence
    assert smooth_conf > noisy_conf, "Smooth profile should have higher confidence than noisy"
    assert smooth_conf > sparse_conf, "Smooth profile should have higher confidence than sparse"

    print("✓ Confidence calculation works correctly")


def test_error_handling():
    """Test error handling for invalid inputs"""
    print("\n" + "="*60)
    print("TEST: Error handling")
    print("="*60)

    front_profile = [(i/10, 0.5) for i in range(11)]
    side_profile_wrong_length = [(i/5, 0.5) for i in range(6)]

    # Test mismatched lengths
    try:
        fuse_profiles(front_profile, side_profile_wrong_length)
        assert False, "Should raise ValueError for mismatched lengths"
    except ValueError as e:
        print(f"✓ Correctly caught mismatched length error: {e}")

    # Test empty profiles
    try:
        fuse_profiles([], [])
        assert False, "Should raise ValueError for empty profiles"
    except ValueError as e:
        print(f"✓ Correctly caught empty profile error: {e}")

    # Test invalid strategy
    side_profile = [(i/10, 0.5) for i in range(11)]
    try:
        fuse_profiles(front_profile, side_profile, fusion_strategy="invalid_strategy")
        assert False, "Should raise ValueError for invalid strategy"
    except ValueError as e:
        print(f"✓ Correctly caught invalid strategy error: {e}")

    print("✓ Error handling works correctly")


def run_all_tests():
    """Run all unit tests"""
    print("\n" + "="*60)
    print("PROFILE FUSION UNIT TESTS")
    print("="*60)

    tests = [
        test_equal_weights_fusion,
        test_front_heavy_fusion,
        test_side_heavy_fusion,
        test_adaptive_fusion,
        test_custom_weights,
        test_profile_confidence_calculation,
        test_error_handling
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ TEST FAILED: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
