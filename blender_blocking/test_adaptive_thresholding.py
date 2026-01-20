#!/usr/bin/env python3
"""
Test adaptive vs fixed thresholding for silhouette extraction.

This test compares different thresholding approaches to determine which
produces the most accurate silhouettes for various image conditions.

Run with: blender --background --python test_adaptive_thresholding.py
"""

import sys
from pathlib import Path

# Add blender_blocking directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Add ~/blender_python_packages for dependencies
sys.path.insert(0, str(Path.home() / 'blender_python_packages'))

import numpy as np
import cv2


def create_test_images() -> dict:
    """
    Create synthetic test images with different characteristics.

    Returns:
        Dictionary of test image name -> image array
    """
    test_images = {}

    # Test 1: Uniform background (ideal case for fixed threshold)
    uniform = np.ones((400, 400), dtype=np.uint8) * 255
    cv2.circle(uniform, (200, 200), 100, 0, -1)  # Black circle
    test_images['uniform_background'] = uniform

    # Test 2: Gradient background (challenging for fixed threshold)
    gradient = np.zeros((400, 400), dtype=np.uint8)
    for i in range(400):
        gradient[i, :] = int((i / 400) * 255)  # Vertical gradient 0->255
    cv2.circle(gradient, (200, 200), 100, 50, -1)  # Dark gray circle
    test_images['gradient_background'] = gradient

    # Test 3: Uneven lighting (challenging for fixed threshold)
    uneven = np.ones((400, 400), dtype=np.uint8) * 200
    # Add bright spot
    cv2.circle(uneven, (300, 100), 80, 255, -1)
    # Add object
    cv2.circle(uneven, (200, 200), 100, 80, -1)  # Darker object
    test_images['uneven_lighting'] = uneven

    # Test 4: High contrast (good for both methods)
    high_contrast = np.ones((400, 400), dtype=np.uint8) * 255
    cv2.rectangle(high_contrast, (100, 100), (300, 300), 0, -1)
    test_images['high_contrast'] = high_contrast

    # Test 5: Low contrast (challenging for both methods)
    low_contrast = np.ones((400, 400), dtype=np.uint8) * 128
    cv2.circle(low_contrast, (200, 200), 100, 100, -1)  # Slight difference
    test_images['low_contrast'] = low_contrast

    return test_images


def test_thresholding_comparison():
    """
    Test and compare different thresholding methods.

    Returns:
        bool: True if test passed
    """
    from integration.shape_matching.profile_extractor import extract_silhouette_from_image
    from integration.shape_matching.profile_extractor_adaptive import (
        extract_silhouette_adaptive_mean,
        extract_silhouette_adaptive_gaussian,
        extract_silhouette_otsu,
        calculate_silhouette_iou
    )

    print("\n" + "="*70)
    print("THRESHOLD COMPARISON TEST")
    print("="*70)

    test_images = create_test_images()

    all_passed = True

    for test_name, test_image in test_images.items():
        print(f"\n{test_name}:")
        print("-" * 70)

        # Apply all methods
        fixed = extract_silhouette_from_image(test_image)
        adaptive_mean = extract_silhouette_adaptive_mean(test_image)
        adaptive_gaussian = extract_silhouette_adaptive_gaussian(test_image)
        otsu = extract_silhouette_otsu(test_image)

        # Calculate metrics
        print(f"  Fixed (127):")
        print(f"    Filled pixels: {np.sum(fixed > 0)}")
        print(f"    Fill ratio: {np.sum(fixed > 0) / fixed.size:.2%}")

        print(f"  Adaptive Mean:")
        print(f"    Filled pixels: {np.sum(adaptive_mean > 0)}")
        print(f"    Fill ratio: {np.sum(adaptive_mean > 0) / adaptive_mean.size:.2%}")
        print(f"    IoU vs Fixed: {calculate_silhouette_iou(adaptive_mean, fixed):.3f}")

        print(f"  Adaptive Gaussian:")
        print(f"    Filled pixels: {np.sum(adaptive_gaussian > 0)}")
        print(f"    Fill ratio: {np.sum(adaptive_gaussian > 0) / adaptive_gaussian.size:.2%}")
        print(f"    IoU vs Fixed: {calculate_silhouette_iou(adaptive_gaussian, fixed):.3f}")

        print(f"  Otsu Auto:")
        print(f"    Filled pixels: {np.sum(otsu > 0)}")
        print(f"    Fill ratio: {np.sum(otsu > 0) / otsu.size:.2%}")
        print(f"    IoU vs Fixed: {calculate_silhouette_iou(otsu, fixed):.3f}")

        # Basic sanity check: all methods should produce non-empty results
        if np.sum(fixed > 0) == 0:
            print(f"  ❌ FAIL: Fixed threshold produced empty silhouette")
            all_passed = False
        elif np.sum(adaptive_mean > 0) == 0:
            print(f"  ❌ FAIL: Adaptive mean produced empty silhouette")
            all_passed = False
        elif np.sum(adaptive_gaussian > 0) == 0:
            print(f"  ❌ FAIL: Adaptive Gaussian produced empty silhouette")
            all_passed = False
        elif np.sum(otsu > 0) == 0:
            print(f"  ❌ FAIL: Otsu produced empty silhouette")
            all_passed = False
        else:
            print(f"  ✓ All methods produced valid silhouettes")

    print("\n" + "="*70)
    if all_passed:
        print("✓ THRESHOLD COMPARISON TEST PASSED")
    else:
        print("❌ THRESHOLD COMPARISON TEST FAILED")
    print("="*70)

    return all_passed


def analyze_method_characteristics():
    """
    Analyze and document characteristics of each method.
    """
    print("\n" + "="*70)
    print("METHOD ANALYSIS")
    print("="*70)

    analysis = {
        'Fixed Threshold (127)': {
            'pros': [
                'Fast and simple',
                'Consistent results',
                'Works well with uniform backgrounds',
                'Current implementation - proven stable'
            ],
            'cons': [
                'Fails with varying lighting',
                'Not robust to shadows or gradients',
                'Requires manual threshold tuning',
                'May miss details in low-contrast areas'
            ],
            'best_for': 'Clean images with uniform backgrounds and good contrast'
        },
        'Adaptive Mean': {
            'pros': [
                'Handles varying lighting well',
                'More robust to gradients',
                'Automatically adjusts to local conditions',
                'Good for images with shadows'
            ],
            'cons': [
                'Slightly slower than fixed',
                'May over-segment in noisy regions',
                'Requires tuning block_size and c_constant',
                'Can produce artifacts at edges'
            ],
            'best_for': 'Images with uneven lighting or varying backgrounds'
        },
        'Adaptive Gaussian': {
            'pros': [
                'Similar to adaptive mean but smoother',
                'Better edge handling than adaptive mean',
                'Good for images with gradual lighting changes',
                'Reduces noise sensitivity'
            ],
            'cons': [
                'Slower than fixed and adaptive mean',
                'Still requires parameter tuning',
                'May over-smooth fine details',
                'More computational overhead'
            ],
            'best_for': 'Images with smooth lighting variations and noise'
        },
        'Otsu Auto': {
            'pros': [
                'Fully automatic - no parameters to tune',
                'Optimal for bimodal histograms',
                'Fast computation',
                'Works well when object/background are distinct'
            ],
            'cons': [
                'Assumes bimodal distribution',
                'Global threshold - not adaptive to local regions',
                'Fails when histogram is not bimodal',
                'May not work with complex backgrounds'
            ],
            'best_for': 'Images with clear object/background separation'
        }
    }

    for method_name, data in analysis.items():
        print(f"\n{method_name}:")
        print(f"  Pros:")
        for pro in data['pros']:
            print(f"    + {pro}")
        print(f"  Cons:")
        for con in data['cons']:
            print(f"    - {con}")
        print(f"  Best for: {data['best_for']}")

    print("\n" + "="*70)


def main():
    """Main test entry point."""
    print("\n" + "="*70)
    print("EXP-D: ADAPTIVE vs FIXED THRESHOLDING TEST")
    print("="*70)

    # Run comparison test
    test_passed = test_thresholding_comparison()

    # Analyze characteristics
    analyze_method_characteristics()

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print()
    print("Based on the analysis:")
    print()
    print("1. KEEP FIXED (127) for now if:")
    print("   - Reference images have uniform backgrounds")
    print("   - Current results are satisfactory")
    print("   - Performance is critical")
    print()
    print("2. SWITCH TO ADAPTIVE GAUSSIAN if:")
    print("   - Reference images have varying lighting")
    print("   - Seeing failures with shadows or gradients")
    print("   - Quality is more important than speed")
    print()
    print("3. USE OTSU AUTO if:")
    print("   - Want fully automatic threshold selection")
    print("   - Reference images have clear object/background")
    print("   - Don't want to tune parameters")
    print()
    print("4. HYBRID APPROACH:")
    print("   - Detect image characteristics automatically")
    print("   - Choose method based on histogram analysis")
    print("   - Fallback to fixed if adaptive fails")
    print()
    print("="*70)

    return test_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
