"""
Test script to compare fixed vs adaptive thresholding for silhouette extraction.

This experiment tests different thresholding methods:
1. Fixed threshold (current implementation: threshold=127)
2. Adaptive threshold (Gaussian)
3. Adaptive threshold (Mean)
4. Otsu's method (automatic threshold selection)
"""

import numpy as np
import cv2
from pathlib import Path
import sys


def create_test_image_uniform_lighting(size=512):
    """Create test image with uniform lighting (ideal case)."""
    img = np.ones((size, size), dtype=np.uint8) * 200  # Light background

    # Draw a bottle-like shape in dark color
    center_x = size // 2

    # Neck (thin)
    cv2.rectangle(img, (center_x - 30, 50), (center_x + 30, 150), 40, -1)

    # Body (wider oval)
    cv2.ellipse(img, (center_x, 300), (80, 150), 0, 0, 360, 40, -1)

    return img


def create_test_image_gradient_lighting(size=512):
    """Create test image with gradient lighting (challenging case)."""
    # Create gradient background (darker on left, lighter on right)
    x = np.linspace(0, 1, size)
    y = np.ones(size)
    gradient = np.outer(y, x)
    img = (100 + gradient * 155).astype(np.uint8)  # Background: 100-255

    # Draw a bottle-like shape in variable darkness
    center_x = size // 2

    # Create mask for the bottle shape
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(mask, (center_x - 30, 50), (center_x + 30, 150), 255, -1)
    cv2.ellipse(mask, (center_x, 300), (80, 150), 0, 0, 360, 255, -1)

    # Apply darker shade to bottle area (but varying due to gradient)
    bottle_pixels = mask > 0
    img[bottle_pixels] = (img[bottle_pixels] * 0.3).astype(np.uint8)

    return img


def create_test_image_shadows(size=512):
    """Create test image with shadows (very challenging case)."""
    img = np.ones((size, size), dtype=np.uint8) * 200  # Light background

    # Add shadow (darker region on one side)
    shadow_region = img[:, :size//3]
    img[:, :size//3] = (shadow_region * 0.6).astype(np.uint8)

    # Draw bottle
    center_x = size // 2
    cv2.rectangle(img, (center_x - 30, 50), (center_x + 30, 150), 40, -1)
    cv2.ellipse(img, (center_x, 300), (80, 150), 0, 0, 360, 40, -1)

    return img


def extract_silhouette_fixed_threshold(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Extract silhouette using fixed threshold (current method)."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Fixed threshold
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Fill holes
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 255, -1)

    return filled


def extract_silhouette_adaptive_gaussian(image: np.ndarray, block_size: int = 35, C: int = 10) -> np.ndarray:
    """Extract silhouette using adaptive Gaussian thresholding."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Ensure block_size is odd
    if block_size % 2 == 0:
        block_size += 1

    # Adaptive Gaussian threshold
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, C
    )

    # Fill holes
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 255, -1)

    return filled


def extract_silhouette_adaptive_mean(image: np.ndarray, block_size: int = 35, C: int = 10) -> np.ndarray:
    """Extract silhouette using adaptive mean thresholding."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Ensure block_size is odd
    if block_size % 2 == 0:
        block_size += 1

    # Adaptive mean threshold
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, block_size, C
    )

    # Fill holes
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 255, -1)

    return filled


def extract_silhouette_otsu(image: np.ndarray) -> np.ndarray:
    """Extract silhouette using Otsu's method (automatic threshold)."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Otsu's automatic threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Fill holes
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 255, -1)

    return filled


def calculate_metrics(silhouette: np.ndarray, ground_truth: np.ndarray) -> dict:
    """Calculate comparison metrics."""
    # Resize if needed
    if silhouette.shape != ground_truth.shape:
        silhouette = cv2.resize(silhouette, (ground_truth.shape[1], ground_truth.shape[0]))

    # Convert to binary
    sil_binary = silhouette > 127
    gt_binary = ground_truth > 127

    # Calculate metrics
    intersection = np.logical_and(sil_binary, gt_binary).sum()
    union = np.logical_or(sil_binary, gt_binary).sum()

    iou = intersection / union if union > 0 else 0

    # Pixel accuracy
    correct = (sil_binary == gt_binary).sum()
    total = sil_binary.size
    accuracy = correct / total

    # Precision and Recall
    true_positive = intersection
    false_positive = np.logical_and(sil_binary, ~gt_binary).sum()
    false_negative = np.logical_and(~sil_binary, gt_binary).sum()

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'iou': iou,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def create_comparison_visualization(original, results_dict, output_path):
    """Create a visualization comparing all methods."""
    n_methods = len(results_dict)

    # Create grid: original + all methods
    rows = 1 + (n_methods + 2) // 3
    cols = 3

    fig_height = original.shape[0] * rows
    fig_width = original.shape[1] * cols

    canvas = np.ones((fig_height, fig_width), dtype=np.uint8) * 255

    # Add original at top-left
    canvas[0:original.shape[0], 0:original.shape[1]] = original

    # Add text label
    cv2.putText(canvas, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)

    # Add each method result
    for idx, (method_name, result) in enumerate(results_dict.items(), start=1):
        row = idx // cols
        col = idx % cols

        y_start = row * original.shape[0]
        y_end = y_start + original.shape[0]
        x_start = col * original.shape[1]
        x_end = x_start + original.shape[1]

        canvas[y_start:y_end, x_start:x_end] = result

        # Add text label
        cv2.putText(canvas, method_name, (x_start + 10, y_start + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)

    cv2.imwrite(str(output_path), canvas)


def run_comparison():
    """Run threshold comparison experiment."""
    print("=" * 70)
    print("THRESHOLD COMPARISON EXPERIMENT")
    print("=" * 70)
    print()

    # Create output directory
    output_dir = Path("threshold_comparison_results")
    output_dir.mkdir(exist_ok=True)

    # Test cases
    test_cases = {
        "uniform_lighting": create_test_image_uniform_lighting(),
        "gradient_lighting": create_test_image_gradient_lighting(),
        "shadows": create_test_image_shadows()
    }

    # Methods to test
    methods = {
        "Fixed (127)": lambda img: extract_silhouette_fixed_threshold(img, 127),
        "Adaptive Gaussian": extract_silhouette_adaptive_gaussian,
        "Adaptive Mean": extract_silhouette_adaptive_mean,
        "Otsu": extract_silhouette_otsu
    }

    # Run tests
    all_results = {}

    for test_name, test_image in test_cases.items():
        print(f"\nTest Case: {test_name}")
        print("-" * 70)

        # Save original
        cv2.imwrite(str(output_dir / f"{test_name}_original.png"), test_image)

        # Create ground truth (using Otsu as baseline)
        ground_truth = extract_silhouette_otsu(test_image)

        results = {}
        method_metrics = {}

        for method_name, method_func in methods.items():
            # Extract silhouette
            silhouette = method_func(test_image)
            results[method_name] = silhouette

            # Save individual result
            cv2.imwrite(
                str(output_dir / f"{test_name}_{method_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"),
                silhouette
            )

            # Calculate metrics (compare against Otsu as reference)
            if method_name != "Otsu":
                metrics = calculate_metrics(silhouette, ground_truth)
                method_metrics[method_name] = metrics

                print(f"  {method_name}:")
                print(f"    IoU:       {metrics['iou']:.4f}")
                print(f"    Accuracy:  {metrics['accuracy']:.4f}")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall:    {metrics['recall']:.4f}")
                print(f"    F1:        {metrics['f1']:.4f}")

        # Create comparison visualization
        create_comparison_visualization(
            test_image,
            results,
            output_dir / f"{test_name}_comparison.png"
        )

        all_results[test_name] = method_metrics

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for test_name, method_metrics in all_results.items():
        print(f"\n{test_name}:")

        # Find best method for each metric
        best_iou = max(method_metrics.items(), key=lambda x: x[1]['iou'])
        best_f1 = max(method_metrics.items(), key=lambda x: x[1]['f1'])

        print(f"  Best IoU:  {best_iou[0]} ({best_iou[1]['iou']:.4f})")
        print(f"  Best F1:   {best_f1[0]} ({best_f1[1]['f1']:.4f})")

    print(f"\nResults saved to: {output_dir}/")
    print("\nVisualization files:")
    for test_name in test_cases.keys():
        print(f"  - {test_name}_comparison.png")

    return all_results


def main():
    """Main entry point."""
    print("Testing fixed vs adaptive thresholding for silhouette extraction")
    print()

    results = run_comparison()

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
