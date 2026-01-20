"""
Experiment D: Adaptive vs Fixed Thresholding for Silhouette Extraction

This experiment compares two approaches to silhouette extraction:
1. Fixed thresholding (current implementation, threshold=127)
2. Adaptive thresholding (calculates local thresholds for each region)

The goal is to determine which approach produces more accurate silhouettes
across various lighting conditions and image types.
"""

import sys
from pathlib import Path

# Add blender_python_packages to path for dependencies
sys.path.insert(0, str(Path.home() / 'blender_python_packages'))

import numpy as np
import cv2
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class ThresholdResult:
    """Results from a thresholding approach."""
    silhouette: np.ndarray
    method_name: str
    contour_count: int
    filled_pixel_count: int
    processing_time_ms: float


def extract_silhouette_fixed(
    image: np.ndarray,
    threshold_value: int = 127
) -> np.ndarray:
    """
    Extract silhouette using fixed thresholding (current implementation).

    Args:
        image: Input image (grayscale or color)
        threshold_value: Fixed threshold value (default 127)

    Returns:
        Binary mask where object pixels are 255
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            gray = image[:, :, 3]
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Fixed threshold
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Fill holes
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 255, -1)

    return filled


def extract_silhouette_adaptive_mean(
    image: np.ndarray,
    block_size: int = 11,
    c_constant: int = 2
) -> np.ndarray:
    """
    Extract silhouette using adaptive mean thresholding.

    This method calculates the threshold for each pixel based on the mean
    intensity of the neighboring pixels in a block_size x block_size area.

    Args:
        image: Input image (grayscale or color)
        block_size: Size of pixel neighborhood (must be odd, >= 3)
        c_constant: Constant subtracted from weighted mean

    Returns:
        Binary mask where object pixels are 255
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            gray = image[:, :, 3]
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Adaptive threshold - MEAN method
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c_constant
    )

    # Fill holes
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 255, -1)

    return filled


def extract_silhouette_adaptive_gaussian(
    image: np.ndarray,
    block_size: int = 11,
    c_constant: int = 2
) -> np.ndarray:
    """
    Extract silhouette using adaptive Gaussian thresholding.

    This method calculates the threshold for each pixel based on a Gaussian-
    weighted sum of the neighboring pixels.

    Args:
        image: Input image (grayscale or color)
        block_size: Size of pixel neighborhood (must be odd, >= 3)
        c_constant: Constant subtracted from weighted mean

    Returns:
        Binary mask where object pixels are 255
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            gray = image[:, :, 3]
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Adaptive threshold - GAUSSIAN method
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c_constant
    )

    # Fill holes
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 255, -1)

    return filled


def extract_silhouette_otsu(image: np.ndarray) -> np.ndarray:
    """
    Extract silhouette using Otsu's automatic thresholding.

    Otsu's method automatically determines the optimal threshold value
    by maximizing the between-class variance.

    Args:
        image: Input image (grayscale or color)

    Returns:
        Binary mask where object pixels are 255
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            gray = image[:, :, 3]
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Fill holes
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 255, -1)

    return filled


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two binary masks.

    Args:
        mask1: First binary mask
        mask2: Second binary mask

    Returns:
        IoU score (0.0 to 1.0)
    """
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()

    if union == 0:
        return 0.0

    return intersection / union


def analyze_silhouette_quality(silhouette: np.ndarray) -> Dict[str, float]:
    """
    Analyze quality metrics for a silhouette.

    Args:
        silhouette: Binary silhouette mask

    Returns:
        Dictionary of quality metrics
    """
    contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    metrics = {
        'contour_count': len(contours),
        'filled_pixel_count': np.sum(silhouette > 0),
        'fill_ratio': np.sum(silhouette > 0) / silhouette.size,
    }

    if contours:
        # Largest contour metrics
        largest_contour = max(contours, key=cv2.contourArea)
        metrics['largest_contour_area'] = cv2.contourArea(largest_contour)
        metrics['largest_contour_perimeter'] = cv2.arcLength(largest_contour, True)

        # Compactness (circularity)
        if metrics['largest_contour_perimeter'] > 0:
            metrics['compactness'] = (4 * np.pi * metrics['largest_contour_area']) / \
                                    (metrics['largest_contour_perimeter'] ** 2)
        else:
            metrics['compactness'] = 0.0
    else:
        metrics['largest_contour_area'] = 0
        metrics['largest_contour_perimeter'] = 0
        metrics['compactness'] = 0.0

    return metrics


def compare_thresholding_methods(
    image: np.ndarray,
    reference_silhouette: np.ndarray = None
) -> Dict[str, Dict]:
    """
    Compare all thresholding methods on a single image.

    Args:
        image: Input image
        reference_silhouette: Optional ground truth silhouette for IoU comparison

    Returns:
        Dictionary mapping method name to results and metrics
    """
    results = {}

    # Test each method
    methods = [
        ('Fixed (127)', lambda img: extract_silhouette_fixed(img, 127)),
        ('Adaptive Mean', lambda img: extract_silhouette_adaptive_mean(img, block_size=11, c_constant=2)),
        ('Adaptive Gaussian', lambda img: extract_silhouette_adaptive_gaussian(img, block_size=11, c_constant=2)),
        ('Otsu Auto', extract_silhouette_otsu),
    ]

    for method_name, method_func in methods:
        import time
        start = time.perf_counter()
        silhouette = method_func(image)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Analyze quality
        metrics = analyze_silhouette_quality(silhouette)
        metrics['processing_time_ms'] = elapsed_ms

        # Compare to reference if provided
        if reference_silhouette is not None:
            metrics['iou_vs_reference'] = calculate_iou(silhouette, reference_silhouette)

        results[method_name] = {
            'silhouette': silhouette,
            'metrics': metrics
        }

    return results


def visualize_comparison(
    image: np.ndarray,
    results: Dict[str, Dict],
    save_path: Path = None
):
    """
    Create visualization comparing all thresholding methods.

    Args:
        image: Original input image
        results: Results from compare_thresholding_methods()
        save_path: Optional path to save visualization
    """
    num_methods = len(results)

    fig, axes = plt.subplots(2, (num_methods + 1) // 2 + 1, figsize=(16, 8))
    axes = axes.flatten()

    # Show original image
    if len(image.shape) == 3:
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image', fontsize=10, fontweight='bold')
    axes[0].axis('off')

    # Show each method's result
    for idx, (method_name, result_data) in enumerate(results.items(), start=1):
        silhouette = result_data['silhouette']
        metrics = result_data['metrics']

        axes[idx].imshow(silhouette, cmap='gray')

        # Create title with key metrics
        title = f"{method_name}\n"
        title += f"Contours: {metrics['contour_count']}, "
        title += f"Fill: {metrics['fill_ratio']:.2%}\n"
        title += f"Time: {metrics['processing_time_ms']:.2f}ms"

        if 'iou_vs_reference' in metrics:
            title += f"\nIoU: {metrics['iou_vs_reference']:.3f}"

        axes[idx].set_title(title, fontsize=9)
        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(num_methods + 1, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()


def run_experiment(image_paths: List[Path], output_dir: Path):
    """
    Run threshold comparison experiment on multiple images.

    Args:
        image_paths: List of image file paths to test
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT D: Adaptive vs Fixed Thresholding")
    print("=" * 70)
    print()

    all_results = []

    for img_path in image_paths:
        print(f"\nProcessing: {img_path.name}")
        print("-" * 70)

        # Load image
        image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"  ERROR: Could not load image {img_path}")
            continue

        # Run comparison
        results = compare_thresholding_methods(image)

        # Print metrics
        for method_name, result_data in results.items():
            metrics = result_data['metrics']
            print(f"\n  {method_name}:")
            print(f"    Contours: {metrics['contour_count']}")
            print(f"    Filled pixels: {metrics['filled_pixel_count']}")
            print(f"    Fill ratio: {metrics['fill_ratio']:.2%}")
            print(f"    Compactness: {metrics['compactness']:.3f}")
            print(f"    Processing time: {metrics['processing_time_ms']:.2f} ms")

        # Visualize
        vis_path = output_dir / f"{img_path.stem}_comparison.png"
        visualize_comparison(image, results, save_path=vis_path)

        all_results.append({
            'image_name': img_path.name,
            'results': results
        })

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    return all_results


def main():
    """Main experiment entry point."""
    # Look for test images
    project_root = Path(__file__).parent.parent

    # Try to find reference images or test images
    possible_image_dirs = [
        project_root / "reference_images",
        project_root / "test_data",
        project_root / "blender_blocking" / "test_data",
    ]

    image_paths = []
    for img_dir in possible_image_dirs:
        if img_dir.exists():
            image_paths.extend(img_dir.glob("*.png"))
            image_paths.extend(img_dir.glob("*.jpg"))
            image_paths.extend(img_dir.glob("*.jpeg"))

    if not image_paths:
        print("No test images found. Please provide images to test.")
        print("Searched in:")
        for img_dir in possible_image_dirs:
            print(f"  - {img_dir}")
        return

    # Limit to first 5 images for initial testing
    image_paths = sorted(image_paths)[:5]

    print(f"Found {len(image_paths)} test images")

    # Output directory
    output_dir = project_root / "experiments" / "results" / "exp_d_thresholding"

    # Run experiment
    run_experiment(image_paths, output_dir)


if __name__ == "__main__":
    main()
