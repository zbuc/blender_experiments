"""
Adaptive thresholding variants for silhouette extraction.

This module provides alternative thresholding methods to compare against
the fixed threshold approach in profile_extractor.py.
"""

import numpy as np
import cv2
from typing import List, Tuple


def extract_silhouette_adaptive_mean(
    image: np.ndarray,
    block_size: int = 11,
    c_constant: int = 2
) -> np.ndarray:
    """
    Extract filled silhouette mask using adaptive mean thresholding.

    Adaptive thresholding calculates different threshold values for different
    regions of the image based on local pixel neighborhoods. This can be more
    robust than fixed thresholding for images with:
    - Uneven lighting
    - Varying contrast
    - Shadows or gradients

    Args:
        image: Input image (grayscale or color)
        block_size: Size of pixel neighborhood (must be odd, >= 3)
        c_constant: Constant subtracted from weighted mean

    Returns:
        Binary mask (0 or 255) where object pixels are 255
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
    # For each pixel, threshold = mean of block_size x block_size neighborhood - c_constant
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c_constant
    )

    # Fill any holes in the silhouette
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create filled silhouette
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 255, -1)

    return filled


def extract_silhouette_adaptive_gaussian(
    image: np.ndarray,
    block_size: int = 11,
    c_constant: int = 2
) -> np.ndarray:
    """
    Extract filled silhouette mask using adaptive Gaussian thresholding.

    Similar to adaptive mean, but uses a Gaussian-weighted sum of the
    neighborhood pixels. This gives more weight to pixels closer to the
    center of the neighborhood window.

    Args:
        image: Input image (grayscale or color)
        block_size: Size of pixel neighborhood (must be odd, >= 3)
        c_constant: Constant subtracted from weighted mean

    Returns:
        Binary mask (0 or 255) where object pixels are 255
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

    # Fill any holes in the silhouette
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create filled silhouette
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 255, -1)

    return filled


def extract_silhouette_otsu(image: np.ndarray) -> np.ndarray:
    """
    Extract filled silhouette mask using Otsu's automatic thresholding.

    Otsu's method automatically determines the optimal threshold value
    by maximizing the between-class variance. This is useful when you
    don't know the appropriate threshold in advance.

    Args:
        image: Input image (grayscale or color)

    Returns:
        Binary mask (0 or 255) where object pixels are 255
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            gray = image[:, :, 3]
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Otsu's thresholding - automatically determines threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Fill any holes in the silhouette
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create filled silhouette
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 255, -1)

    return filled


def compare_thresholding_methods(image: np.ndarray) -> dict:
    """
    Compare different thresholding methods and return metrics.

    Args:
        image: Input image

    Returns:
        Dictionary with results for each method
    """
    from profile_extractor import extract_silhouette_from_image
    import time

    results = {}

    # Fixed threshold (current implementation)
    start = time.perf_counter()
    fixed_sil = extract_silhouette_from_image(image)
    results['fixed_127'] = {
        'silhouette': fixed_sil,
        'time_ms': (time.perf_counter() - start) * 1000,
        'pixel_count': np.sum(fixed_sil > 0),
        'fill_ratio': np.sum(fixed_sil > 0) / fixed_sil.size
    }

    # Adaptive mean
    start = time.perf_counter()
    adaptive_mean_sil = extract_silhouette_adaptive_mean(image)
    results['adaptive_mean'] = {
        'silhouette': adaptive_mean_sil,
        'time_ms': (time.perf_counter() - start) * 1000,
        'pixel_count': np.sum(adaptive_mean_sil > 0),
        'fill_ratio': np.sum(adaptive_mean_sil > 0) / adaptive_mean_sil.size
    }

    # Adaptive Gaussian
    start = time.perf_counter()
    adaptive_gauss_sil = extract_silhouette_adaptive_gaussian(image)
    results['adaptive_gaussian'] = {
        'silhouette': adaptive_gauss_sil,
        'time_ms': (time.perf_counter() - start) * 1000,
        'pixel_count': np.sum(adaptive_gauss_sil > 0),
        'fill_ratio': np.sum(adaptive_gauss_sil > 0) / adaptive_gauss_sil.size
    }

    # Otsu's method
    start = time.perf_counter()
    otsu_sil = extract_silhouette_otsu(image)
    results['otsu'] = {
        'silhouette': otsu_sil,
        'time_ms': (time.perf_counter() - start) * 1000,
        'pixel_count': np.sum(otsu_sil > 0),
        'fill_ratio': np.sum(otsu_sil > 0) / otsu_sil.size
    }

    return results


def calculate_silhouette_iou(sil1: np.ndarray, sil2: np.ndarray) -> float:
    """
    Calculate IoU between two silhouettes.

    Args:
        sil1: First silhouette
        sil2: Second silhouette

    Returns:
        IoU score (0.0 to 1.0)
    """
    intersection = np.logical_and(sil1 > 0, sil2 > 0).sum()
    union = np.logical_or(sil1 > 0, sil2 > 0).sum()

    if union == 0:
        return 0.0

    return intersection / union
