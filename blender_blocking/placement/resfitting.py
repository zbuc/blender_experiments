"""
Residual Primitive Fitting (ResFit) algorithm.

Based on arXiv:2512.09201 Section 3.2 - Residual Primitive Fitting

Simplified implementation:
1. Initialize primitives from voxel/slice analysis
2. Optimize primitive parameters using gradient descent on SDF
3. Compute residual error (uncovered regions)
4. Add new primitives in high-error regions
5. Repeat until convergence

References:
- https://arxiv.org/abs/2512.09201
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from primitives.superfrustum import SuperFrustum


class ResiduaFitter:
    """
    Residual Primitive Fitting algorithm for optimal primitive placement.

    Iteratively fits primitives to 3D shapes by:
    1. Placing initial primitives
    2. Optimizing parameters via gradient descent
    3. Computing residual error
    4. Adding primitives in high-error regions
    """

    def __init__(
        self,
        max_primitives: int = 20,
        max_iterations: int = 10,
        error_threshold: float = 0.01,
        learning_rate: float = 0.01,
        optimization_steps: int = 50
    ):
        """
        Initialize ResFit algorithm.

        Args:
            max_primitives: Maximum number of primitives to place
            max_iterations: Maximum fitting iterations
            error_threshold: Target error threshold for convergence
            learning_rate: Learning rate for gradient descent
            optimization_steps: Number of optimization steps per iteration
        """
        self.max_primitives = max_primitives
        self.max_iterations = max_iterations
        self.error_threshold = error_threshold
        self.learning_rate = learning_rate
        self.optimization_steps = optimization_steps

        self.primitives: List[SuperFrustum] = []
        self.errors: List[float] = []

    def initialize_from_slices(
        self,
        slice_data: List[Dict],
        num_initial: int = 5
    ) -> List[SuperFrustum]:
        """
        Initialize primitives from slice analysis data.

        Args:
            slice_data: List of slice analysis results from SliceAnalyzer
            num_initial: Number of initial primitives to place

        Returns:
            List of initialized SuperFrustum primitives
        """
        primitives = []

        # Group slices into segments for primitive placement
        slices_per_primitive = max(1, len(slice_data) // num_initial)

        for i in range(num_initial):
            start_idx = i * slices_per_primitive
            end_idx = min((i + 1) * slices_per_primitive, len(slice_data))

            if start_idx >= len(slice_data):
                break

            # Extract slice segment
            segment = slice_data[start_idx:end_idx]

            # Compute average properties
            avg_center = np.mean([s['center'] for s in segment], axis=0)
            avg_radius = np.mean([s['radius'] for s in segment])

            # Estimate bottom and top radii
            radius_bottom = segment[0]['radius'] if segment else avg_radius
            radius_top = segment[-1]['radius'] if len(segment) > 1 else avg_radius

            # Estimate height
            if len(segment) > 1:
                z_min = segment[0]['center'][2]
                z_max = segment[-1]['center'][2]
                height = abs(z_max - z_min)
            else:
                height = 2.0  # Default

            # Create SuperFrustum
            sf = SuperFrustum(
                position=tuple(avg_center),
                orientation=(0.0, 0.0),  # Aligned with Z axis
                radius_bottom=max(radius_bottom, 0.1),
                radius_top=max(radius_top, 0.1),
                height=max(height, 0.5)
            )

            primitives.append(sf)

        return primitives

    def initialize_from_voxels(
        self,
        voxel_grid: np.ndarray,
        num_initial: int = 5
    ) -> List[SuperFrustum]:
        """
        Initialize primitives from voxel grid (simplified MSD approach).

        Args:
            voxel_grid: 3D occupancy grid (1 = filled, 0 = empty)
            num_initial: Number of initial primitives

        Returns:
            List of initialized SuperFrustum primitives
        """
        primitives = []

        # Find occupied voxels
        occupied = np.argwhere(voxel_grid > 0.5)

        if len(occupied) == 0:
            return primitives

        # Simple k-means-like clustering to find initial placements
        # For now, use vertical slicing as a proxy
        z_min, z_max = occupied[:, 2].min(), occupied[:, 2].max()
        z_step = (z_max - z_min) / num_initial

        for i in range(num_initial):
            z_start = z_min + i * z_step
            z_end = z_min + (i + 1) * z_step

            # Get voxels in this z-range
            mask = (occupied[:, 2] >= z_start) & (occupied[:, 2] < z_end)
            segment_voxels = occupied[mask]

            if len(segment_voxels) == 0:
                continue

            # Compute center
            center = segment_voxels.mean(axis=0)

            # Estimate radius from XY extent
            xy_extent = segment_voxels[:, :2] - center[:2]
            radius = np.percentile(np.linalg.norm(xy_extent, axis=1), 75)

            # Estimate height
            height = z_end - z_start

            # Create primitive
            sf = SuperFrustum(
                position=tuple(center),
                orientation=(0.0, 0.0),
                radius_bottom=max(radius, 0.1),
                radius_top=max(radius * 0.8, 0.1),  # Slight taper
                height=max(height, 0.5)
            )

            primitives.append(sf)

        return primitives

    def optimize_primitives(
        self,
        primitives: List[SuperFrustum],
        target_points: np.ndarray,
        steps: int = 50
    ) -> List[SuperFrustum]:
        """
        Optimize primitive parameters using gradient descent.

        Args:
            primitives: List of SuperFrustum primitives to optimize
            target_points: Nx3 array of target surface points
            steps: Number of optimization steps

        Returns:
            Optimized primitives
        """
        for step in range(steps):
            total_loss = 0.0

            for sf in primitives:
                # Compute SDF for all target points
                sdf_values = np.array([sf.sdf(p) for p in target_points])

                # Loss: sum of squared distances (points should be on surface, SDF=0)
                loss = np.mean(sdf_values ** 2)
                total_loss += loss

                # Compute gradients for this primitive
                # Average gradient across all target points
                grad_pos = np.zeros(3)
                grad_orient = np.zeros(2)
                grad_rb = 0.0
                grad_rt = 0.0
                grad_h = 0.0

                for p in target_points:
                    grads = sf.gradient(p)
                    # Gradient of loss w.r.t. parameters
                    # d(loss)/d(param) = d(sdf²)/d(param) = 2*sdf * d(sdf)/d(param)
                    sdf_val = sf.sdf(p)
                    grad_pos += 2 * sdf_val * grads['position']
                    grad_orient += 2 * sdf_val * grads['orientation']
                    grad_rb += 2 * sdf_val * grads['radius_bottom']
                    grad_rt += 2 * sdf_val * grads['radius_top']
                    grad_h += 2 * sdf_val * grads['height']

                # Average gradients
                n = len(target_points)
                grad_pos /= n
                grad_orient /= n
                grad_rb /= n
                grad_rt /= n
                grad_h /= n

                # Gradient descent update
                sf.position -= self.learning_rate * grad_pos
                sf.orientation -= self.learning_rate * grad_orient
                sf.radius_bottom -= self.learning_rate * grad_rb
                sf.radius_top -= self.learning_rate * grad_rt
                sf.height -= self.learning_rate * grad_h

                # Clamp parameters to valid ranges
                sf.radius_bottom = max(sf.radius_bottom, 0.1)
                sf.radius_top = max(sf.radius_top, 0.01)
                sf.height = max(sf.height, 0.1)

            # Print progress every 10 steps
            if step % 10 == 0:
                print(f"    Optimization step {step}/{steps}: loss = {total_loss:.6f}")

        return primitives

    def compute_residual_error(
        self,
        primitives: List[SuperFrustum],
        target_points: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Compute residual error (points not well-covered by primitives).

        Args:
            primitives: List of SuperFrustum primitives
            target_points: Nx3 array of target surface points

        Returns:
            Tuple of (total_error, per_point_errors)
        """
        per_point_errors = np.zeros(len(target_points))

        for i, p in enumerate(target_points):
            # Find minimum SDF across all primitives
            min_sdf = float('inf')
            for sf in primitives:
                sdf = abs(sf.sdf(p))
                min_sdf = min(min_sdf, sdf)

            per_point_errors[i] = min_sdf

        total_error = np.mean(per_point_errors)
        return total_error, per_point_errors

    def add_primitive_at_error_region(
        self,
        primitives: List[SuperFrustum],
        target_points: np.ndarray,
        per_point_errors: np.ndarray
    ) -> Optional[SuperFrustum]:
        """
        Add new primitive in highest-error region.

        Args:
            primitives: Existing primitives
            target_points: Nx3 array of target points
            per_point_errors: Error value for each point

        Returns:
            New SuperFrustum primitive, or None if can't add
        """
        if len(primitives) >= self.max_primitives:
            return None

        # Find points with high error (top 25%)
        error_threshold = np.percentile(per_point_errors, 75)
        high_error_mask = per_point_errors > error_threshold
        high_error_points = target_points[high_error_mask]

        if len(high_error_points) == 0:
            return None

        # Initialize new primitive at high-error region
        center = high_error_points.mean(axis=0)

        # Estimate radius from point spread
        distances = np.linalg.norm(high_error_points - center, axis=1)
        radius = np.percentile(distances, 50)

        # Create new primitive
        new_sf = SuperFrustum(
            position=tuple(center),
            orientation=(0.0, 0.0),
            radius_bottom=max(radius, 0.1),
            radius_top=max(radius * 0.7, 0.1),
            height=max(radius * 2.0, 0.5)
        )

        return new_sf

    def fit(
        self,
        target_points: np.ndarray,
        initial_primitives: Optional[List[SuperFrustum]] = None,
        num_initial: int = 5,
        verbose: bool = True
    ) -> List[SuperFrustum]:
        """
        Run ResFit algorithm to fit primitives to target points.

        Args:
            target_points: Nx3 array of target surface points
            initial_primitives: Optional initial primitives (if None, auto-initialize)
            num_initial: Number of initial primitives (if auto-initializing)
            verbose: Print progress

        Returns:
            List of fitted SuperFrustum primitives
        """
        if verbose:
            print("\n" + "="*70)
            print("RESIDUAL PRIMITIVE FITTING (ResFit)")
            print("="*70)

        # Initialize primitives
        if initial_primitives is None:
            if verbose:
                print(f"\nInitializing {num_initial} primitives from target points...")

            # Simple initialization: cluster target points
            # For now, use vertical slicing
            z_coords = target_points[:, 2]
            z_min, z_max = z_coords.min(), z_coords.max()

            self.primitives = []
            for i in range(num_initial):
                z_start = z_min + i * (z_max - z_min) / num_initial
                z_end = z_min + (i + 1) * (z_max - z_min) / num_initial

                # Get points in this z-range
                mask = (z_coords >= z_start) & (z_coords < z_end)
                segment_points = target_points[mask]

                if len(segment_points) == 0:
                    continue

                # Compute properties
                center = segment_points.mean(axis=0)
                xy_extent = segment_points[:, :2] - center[:2]
                radius = np.percentile(np.linalg.norm(xy_extent, axis=1), 75)

                sf = SuperFrustum(
                    position=tuple(center),
                    orientation=(0.0, 0.0),
                    radius_bottom=max(radius, 0.1),
                    radius_top=max(radius * 0.8, 0.1),
                    height=z_end - z_start
                )
                self.primitives.append(sf)
        else:
            self.primitives = initial_primitives

        if verbose:
            print(f"  ✓ Initialized {len(self.primitives)} primitives")

        # Iterative fitting
        self.errors = []

        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")

            # Optimize primitives
            if verbose:
                print(f"  Optimizing {len(self.primitives)} primitives...")
            self.primitives = self.optimize_primitives(
                self.primitives,
                target_points,
                steps=self.optimization_steps
            )

            # Compute residual error
            total_error, per_point_errors = self.compute_residual_error(
                self.primitives,
                target_points
            )
            self.errors.append(total_error)

            if verbose:
                print(f"  Residual error: {total_error:.6f}")

            # Check convergence
            if total_error < self.error_threshold:
                if verbose:
                    print(f"\n✓ Converged! Error {total_error:.6f} < {self.error_threshold}")
                break

            # Add new primitive in high-error region
            new_primitive = self.add_primitive_at_error_region(
                self.primitives,
                target_points,
                per_point_errors
            )

            if new_primitive is not None:
                self.primitives.append(new_primitive)
                if verbose:
                    print(f"  + Added primitive in high-error region (total: {len(self.primitives)})")
            else:
                if verbose:
                    print(f"  No new primitive added (max reached or no high-error regions)")

        if verbose:
            print("\n" + "="*70)
            print("RESFITTING COMPLETE")
            print("="*70)
            print(f"  Final primitives: {len(self.primitives)}")
            print(f"  Final error: {self.errors[-1]:.6f}")
            print(f"  Iterations: {len(self.errors)}")

        return self.primitives

    def get_history(self) -> Dict:
        """
        Get fitting history.

        Returns:
            Dictionary with fitting history
        """
        return {
            'num_primitives': len(self.primitives),
            'errors': self.errors,
            'iterations': len(self.errors),
            'final_error': self.errors[-1] if self.errors else None
        }
