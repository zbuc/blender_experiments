"""
Multi-View Visual Hull Reconstruction.

Extends current 3-view pipeline to support N views (typically 8-12) for higher accuracy.
Based on research document: MULTI_VIEW_RECONSTRUCTION_RESEARCH.md

Algorithm:
1. Initialize 3D voxel grid (128³ or 256³)
2. For each of N views:
   - Back-project silhouette into 3D space (silhouette cone)
   - Mark voxels inside this view's cone
3. Intersect all N cones:
   - Voxel is INSIDE if inside ALL N silhouette cones
   - Voxel is OUTSIDE if outside ANY silhouette cone
4. Extract mesh from final voxel grid (Marching Cubes)

Expected performance:
- 3 views: ~30s
- 12 views: ~60-80s
- Complexity: O(R³ × N) where R = voxel resolution, N = views

References:
- Research doc: MULTI_VIEW_RECONSTRUCTION_RESEARCH.md
- Visual Hull: https://en.wikipedia.org/wiki/Visual_hull
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import math


class CameraView:
    """
    Represents a single camera view for Visual Hull reconstruction.

    Attributes:
        silhouette: 2D binary silhouette (HxW, 1=object, 0=background)
        angle: Rotation angle in degrees (for turntable setup)
        view_type: 'lateral' or 'top'
        camera_distance: Distance from rotation center
        focal_length: Camera focal length (optional, for calibration)
    """

    def __init__(
        self,
        silhouette: np.ndarray,
        angle: float = 0.0,
        view_type: str = 'lateral',
        camera_distance: float = 1.0,
        focal_length: Optional[float] = None
    ):
        """
        Initialize camera view.

        Args:
            silhouette: Binary silhouette mask (HxW)
            angle: Turntable rotation angle in degrees (0-360)
            view_type: 'lateral' (horizontal) or 'top' (overhead)
            camera_distance: Distance from turntable center
            focal_length: Camera focal length (optional)
        """
        self.silhouette = silhouette.astype(bool)
        self.angle = angle
        self.view_type = view_type
        self.camera_distance = camera_distance
        self.focal_length = focal_length or camera_distance  # Default to orthographic

        self.height, self.width = silhouette.shape

    def is_point_in_silhouette_cone(
        self,
        point_3d: np.ndarray,
        voxel_bounds: Tuple[np.ndarray, np.ndarray]
    ) -> bool:
        """
        Check if a 3D point projects inside this view's silhouette.

        Args:
            point_3d: 3D point (x, y, z)
            voxel_bounds: (min_bounds, max_bounds) for voxel grid

        Returns:
            True if point is inside silhouette cone, False otherwise
        """
        # Project 3D point to 2D silhouette coordinates
        u, v = self.project_point(point_3d, voxel_bounds)

        # Check if projection is within image bounds
        if u < 0 or u >= self.width or v < 0 or v >= self.height:
            return False

        # Check if projection falls inside silhouette
        return self.silhouette[v, u]

    def project_point(
        self,
        point_3d: np.ndarray,
        voxel_bounds: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[int, int]:
        """
        Project 3D point to 2D silhouette coordinates.

        Uses simplified camera model for turntable setup:
        - Orthographic projection (parallel rays)
        - Camera rotates around object (turntable)
        - Object centered at origin

        Args:
            point_3d: 3D point (x, y, z) in voxel space
            voxel_bounds: (min_bounds, max_bounds) for normalization

        Returns:
            (u, v) pixel coordinates in silhouette
        """
        min_bounds, max_bounds = voxel_bounds

        if self.view_type == 'top':
            # Top view: camera above, looking down
            # Project (x, y, z) -> (x, y)
            world_x = point_3d[0]
            world_y = point_3d[1]
        else:
            # Lateral view: rotate point to camera's reference frame
            angle_rad = math.radians(self.angle)

            # Rotate point around Z axis by -angle (inverse of camera rotation)
            cos_a = math.cos(-angle_rad)
            sin_a = math.sin(-angle_rad)

            rotated_x = point_3d[0] * cos_a - point_3d[1] * sin_a
            rotated_y = point_3d[0] * sin_a + point_3d[1] * cos_a

            # After rotation, camera looks along +Y axis
            # Project (rotated_x, rotated_y, z) -> (rotated_x, z)
            world_x = rotated_x
            world_y = point_3d[2]  # Z becomes vertical in image

        # Normalize to [0, 1] based on voxel bounds
        x_range = max_bounds[0] - min_bounds[0]
        y_range_idx = 1 if self.view_type == 'top' else 2
        y_range = max_bounds[y_range_idx] - min_bounds[y_range_idx]

        x_norm = (world_x - min_bounds[0]) / x_range if x_range > 0 else 0.5
        y_norm = (world_y - min_bounds[y_range_idx]) / y_range if y_range > 0 else 0.5

        # Map to pixel coordinates
        u = int(x_norm * (self.width - 1))
        v = int((1.0 - y_norm) * (self.height - 1))  # Flip Y (image origin top-left)

        return (u, v)


class MultiViewVisualHull:
    """
    Multi-view Visual Hull reconstruction using voxel intersection.

    Supports N camera views (typically 8-12) to achieve higher accuracy
    than 3-view baseline.
    """

    def __init__(
        self,
        resolution: int = 128,
        bounds_min: Optional[np.ndarray] = None,
        bounds_max: Optional[np.ndarray] = None
    ):
        """
        Initialize Visual Hull reconstructor.

        Args:
            resolution: Voxel grid resolution (default: 128)
            bounds_min: Minimum bounds (x, y, z), default: (-1, -1, -1)
            bounds_max: Maximum bounds (x, y, z), default: (1, 1, 1)
        """
        self.resolution = resolution
        self.bounds_min = bounds_min if bounds_min is not None else np.array([-1.0, -1.0, -1.0])
        self.bounds_max = bounds_max if bounds_max is not None else np.array([1.0, 1.0, 1.0])

        # Voxel grid will be created during reconstruction
        self.voxel_grid = None
        self.views: List[CameraView] = []

    def add_view(self, view: CameraView):
        """Add a camera view to the reconstruction."""
        self.views.append(view)

    def add_view_from_silhouette(
        self,
        silhouette: np.ndarray,
        angle: float = 0.0,
        view_type: str = 'lateral',
        camera_distance: float = 1.0
    ):
        """
        Convenience method to add a view from silhouette image.

        Args:
            silhouette: Binary silhouette mask
            angle: Rotation angle in degrees
            view_type: 'lateral' or 'top'
            camera_distance: Distance from center
        """
        view = CameraView(silhouette, angle, view_type, camera_distance)
        self.add_view(view)

    def reconstruct(self, verbose: bool = True) -> np.ndarray:
        """
        Reconstruct 3D volume using multi-view Visual Hull algorithm.

        Returns:
            3D voxel grid (resolution³) with 1=inside, 0=outside
        """
        if len(self.views) == 0:
            raise ValueError("No views added. Call add_view() first.")

        if verbose:
            print(f"\n{'='*70}")
            print(f"MULTI-VIEW VISUAL HULL RECONSTRUCTION")
            print(f"{'='*70}")
            print(f"  Views: {len(self.views)}")
            print(f"  Resolution: {self.resolution}³")
            print(f"  Bounds: {self.bounds_min} to {self.bounds_max}")

        # Initialize voxel grid (all True initially, will be intersected)
        self.voxel_grid = np.ones((self.resolution, self.resolution, self.resolution), dtype=bool)

        # Compute voxel centers
        voxel_centers = self._compute_voxel_centers()

        if verbose:
            print(f"\n  Processing {len(self.views)} views...")

        # For each view, mark voxels that are inside the silhouette cone
        for i, view in enumerate(self.views):
            if verbose:
                view_desc = f"{view.view_type} @ {view.angle}°" if view.view_type == 'lateral' else "top view"
                print(f"    [{i+1}/{len(self.views)}] {view_desc}... ", end='', flush=True)

            # Create mask for this view
            view_mask = np.zeros_like(self.voxel_grid, dtype=bool)

            # Check each voxel
            total_voxels = self.resolution ** 3
            for ix in range(self.resolution):
                for iy in range(self.resolution):
                    for iz in range(self.resolution):
                        point = voxel_centers[ix, iy, iz]
                        if view.is_point_in_silhouette_cone(point, (self.bounds_min, self.bounds_max)):
                            view_mask[ix, iy, iz] = True

            # Intersect with existing grid (voxel must be in ALL views)
            self.voxel_grid &= view_mask

            occupied = view_mask.sum()
            if verbose:
                print(f"✓ ({occupied:,} voxels inside)")

        if verbose:
            final_occupied = self.voxel_grid.sum()
            print(f"\n  Final occupied voxels: {final_occupied:,} / {total_voxels:,}")
            print(f"  Occupancy: {final_occupied / total_voxels * 100:.2f}%")
            print(f"{'='*70}\n")

        return self.voxel_grid

    def _compute_voxel_centers(self) -> np.ndarray:
        """
        Compute 3D center coordinates for all voxels.

        Returns:
            Array of shape (resolution, resolution, resolution, 3) with voxel centers
        """
        # Create coordinate arrays
        x = np.linspace(self.bounds_min[0], self.bounds_max[0], self.resolution)
        y = np.linspace(self.bounds_min[1], self.bounds_max[1], self.resolution)
        z = np.linspace(self.bounds_min[2], self.bounds_max[2], self.resolution)

        # Create meshgrid
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

        # Stack into (resolution, resolution, resolution, 3) array
        voxel_centers = np.stack([xx, yy, zz], axis=-1)

        return voxel_centers

    def extract_mesh_points(self, surface_only: bool = True) -> np.ndarray:
        """
        Extract point cloud from voxel grid.

        Args:
            surface_only: If True, only extract surface voxels (faster)

        Returns:
            Nx3 array of 3D points
        """
        if self.voxel_grid is None:
            raise ValueError("No reconstruction yet. Call reconstruct() first.")

        if surface_only:
            # Extract surface voxels (adjacent to empty space)
            # Simple surface detection without scipy: check 6-connected neighbors
            mask = self._extract_surface_voxels(self.voxel_grid)
        else:
            mask = self.voxel_grid

        # Get voxel indices
        indices = np.argwhere(mask)

        # Convert to world coordinates
        voxel_centers = self._compute_voxel_centers()
        points = voxel_centers[indices[:, 0], indices[:, 1], indices[:, 2]]

        return points

    def _extract_surface_voxels(self, voxel_grid: np.ndarray) -> np.ndarray:
        """
        Extract surface voxels (those adjacent to empty space).

        Uses simple 6-connectivity check (no scipy required).

        Args:
            voxel_grid: 3D boolean array

        Returns:
            3D boolean array with surface voxels marked
        """
        surface = np.zeros_like(voxel_grid, dtype=bool)
        r, c, d = voxel_grid.shape

        # Check each occupied voxel
        for i in range(r):
            for j in range(c):
                for k in range(d):
                    if not voxel_grid[i, j, k]:
                        continue

                    # Check if any 6-connected neighbor is empty (then it's surface)
                    is_surface = False

                    # Check -X neighbor
                    if i == 0 or not voxel_grid[i-1, j, k]:
                        is_surface = True
                    # Check +X neighbor
                    elif i == r-1 or not voxel_grid[i+1, j, k]:
                        is_surface = True
                    # Check -Y neighbor
                    elif j == 0 or not voxel_grid[i, j-1, k]:
                        is_surface = True
                    # Check +Y neighbor
                    elif j == c-1 or not voxel_grid[i, j+1, k]:
                        is_surface = True
                    # Check -Z neighbor
                    elif k == 0 or not voxel_grid[i, j, k-1]:
                        is_surface = True
                    # Check +Z neighbor
                    elif k == d-1 or not voxel_grid[i, j, k+1]:
                        is_surface = True

                    if is_surface:
                        surface[i, j, k] = True

        return surface

    def get_stats(self) -> Dict:
        """Get reconstruction statistics."""
        if self.voxel_grid is None:
            return {'reconstructed': False}

        total = self.voxel_grid.size
        occupied = self.voxel_grid.sum()

        return {
            'reconstructed': True,
            'num_views': len(self.views),
            'resolution': self.resolution,
            'total_voxels': total,
            'occupied_voxels': int(occupied),
            'occupancy': float(occupied) / total,
            'bounds_min': self.bounds_min.tolist(),
            'bounds_max': self.bounds_max.tolist()
        }


def load_multi_view_turntable(
    silhouette_paths: List[str],
    angles: List[float],
    include_top: bool = True,
    top_path: Optional[str] = None
) -> MultiViewVisualHull:
    """
    Convenience function to load turntable sequence.

    Args:
        silhouette_paths: List of paths to lateral view silhouettes
        angles: List of rotation angles (degrees) for each view
        include_top: Whether to include top view
        top_path: Path to top view silhouette (if include_top=True)

    Returns:
        MultiViewVisualHull instance with views loaded
    """
    from PIL import Image

    if len(silhouette_paths) != len(angles):
        raise ValueError(f"Number of paths ({len(silhouette_paths)}) must match angles ({len(angles)})")

    hull = MultiViewVisualHull()

    # Load lateral views
    for path, angle in zip(silhouette_paths, angles):
        img = Image.open(path).convert('L')  # Grayscale
        silhouette = np.array(img) > 127  # Threshold to binary
        hull.add_view_from_silhouette(silhouette, angle=angle, view_type='lateral')

    # Load top view
    if include_top and top_path:
        img = Image.open(top_path).convert('L')
        silhouette = np.array(img) > 127
        hull.add_view_from_silhouette(silhouette, angle=0.0, view_type='top')

    return hull
