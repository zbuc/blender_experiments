"""
Slice-based Shape Matching Algorithm for Blender

This module implements a shape matching algorithm that compares 3D meshes
by analyzing their cross-sectional slices at regular intervals.
"""

import bpy
import bmesh
import mathutils
from mathutils import Vector
from typing import List, Tuple, Dict
import numpy as np


class SliceProfile:
    """Represents a 2D cross-sectional profile from a 3D mesh slice."""

    def __init__(self, points: List[Vector], plane_height: float):
        self.points = points
        self.plane_height = plane_height
        self.centroid = self._calculate_centroid()
        self.area = self._calculate_area()
        self.perimeter = self._calculate_perimeter()

    def _calculate_centroid(self) -> Vector:
        """Calculate the centroid of the slice profile."""
        if not self.points:
            return Vector((0, 0, 0))

        x = sum(p.x for p in self.points) / len(self.points)
        y = sum(p.y for p in self.points) / len(self.points)
        return Vector((x, y, self.plane_height))

    def _calculate_area(self) -> float:
        """Calculate the area of the slice profile using the shoelace formula."""
        if len(self.points) < 3:
            return 0.0

        area = 0.0
        for i in range(len(self.points)):
            j = (i + 1) % len(self.points)
            area += self.points[i].x * self.points[j].y
            area -= self.points[j].x * self.points[i].y

        return abs(area) / 2.0

    def _calculate_perimeter(self) -> float:
        """Calculate the perimeter of the slice profile."""
        if len(self.points) < 2:
            return 0.0

        perimeter = 0.0
        for i in range(len(self.points)):
            j = (i + 1) % len(self.points)
            perimeter += (self.points[i] - self.points[j]).length

        return perimeter

    def to_feature_vector(self) -> np.ndarray:
        """Convert slice profile to a feature vector for comparison."""
        return np.array([
            self.area,
            self.perimeter,
            self.area / (self.perimeter ** 2) if self.perimeter > 0 else 0,  # Compactness
            len(self.points)  # Complexity
        ])


class SliceBasedShapeMatcher:
    """
    Implements slice-based shape matching for 3D meshes.

    The algorithm:
    1. Slices both meshes at regular intervals along the Z-axis
    2. Extracts cross-sectional profiles from each slice
    3. Compares profiles using multiple metrics
    4. Computes overall similarity score
    """

    def __init__(self, num_slices: int = 20):
        self.num_slices = num_slices

    def slice_mesh(self, obj: bpy.types.Object, axis: str = 'Z') -> List[SliceProfile]:
        """
        Slice a mesh object along the specified axis.

        Args:
            obj: Blender mesh object
            axis: Axis to slice along ('X', 'Y', or 'Z')

        Returns:
            List of SliceProfile objects
        """
        # Get mesh bounds
        bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

        if axis == 'Z':
            min_val = min(v.z for v in bbox)
            max_val = max(v.z for v in bbox)
            axis_idx = 2
        elif axis == 'Y':
            min_val = min(v.y for v in bbox)
            max_val = max(v.y for v in bbox)
            axis_idx = 1
        else:  # X
            min_val = min(v.x for v in bbox)
            max_val = max(v.x for v in bbox)
            axis_idx = 0

        # Create BMesh
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.transform(obj.matrix_world)

        profiles = []
        step = (max_val - min_val) / (self.num_slices - 1) if self.num_slices > 1 else 0

        for i in range(self.num_slices):
            plane_pos = min_val + i * step
            profile_points = self._slice_at_plane(bm, axis_idx, plane_pos)

            if profile_points:
                profiles.append(SliceProfile(profile_points, plane_pos))

        bm.free()
        return profiles

    def _slice_at_plane(self, bm: bmesh.types.BMesh, axis_idx: int,
                       plane_pos: float, epsilon: float = 0.001) -> List[Vector]:
        """
        Extract intersection points where mesh edges cross a plane.

        Args:
            bm: BMesh object
            axis_idx: Index of axis (0=X, 1=Y, 2=Z)
            plane_pos: Position of the cutting plane
            epsilon: Tolerance for plane intersection

        Returns:
            List of intersection points
        """
        intersection_points = []

        for edge in bm.edges:
            v1, v2 = edge.verts
            p1 = v1.co[axis_idx]
            p2 = v2.co[axis_idx]

            # Check if edge crosses the plane
            if (p1 - plane_pos) * (p2 - plane_pos) <= 0:
                # Linear interpolation to find intersection point
                if abs(p2 - p1) > epsilon:
                    t = (plane_pos - p1) / (p2 - p1)
                    t = max(0.0, min(1.0, t))  # Clamp to [0, 1]

                    intersection = v1.co.lerp(v2.co, t)
                    intersection_points.append(intersection)

        return intersection_points

    def compare_profiles(self, profiles1: List[SliceProfile],
                        profiles2: List[SliceProfile]) -> float:
        """
        Compare two sets of slice profiles.

        Args:
            profiles1: First set of profiles
            profiles2: Second set of profiles

        Returns:
            Similarity score between 0 (completely different) and 1 (identical)
        """
        if not profiles1 or not profiles2:
            return 0.0

        # Ensure same number of profiles for comparison
        min_len = min(len(profiles1), len(profiles2))
        profiles1 = profiles1[:min_len]
        profiles2 = profiles2[:min_len]

        # Extract feature vectors
        features1 = np.array([p.to_feature_vector() for p in profiles1])
        features2 = np.array([p.to_feature_vector() for p in profiles2])

        # Normalize features
        features1_norm = self._normalize_features(features1)
        features2_norm = self._normalize_features(features2)

        # Calculate similarity using multiple metrics
        similarity_scores = []

        # 1. Feature vector cosine similarity
        cosine_sim = self._cosine_similarity(features1_norm, features2_norm)
        similarity_scores.append(cosine_sim)

        # 2. Area correlation
        area_corr = self._correlation(features1[:, 0], features2[:, 0])
        similarity_scores.append(area_corr)

        # 3. Shape descriptor distance
        descriptor_sim = 1.0 / (1.0 + np.mean(np.abs(features1_norm - features2_norm)))
        similarity_scores.append(descriptor_sim)

        # Weighted average of similarity metrics
        weights = [0.4, 0.3, 0.3]
        overall_similarity = sum(s * w for s, w in zip(similarity_scores, weights))

        return overall_similarity

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize feature vectors to [0, 1] range."""
        normalized = np.zeros_like(features)

        for i in range(features.shape[1]):
            col = features[:, i]
            min_val, max_val = col.min(), col.max()

            if max_val - min_val > 0:
                normalized[:, i] = (col - min_val) / (max_val - min_val)
            else:
                normalized[:, i] = col

        return normalized

    def _cosine_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate average cosine similarity between feature vectors."""
        similarities = []

        for f1, f2 in zip(features1, features2):
            norm1 = np.linalg.norm(f1)
            norm2 = np.linalg.norm(f2)

            if norm1 > 0 and norm2 > 0:
                sim = np.dot(f1, f2) / (norm1 * norm2)
                similarities.append(sim)

        return np.mean(similarities) if similarities else 0.0

    def _correlation(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        """Calculate correlation coefficient between two arrays."""
        if len(arr1) < 2 or len(arr2) < 2:
            return 0.0

        corr_matrix = np.corrcoef(arr1, arr2)
        return abs(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0

    def match_shapes(self, obj1: bpy.types.Object, obj2: bpy.types.Object,
                    axis: str = 'Z') -> Dict[str, any]:
        """
        Match two 3D shapes using slice-based comparison.

        Args:
            obj1: First Blender mesh object
            obj2: Second Blender mesh object
            axis: Axis to slice along

        Returns:
            Dictionary containing similarity score and detailed results
        """
        # Slice both meshes
        profiles1 = self.slice_mesh(obj1, axis)
        profiles2 = self.slice_mesh(obj2, axis)

        # Compare profiles
        similarity = self.compare_profiles(profiles1, profiles2)

        return {
            'similarity': similarity,
            'num_slices_obj1': len(profiles1),
            'num_slices_obj2': len(profiles2),
            'match_quality': 'High' if similarity > 0.8 else 'Medium' if similarity > 0.5 else 'Low'
        }


def test_slice_matcher():
    """Test the slice-based shape matcher with objects in the current scene."""
    if len(bpy.context.selected_objects) < 2:
        print("Please select at least 2 mesh objects to compare")
        return

    obj1, obj2 = bpy.context.selected_objects[:2]

    if obj1.type != 'MESH' or obj2.type != 'MESH':
        print("Both objects must be mesh objects")
        return

    matcher = SliceBasedShapeMatcher(num_slices=30)
    result = matcher.match_shapes(obj1, obj2)

    print(f"\n{'='*50}")
    print(f"Shape Matching Results")
    print(f"{'='*50}")
    print(f"Object 1: {obj1.name}")
    print(f"Object 2: {obj2.name}")
    print(f"Similarity Score: {result['similarity']:.3f}")
    print(f"Match Quality: {result['match_quality']}")
    print(f"Slices analyzed: {result['num_slices_obj1']} vs {result['num_slices_obj2']}")
    print(f"{'='*50}\n")

    return result


if __name__ == "__main__":
    test_slice_matcher()
