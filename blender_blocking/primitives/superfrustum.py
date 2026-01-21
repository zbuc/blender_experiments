"""
SuperFrustum primitive with differentiable SDF.

Based on arXiv:2512.09201 - Residual Primitive Fitting of 3D Shapes with SuperFrusta
Implementation uses signed distance fields (SDF) from Inigo Quilez's distance functions.

References:
- https://arxiv.org/abs/2512.09201
- https://iquilezles.org/articles/distfunctions/
"""

import numpy as np
from typing import Tuple, Optional
import math


class SuperFrustum:
    """
    SuperFrustum primitive with 8 parameters and differentiable SDF.

    Parameters (8 total):
        - position (x, y, z): 3 parameters
        - orientation (theta, phi): 2 parameters (spherical coordinates for axis direction)
        - radius_top: 1 parameter
        - radius_bottom: 1 parameter
        - height: 1 parameter

    Can represent:
        - Cylinder: radius_top = radius_bottom
        - Cone: radius_top = 0 OR radius_bottom = 0
        - Sphere: radius_top = radius_bottom = height/2 (approximately)
        - Tapered frustum: radius_top ≠ radius_bottom
    """

    def __init__(
        self,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: Tuple[float, float] = (0.0, 0.0),  # (theta, phi) in radians
        radius_bottom: float = 1.0,
        radius_top: float = 0.5,
        height: float = 2.0
    ):
        """
        Initialize SuperFrustum.

        Args:
            position: (x, y, z) center position
            orientation: (theta, phi) axis direction in spherical coordinates
                        theta: azimuthal angle (rotation around Z)
                        phi: polar angle (tilt from Z axis)
            radius_bottom: Radius at bottom cap
            radius_top: Radius at top cap
            height: Height along axis
        """
        self.position = np.array(position, dtype=np.float64)
        self.orientation = np.array(orientation, dtype=np.float64)
        self.radius_bottom = float(radius_bottom)
        self.radius_top = float(radius_top)
        self.height = float(height)

    def get_axis_vector(self) -> np.ndarray:
        """
        Compute axis direction vector from spherical coordinates.

        Returns:
            Unit vector (3D) representing axis direction
        """
        theta, phi = self.orientation
        # Convert spherical to Cartesian
        # Default axis is +Z (0, 0, 1) when theta=0, phi=0
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.array([x, y, z], dtype=np.float64)

    def sdf(self, point: np.ndarray) -> float:
        """
        Compute signed distance field at a point.

        Uses Inigo Quilez's capped cone/frustum SDF formula.

        Args:
            point: 3D point (x, y, z) as numpy array

        Returns:
            Signed distance (negative inside, positive outside, zero on surface)
        """
        # Transform point to local coordinate system
        # 1. Translate to origin
        p = point - self.position

        # 2. Rotate to align axis with Z
        axis = self.get_axis_vector()
        p_local = self._rotate_to_z_axis(p, axis)

        # 3. Apply capped cone SDF (from Inigo Quilez)
        # Cone is centered at origin, extends from -h/2 to +h/2 along Z
        r1 = self.radius_bottom
        r2 = self.radius_top
        h = self.height / 2.0  # Half-height for centered cone

        # Translate to cone's coordinate system (base at origin, extends upward)
        p_cone = p_local + np.array([0, 0, h])

        # Capped cone SDF (exact formula from IQ)
        q = np.array([np.linalg.norm(p_cone[:2]), p_cone[2]])
        k1 = np.array([r2, self.height])
        k2 = np.array([r2 - r1, 2.0 * self.height])

        ca = np.array([
            q[0] - min(q[0], r1 if q[1] < 0.0 else r2),
            abs(q[1]) - self.height
        ])

        cb = q - k1 + k2 * np.clip(
            np.dot(k1 - q, k2) / np.dot(k2, k2),
            0.0,
            1.0
        )

        s = -1.0 if (cb[0] < 0.0 and ca[1] < 0.0) else 1.0

        return s * np.sqrt(min(np.dot(ca, ca), np.dot(cb, cb)))

    def _rotate_to_z_axis(self, point: np.ndarray, axis: np.ndarray) -> np.ndarray:
        """
        Rotate point so that given axis aligns with Z axis.

        Args:
            point: 3D point to rotate
            axis: Current axis direction (unit vector)

        Returns:
            Rotated point
        """
        # Target axis is Z (0, 0, 1)
        z_axis = np.array([0.0, 0.0, 1.0])

        # If axis is already aligned with Z, no rotation needed
        if np.allclose(axis, z_axis):
            return point

        # If axis is opposite to Z, rotate 180 degrees around X
        if np.allclose(axis, -z_axis):
            return np.array([point[0], -point[1], -point[2]])

        # Compute rotation axis (cross product)
        rot_axis = np.cross(axis, z_axis)
        rot_axis = rot_axis / np.linalg.norm(rot_axis)

        # Compute rotation angle
        cos_angle = np.dot(axis, z_axis)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

        # Apply Rodrigues' rotation formula
        return self._rodrigues_rotation(point, rot_axis, angle)

    def _rodrigues_rotation(
        self,
        point: np.ndarray,
        axis: np.ndarray,
        angle: float
    ) -> np.ndarray:
        """
        Rotate point around axis by angle using Rodrigues' formula.

        Args:
            point: 3D point
            axis: Rotation axis (unit vector)
            angle: Rotation angle in radians

        Returns:
            Rotated point
        """
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Rodrigues' rotation formula
        # v_rot = v*cos(a) + (k×v)*sin(a) + k*(k·v)*(1-cos(a))
        return (
            point * cos_a +
            np.cross(axis, point) * sin_a +
            axis * np.dot(axis, point) * (1.0 - cos_a)
        )

    def gradient(self, point: np.ndarray, epsilon: float = 1e-5) -> dict:
        """
        Compute gradients of SDF with respect to all 8 parameters using finite differences.

        Args:
            point: 3D point where to compute gradients
            epsilon: Small perturbation for finite differences

        Returns:
            Dictionary with gradients for each parameter:
                - 'position': (3,) array - gradient w.r.t. position
                - 'orientation': (2,) array - gradient w.r.t. orientation
                - 'radius_bottom': float - gradient w.r.t. bottom radius
                - 'radius_top': float - gradient w.r.t. top radius
                - 'height': float - gradient w.r.t. height
        """
        grads = {}

        # Current SDF value
        f0 = self.sdf(point)

        # Gradient w.r.t. position (3 parameters)
        grad_pos = np.zeros(3)
        for i in range(3):
            pos_plus = self.position.copy()
            pos_plus[i] += epsilon
            old_pos = self.position.copy()
            self.position = pos_plus
            f_plus = self.sdf(point)
            self.position = old_pos
            grad_pos[i] = (f_plus - f0) / epsilon
        grads['position'] = grad_pos

        # Gradient w.r.t. orientation (2 parameters)
        grad_orient = np.zeros(2)
        for i in range(2):
            orient_plus = self.orientation.copy()
            orient_plus[i] += epsilon
            old_orient = self.orientation.copy()
            self.orientation = orient_plus
            f_plus = self.sdf(point)
            self.orientation = old_orient
            grad_orient[i] = (f_plus - f0) / epsilon
        grads['orientation'] = grad_orient

        # Gradient w.r.t. radius_bottom
        old_rb = self.radius_bottom
        self.radius_bottom += epsilon
        f_plus = self.sdf(point)
        self.radius_bottom = old_rb
        grads['radius_bottom'] = (f_plus - f0) / epsilon

        # Gradient w.r.t. radius_top
        old_rt = self.radius_top
        self.radius_top += epsilon
        f_plus = self.sdf(point)
        self.radius_top = old_rt
        grads['radius_top'] = (f_plus - f0) / epsilon

        # Gradient w.r.t. height
        old_h = self.height
        self.height += epsilon
        f_plus = self.sdf(point)
        self.height = old_h
        grads['height'] = (f_plus - f0) / epsilon

        return grads

    def to_dict(self) -> dict:
        """
        Export parameters as dictionary.

        Returns:
            Dictionary with all 8 parameters
        """
        return {
            'position': self.position.tolist(),
            'orientation': self.orientation.tolist(),
            'radius_bottom': self.radius_bottom,
            'radius_top': self.radius_top,
            'height': self.height
        }

    @classmethod
    def from_dict(cls, params: dict) -> 'SuperFrustum':
        """
        Create SuperFrustum from parameter dictionary.

        Args:
            params: Dictionary with parameters

        Returns:
            SuperFrustum instance
        """
        return cls(
            position=tuple(params['position']),
            orientation=tuple(params['orientation']),
            radius_bottom=params['radius_bottom'],
            radius_top=params['radius_top'],
            height=params['height']
        )

    def __repr__(self) -> str:
        return (
            f"SuperFrustum(pos={self.position}, "
            f"orient={self.orientation}, "
            f"r_bot={self.radius_bottom:.3f}, "
            f"r_top={self.radius_top:.3f}, "
            f"h={self.height:.3f})"
        )
