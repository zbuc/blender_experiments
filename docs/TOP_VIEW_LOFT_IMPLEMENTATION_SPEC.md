# Top-View Constrained Lofting - Implementation Specification

## Overview

This specification details the exact changes needed to implement top-view constrained lofting for the `loft_profile` reconstruction mode. This will fix the circular top-view issue for non-rotationally-symmetric objects like cubes, stars, and cars.

---

## Phase 1: Data Models & Utilities (Foundation)

### 1.1 New File: `blender_blocking/geometry/contour_models.py`

**Location**: Create new file
**Lines**: ~60 lines
**Purpose**: Data contracts for contour-based lofting

```python
"""Data models for contour-based loft reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np


@dataclass(frozen=True)
class ContourTemplate:
    """Normalized 2D contour template for loft cross-sections.

    Contour points are normalized to unit square [-0.5, 0.5] with
    consistent vertex count for ring bridging.
    """

    points: np.ndarray  # (N, 2) array of normalized (x, y) coordinates
    num_vertices: int
    source_view: str = "top"
    original_bbox: Optional[tuple] = None  # (x, y, w, h) in pixels

    def __post_init__(self):
        if self.points.shape[1] != 2:
            raise ValueError("ContourTemplate points must be (N, 2)")
        if len(self.points) != self.num_vertices:
            raise ValueError(
                f"Point count {len(self.points)} != num_vertices {self.num_vertices}"
            )


@dataclass(frozen=True)
class ContourSlice:
    """Single loft slice using contour template with scale factors.

    The contour template is scaled by (scale_x, scale_y) at height z.
    """

    z: float
    scale_x: float  # Width scale factor (from front view profile)
    scale_y: float  # Depth scale factor (from side view profile)
    cx: float = 0.0  # Center X offset
    cy: float = 0.0  # Center Y offset

    def __post_init__(self):
        if self.scale_x < 0 or self.scale_y < 0:
            raise ValueError("Scale factors must be non-negative")
```

---

### 1.2 New File: `blender_blocking/geometry/contour_utils.py`

**Location**: Create new file
**Lines**: ~150 lines
**Purpose**: Contour manipulation utilities

```python
"""Utilities for contour normalization, resampling, and scaling."""

from __future__ import annotations

import numpy as np
import cv2
from typing import Tuple

from geometry.contour_models import ContourTemplate


def normalize_contour(
    contour: np.ndarray,
    bbox: Tuple[int, int, int, int] = None,
) -> np.ndarray:
    """
    Normalize contour to unit square [-0.5, 0.5] centered at origin.

    Args:
        contour: OpenCV contour (N, 1, 2) or (N, 2)
        bbox: Optional pre-computed bounding box (x, y, w, h)

    Returns:
        Normalized contour (N, 2) in range [-0.5, 0.5]
    """
    # Reshape to (N, 2) if needed
    if contour.ndim == 3 and contour.shape[1] == 1:
        contour = contour.squeeze(1)

    contour = contour.astype(np.float32)

    # Get bounding box
    if bbox is None:
        x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
    else:
        x, y, w, h = bbox

    # Center at origin
    cx = x + w / 2.0
    cy = y + h / 2.0
    centered = contour - np.array([[cx, cy]], dtype=np.float32)

    # Scale to unit square [-0.5, 0.5]
    scale = max(w, h)
    if scale > 0:
        normalized = centered / scale
    else:
        normalized = centered

    return normalized


def resample_contour_uniform(
    contour: np.ndarray,
    num_points: int,
) -> np.ndarray:
    """
    Resample contour to exactly num_points uniformly spaced by arc length.

    This ensures consistent vertex correspondence between loft slices.

    Args:
        contour: Input contour (N, 2)
        num_points: Target number of vertices

    Returns:
        Resampled contour (num_points, 2)
    """
    if len(contour) < 2:
        raise ValueError("Contour must have at least 2 points")

    # Ensure closed contour for arc length calculation
    contour_closed = np.vstack([contour, contour[0:1]])

    # Calculate cumulative arc length
    distances = np.sqrt(
        np.sum(np.diff(contour_closed, axis=0)**2, axis=1)
    )
    cumulative = np.concatenate([[0], np.cumsum(distances)])
    perimeter = cumulative[-1]

    if perimeter == 0:
        # Degenerate contour, return duplicated first point
        return np.tile(contour[0:1], (num_points, 1))

    # Target arc lengths for new points
    target_lengths = np.linspace(0, perimeter, num_points, endpoint=False)

    # Interpolate x and y coordinates
    resampled = np.zeros((num_points, 2), dtype=np.float32)
    resampled[:, 0] = np.interp(
        target_lengths,
        cumulative,
        contour_closed[:, 0]
    )
    resampled[:, 1] = np.interp(
        target_lengths,
        cumulative,
        contour_closed[:, 1]
    )

    return resampled


def create_contour_template(
    contour: np.ndarray,
    num_vertices: int,
    source_view: str = "top",
) -> ContourTemplate:
    """
    Create a normalized, resampled contour template.

    Args:
        contour: Raw contour from image processing (N, 1, 2) or (N, 2)
        num_vertices: Number of vertices for resampling
        source_view: View name (for metadata)

    Returns:
        ContourTemplate ready for loft mesh generation
    """
    # Get bounding box for metadata
    if contour.ndim == 3:
        contour_2d = contour.squeeze(1)
    else:
        contour_2d = contour

    x, y, w, h = cv2.boundingRect(contour_2d.astype(np.int32))

    # Normalize and resample
    normalized = normalize_contour(contour, bbox=(x, y, w, h))
    resampled = resample_contour_uniform(normalized, num_vertices)

    return ContourTemplate(
        points=resampled,
        num_vertices=num_vertices,
        source_view=source_view,
        original_bbox=(x, y, w, h),
    )


def scale_contour_2d(
    normalized_contour: np.ndarray,
    scale_x: float,
    scale_y: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> np.ndarray:
    """
    Scale normalized contour by profile factors.

    Args:
        normalized_contour: Contour in unit square [-0.5, 0.5]
        scale_x: Width scale factor (rx from front view)
        scale_y: Depth scale factor (ry from side view)
        center_x: X offset in world units
        center_y: Y offset in world units

    Returns:
        Scaled contour in world coordinates (N, 2)
    """
    scaled = normalized_contour.copy()

    # Scale from [-0.5, 0.5] to actual dimensions
    # Multiply by 2 because normalized range is 1.0 wide but represents full diameter
    scaled[:, 0] = scaled[:, 0] * scale_x * 2.0 + center_x
    scaled[:, 1] = scaled[:, 1] * scale_y * 2.0 + center_y

    return scaled
```

---

### 1.3 Modified File: `blender_blocking/config.py`

**Location**: Lines 95-136 (LoftMeshOptions class)
**Changes**: Add new configuration fields

**Before** (lines 95-109):
```python
@dataclass
class LoftMeshOptions:
    """Configuration for loft mesh generation."""

    radial_segments: int = 24
    cap_mode: str = "fan"
    min_radius_u: float = 0.0
    merge_threshold_u: float = 0.0
    recalc_normals: bool = True
    shade_smooth: bool = True
    weld_degenerate_rings: bool = True
    apply_decimation: bool = True
    decimate_ratio: float = 0.1
    decimate_method: str = "COLLAPSE"
```

**After** (lines 95-114):
```python
@dataclass
class LoftMeshOptions:
    """Configuration for loft mesh generation."""

    radial_segments: int = 24
    cap_mode: str = "fan"
    min_radius_u: float = 0.0
    merge_threshold_u: float = 0.0
    recalc_normals: bool = True
    shade_smooth: bool = True
    weld_degenerate_rings: bool = True
    apply_decimation: bool = True
    decimate_ratio: float = 0.1
    decimate_method: str = "COLLAPSE"

    # Top-view contour lofting
    use_top_contour: bool = True  # Use top-view for cross-section shape
    contour_simplify_epsilon: float = 0.001  # Douglas-Peucker simplification
    fallback_to_elliptical: bool = True  # Fallback if top-view unavailable
```

**Location**: Lines 110-121 (validate method)
**Changes**: Add validation for new fields

**Add after line 121**:
```python
        if self.contour_simplify_epsilon < 0:
            raise ValueError("contour_simplify_epsilon must be >= 0")
```

**Location**: Lines 123-136 (to_dict method)
**Changes**: Include new fields in serialization

**Add after line 135** (before closing brace):
```python
            "use_top_contour": self.use_top_contour,
            "contour_simplify_epsilon": self.contour_simplify_epsilon,
            "fallback_to_elliptical": self.fallback_to_elliptical,
```

---

## Phase 2: Mesh Generation (Core Implementation)

### 2.1 New File: `blender_blocking/integration/blender_ops/contour_loft_mesh.py`

**Location**: Create new file
**Lines**: ~200 lines
**Purpose**: Contour-based mesh generation

```python
"""Loft mesh generation using top-view contour templates."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from geometry.profile_models import EllipticalSlice
from geometry.contour_models import ContourTemplate, ContourSlice
from geometry.contour_utils import scale_contour_2d

try:
    import bpy
    import bmesh
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False


def _ring_vertices_from_contour(
    bm: "bmesh.types.BMesh",
    slice_data: ContourSlice,
    template: ContourTemplate,
    min_radius_u: float,
    weld_degenerate_rings: bool,
) -> Tuple[List["bmesh.types.BMVert"], bool]:
    """
    Generate ring vertices using contour template scaled by profile factors.

    Args:
        bm: BMesh instance
        slice_data: ContourSlice with z position and scale factors
        template: Normalized contour template
        min_radius_u: Minimum radius for welding
        weld_degenerate_rings: Whether to collapse degenerate rings to points

    Returns:
        (vertices, is_degenerate)
    """
    # Check for degenerate slice
    if weld_degenerate_rings and (
        slice_data.scale_x <= min_radius_u or slice_data.scale_y <= min_radius_u
    ):
        vert = bm.verts.new((slice_data.cx, slice_data.cy, slice_data.z))
        return [vert], True

    # Scale template contour by profile factors
    scaled_2d = scale_contour_2d(
        template.points,
        scale_x=slice_data.scale_x,
        scale_y=slice_data.scale_y,
        center_x=slice_data.cx,
        center_y=slice_data.cy,
    )

    # Create 3D vertices at height z
    verts = []
    for x, y in scaled_2d:
        verts.append(bm.verts.new((x, y, slice_data.z)))

    return verts, False


def _bridge_rings(
    bm: "bmesh.types.BMesh",
    ring_a: List["bmesh.types.BMVert"],
    ring_b: List["bmesh.types.BMVert"],
) -> None:
    """Bridge two vertex rings (identical to elliptical version)."""
    if len(ring_a) == 1 and len(ring_b) == 1:
        return

    if len(ring_a) == 1:
        center = ring_a[0]
        for i in range(len(ring_b)):
            v1 = ring_b[i]
            v2 = ring_b[(i + 1) % len(ring_b)]
            bm.faces.new((center, v1, v2))
        return

    if len(ring_b) == 1:
        center = ring_b[0]
        for i in range(len(ring_a)):
            v1 = ring_a[i]
            v2 = ring_a[(i + 1) % len(ring_a)]
            bm.faces.new((v1, v2, center))
        return

    count = min(len(ring_a), len(ring_b))
    for i in range(count):
        v1 = ring_a[i]
        v2 = ring_a[(i + 1) % count]
        v3 = ring_b[(i + 1) % count]
        v4 = ring_b[i]
        try:
            bm.faces.new((v1, v2, v3, v4))
        except ValueError:
            continue


def _cap_ring(
    bm: "bmesh.types.BMesh",
    ring: List["bmesh.types.BMVert"],
    cap_mode: str,
) -> None:
    """Cap a ring (identical to elliptical version)."""
    if len(ring) < 3:
        return

    if cap_mode == "fan":
        center_coords = [sum(v.co[i] for v in ring) / len(ring) for i in range(3)]
        center = bm.verts.new(center_coords)
        for i in range(len(ring)):
            v1 = ring[i]
            v2 = ring[(i + 1) % len(ring)]
            try:
                bm.faces.new((center, v1, v2))
            except ValueError:
                continue
        return

    if cap_mode == "ngon":
        try:
            bm.faces.new(ring)
        except ValueError:
            edges = []
            for i in range(len(ring)):
                v1 = ring[i]
                v2 = ring[(i + 1) % len(ring)]
                try:
                    edges.append(bm.edges.new((v1, v2)))
                except ValueError:
                    continue
            if edges:
                bmesh.ops.triangle_fill(bm, edges=edges, use_beauty=True)
        return

    if cap_mode != "none":
        raise ValueError(f"Unknown cap_mode: {cap_mode}")


def create_contour_loft_mesh(
    slices: Sequence[ContourSlice],
    template: ContourTemplate,
    *,
    name: str = "ContourLoftMesh",
    cap_mode: str = "fan",
    min_radius_u: float = 0.0,
    merge_threshold_u: float = 0.0,
    recalc_normals: bool = True,
    shade_smooth: bool = True,
    weld_degenerate_rings: bool = True,
) -> Optional[object]:
    """
    Create Blender mesh object lofted from contour template.

    Args:
        slices: Sequence of ContourSlice with z positions and scale factors
        template: Normalized contour template for cross-section shape
        name: Mesh object name
        cap_mode: Capping strategy ("fan", "ngon", "none")
        min_radius_u: Minimum radius for welding degenerate rings
        merge_threshold_u: Distance threshold for vertex merging
        recalc_normals: Recalculate face normals
        shade_smooth: Enable smooth shading
        weld_degenerate_rings: Collapse degenerate rings to points

    Returns:
        Blender mesh object or None on error
    """
    if not BLENDER_AVAILABLE:
        print("Warning: Blender API not available")
        return None

    if not slices:
        raise ValueError("slices must not be empty")

    if template.num_vertices < 3:
        raise ValueError("template must have at least 3 vertices")

    bm = bmesh.new()

    # Generate rings using contour template
    rings: List[List[bmesh.types.BMVert]] = []
    for slice_data in slices:
        ring, _ = _ring_vertices_from_contour(
            bm,
            slice_data,
            template,
            min_radius_u,
            weld_degenerate_rings,
        )
        rings.append(ring)

    # Bridge consecutive rings
    for ring_a, ring_b in zip(rings[:-1], rings[1:]):
        _bridge_rings(bm, ring_a, ring_b)

    # Cap ends
    if cap_mode != "none":
        _cap_ring(bm, rings[0], cap_mode)
        _cap_ring(bm, rings[-1], cap_mode)

    # Merge close vertices
    if merge_threshold_u > 0:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_threshold_u)

    # Recalculate normals
    if recalc_normals:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    # Create mesh and object
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    # Apply smooth shading
    if shade_smooth:
        for polygon in obj.data.polygons:
            polygon.use_smooth = True

    return obj
```

---

## Phase 3: Integration (Wiring It Together)

### 3.1 Modified File: `blender_blocking/main_integration.py`

**Location**: Lines 594-745 (create_3d_blockout_loft method)
**Changes**: Integrate top-view contour extraction and routing

**Add imports** (after line 83):
```python
from geometry.contour_utils import create_contour_template
from geometry.contour_models import ContourSlice
from integration.blender_ops.contour_loft_mesh import create_contour_loft_mesh
```

**Modify create_3d_blockout_loft method** starting at line 626:

**Before** (lines 626-641):
```python
        warnings: List[str] = []
        front_mask = None
        side_mask = None

        if "front" in self.views:
            try:
                front_mask = extract_binary_silhouette(self.views["front"])
            except Exception as exc:
                warnings.append(f"Failed to extract front silhouette: {exc}")

        if "side" in self.views:
            try:
                side_mask = extract_binary_silhouette(self.views["side"])
            except Exception as exc:
                warnings.append(f"Failed to extract side silhouette: {exc}")
```

**After** (lines 626-650):
```python
        warnings: List[str] = []
        front_mask = None
        side_mask = None
        top_mask = None
        top_contour_template = None

        if "front" in self.views:
            try:
                front_mask = extract_binary_silhouette(self.views["front"])
            except Exception as exc:
                warnings.append(f"Failed to extract front silhouette: {exc}")

        if "side" in self.views:
            try:
                side_mask = extract_binary_silhouette(self.views["side"])
            except Exception as exc:
                warnings.append(f"Failed to extract side silhouette: {exc}")

        # Extract top-view contour template if available and enabled
        if "top" in self.views and self.config.mesh_from_profile.use_top_contour:
            try:
                top_mask = extract_binary_silhouette(self.views["top"])
                from integration.shape_matching.contour_analyzer import find_contours
                import cv2

                # Find largest contour in top view
                contours, _ = find_contours(
                    (top_mask.astype(np.uint8) * 255),
                    mode="external",
                    return_hierarchy=True,
                )

                if contours:
                    # Use largest contour
                    largest_contour = max(contours, key=cv2.contourArea)

                    # Simplify if configured
                    epsilon = self.config.mesh_from_profile.contour_simplify_epsilon
                    if epsilon > 0:
                        perimeter = cv2.arcLength(largest_contour, True)
                        largest_contour = cv2.approxPolyDP(
                            largest_contour,
                            epsilon * perimeter,
                            True
                        )

                    # Create normalized template
                    top_contour_template = create_contour_template(
                        largest_contour,
                        num_vertices=self.config.mesh_from_profile.radial_segments,
                        source_view="top",
                    )
                    print(f"  Extracted top-view contour: {top_contour_template.num_vertices} vertices")
                else:
                    warnings.append("No contours found in top view")

            except Exception as exc:
                warnings.append(f"Failed to extract top contour: {exc}")
                if not self.config.mesh_from_profile.fallback_to_elliptical:
                    raise
```

**Add new logic after profile generation** (insert after line 676):

```python
        # Choose mesh generation strategy based on top-view availability
        use_contour_lofting = (
            top_contour_template is not None
            and self.config.mesh_from_profile.use_top_contour
        )

        if use_contour_lofting:
            print("  Using top-view constrained lofting (contour-based)")
        else:
            print("  Using elliptical lofting (circular cross-sections)")
```

**Modify mesh creation** (replace lines 696-706):

**Before**:
```python
        final_mesh = create_loft_mesh_from_slices(
            slices,
            name="Blockout_Mesh",
            radial_segments=self.config.mesh_from_profile.radial_segments,
            cap_mode=self.config.mesh_from_profile.cap_mode,
            min_radius_u=self.config.mesh_from_profile.min_radius_u,
            merge_threshold_u=self.config.mesh_from_profile.merge_threshold_u,
            recalc_normals=self.config.mesh_from_profile.recalc_normals,
            shade_smooth=self.config.mesh_from_profile.shade_smooth,
            weld_degenerate_rings=self.config.mesh_from_profile.weld_degenerate_rings,
        )
```

**After**:
```python
        if use_contour_lofting:
            # Convert elliptical slices to contour slices
            contour_slices = [
                ContourSlice(
                    z=s.z,
                    scale_x=s.rx,
                    scale_y=s.ry,
                    cx=s.cx if s.cx is not None else 0.0,
                    cy=s.cy if s.cy is not None else 0.0,
                )
                for s in slices
            ]

            final_mesh = create_contour_loft_mesh(
                contour_slices,
                top_contour_template,
                name="Blockout_Mesh",
                cap_mode=self.config.mesh_from_profile.cap_mode,
                min_radius_u=self.config.mesh_from_profile.min_radius_u,
                merge_threshold_u=self.config.mesh_from_profile.merge_threshold_u,
                recalc_normals=self.config.mesh_from_profile.recalc_normals,
                shade_smooth=self.config.mesh_from_profile.shade_smooth,
                weld_degenerate_rings=self.config.mesh_from_profile.weld_degenerate_rings,
            )
        else:
            # Fallback to elliptical lofting
            final_mesh = create_loft_mesh_from_slices(
                slices,
                name="Blockout_Mesh",
                radial_segments=self.config.mesh_from_profile.radial_segments,
                cap_mode=self.config.mesh_from_profile.cap_mode,
                min_radius_u=self.config.mesh_from_profile.min_radius_u,
                merge_threshold_u=self.config.mesh_from_profile.merge_threshold_u,
                recalc_normals=self.config.mesh_from_profile.recalc_normals,
                shade_smooth=self.config.mesh_from_profile.shade_smooth,
                weld_degenerate_rings=self.config.mesh_from_profile.weld_degenerate_rings,
            )
```

---

## Phase 4: Configuration Updates

### 4.1 Config Files

**Location**: All `configs/loft_profile-*.json` files
**Changes**: Add new configuration fields

**Files to modify**:
- `configs/loft_profile-default.json`
- `configs/loft_profile-higher.json`
- `configs/loft_profile-ultra.json`
- `configs/loft_profile-extreme-ultra.json`

**Add to `mesh_from_profile` section**:
```json
{
  "mesh_from_profile": {
    "radial_segments": 64,
    "cap_mode": "fan",
    "min_radius_u": 0.0005,
    "merge_threshold_u": 0.0005,
    "recalc_normals": true,
    "shade_smooth": true,
    "weld_degenerate_rings": true,
    "apply_decimation": true,
    "decimate_ratio": 0.1,
    "decimate_method": "COLLAPSE",
    "use_top_contour": true,
    "contour_simplify_epsilon": 0.001,
    "fallback_to_elliptical": true
  }
}
```

---

## Phase 5: Testing

### 5.1 New File: `blender_blocking/test_contour_loft.py`

**Location**: Create new file
**Lines**: ~150 lines
**Purpose**: Unit tests for contour utilities and mesh generation

```python
"""Unit tests for contour-based lofting."""

import unittest
import numpy as np

from geometry.contour_utils import (
    normalize_contour,
    resample_contour_uniform,
    scale_contour_2d,
    create_contour_template,
)
from geometry.contour_models import ContourSlice


class TestContourNormalization(unittest.TestCase):
    """Test contour normalization to unit square."""

    def test_normalize_square(self):
        """Verify square normalizes to [-0.5, 0.5]."""
        square = np.array([
            [0, 0],
            [100, 0],
            [100, 100],
            [0, 100],
        ], dtype=np.float32)

        normalized = normalize_contour(square)

        expected = np.array([
            [-0.5, -0.5],
            [0.5, -0.5],
            [0.5, 0.5],
            [-0.5, 0.5],
        ], dtype=np.float32)

        np.testing.assert_array_almost_equal(normalized, expected, decimal=5)

    def test_normalize_rectangle(self):
        """Verify rectangle maintains aspect ratio."""
        rect = np.array([
            [0, 0],
            [200, 0],
            [200, 100],
            [0, 100],
        ], dtype=np.float32)

        normalized = normalize_contour(rect)

        # Width should be 1.0, height should be 0.5
        width = normalized[:, 0].max() - normalized[:, 0].min()
        height = normalized[:, 1].max() - normalized[:, 1].min()

        self.assertAlmostEqual(width, 1.0, places=5)
        self.assertAlmostEqual(height, 0.5, places=5)


class TestContourResampling(unittest.TestCase):
    """Test uniform contour resampling."""

    def test_resample_count(self):
        """Verify resampling produces exact vertex count."""
        square = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ], dtype=np.float32)

        for num_points in [4, 8, 16, 32, 64]:
            resampled = resample_contour_uniform(square, num_points)
            self.assertEqual(len(resampled), num_points)

    def test_resample_preserves_shape(self):
        """Verify resampling maintains shape geometry."""
        square = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ], dtype=np.float32)

        resampled = resample_contour_uniform(square, 100)

        # Check bounding box is preserved
        self.assertAlmostEqual(resampled[:, 0].min(), 0.0, places=2)
        self.assertAlmostEqual(resampled[:, 0].max(), 1.0, places=2)
        self.assertAlmostEqual(resampled[:, 1].min(), 0.0, places=2)
        self.assertAlmostEqual(resampled[:, 1].max(), 1.0, places=2)


class TestContourScaling(unittest.TestCase):
    """Test contour scaling by profile factors."""

    def test_scale_uniform(self):
        """Verify uniform scaling."""
        normalized = np.array([
            [-0.5, -0.5],
            [0.5, -0.5],
            [0.5, 0.5],
            [-0.5, 0.5],
        ], dtype=np.float32)

        scaled = scale_contour_2d(normalized, scale_x=2.0, scale_y=2.0)

        expected = np.array([
            [-2.0, -2.0],
            [2.0, -2.0],
            [2.0, 2.0],
            [-2.0, 2.0],
        ], dtype=np.float32)

        np.testing.assert_array_almost_equal(scaled, expected, decimal=5)

    def test_scale_non_uniform(self):
        """Verify non-uniform scaling (ellipse)."""
        normalized = np.array([
            [-0.5, -0.5],
            [0.5, -0.5],
            [0.5, 0.5],
            [-0.5, 0.5],
        ], dtype=np.float32)

        scaled = scale_contour_2d(normalized, scale_x=2.0, scale_y=1.0)

        # Width should be 4.0, height should be 2.0
        width = scaled[:, 0].max() - scaled[:, 0].min()
        height = scaled[:, 1].max() - scaled[:, 1].min()

        self.assertAlmostEqual(width, 4.0, places=5)
        self.assertAlmostEqual(height, 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
```

### 5.2 Modified File: `blender_blocking/test_e2e_validation.py`

**Location**: No changes needed - existing test will automatically use new mode when configs are updated
**Verification**: Cube test case should achieve IoU > 0.95 after implementation

---

## Implementation Timeline

### Day 1: Foundation (Data Models & Utilities)
- [ ] Create `geometry/contour_models.py` (1 hour)
- [ ] Create `geometry/contour_utils.py` (2 hours)
- [ ] Update `config.py` LoftMeshOptions (30 min)
- [ ] Write unit tests for contour utilities (1.5 hours)
- [ ] Run tests and fix issues (1 hour)

**Deliverable**: ContourTemplate and utilities working, all tests passing

### Day 2: Core Implementation (Mesh Generation)
- [ ] Create `contour_loft_mesh.py` (2 hours)
- [ ] Integrate into `main_integration.py` (2 hours)
- [ ] Update config files (30 min)
- [ ] Basic smoke testing (1.5 hours)

**Deliverable**: Top-view constrained lofting functional end-to-end

### Day 3: Testing & Refinement
- [ ] Run E2E tests with cube/star/car (1 hour)
- [ ] Fix edge cases and bugs (2 hours)
- [ ] Performance optimization (1 hour)
- [ ] Documentation updates (1 hour)
- [ ] Final validation (1 hour)

**Deliverable**: Production-ready feature, IoU > 0.95 for cube test case

---

## Rollback Plan

If issues arise, the feature can be disabled via config:

```json
{
  "mesh_from_profile": {
    "use_top_contour": false
  }
}
```

This will revert to elliptical lofting with no code changes needed.

---

## Verification Checklist

- [ ] All unit tests passing
- [ ] Cube test case IoU > 0.95
- [ ] Star test case IoU > 0.85
- [ ] Car test case improved IoU
- [ ] Bottle/vase tests still passing (>0.95)
- [ ] No regressions in existing test cases
- [ ] Config fallback working (use_top_contour=false)
- [ ] Error handling for missing top view
- [ ] Performance acceptable (< 10% slowdown)
- [ ] Documentation complete

---

## Success Metrics

| Test Case | Current IoU | Target IoU | Status |
|-----------|-------------|------------|--------|
| Cube      | 0.929       | > 0.95     | 🎯 Primary |
| Star      | 0.827       | > 0.90     | 🎯 Primary |
| Car       | 0.756       | > 0.80     | Secondary |
| Bottle    | 0.994       | > 0.99     | Maintain |
| Vase      | 0.886       | > 0.88     | Maintain |

---

## File Summary

**New Files (5)**:
1. `blender_blocking/geometry/contour_models.py` (~60 lines)
2. `blender_blocking/geometry/contour_utils.py` (~150 lines)
3. `blender_blocking/integration/blender_ops/contour_loft_mesh.py` (~200 lines)
4. `blender_blocking/test_contour_loft.py` (~150 lines)
5. `docs/TOP_VIEW_LOFT_IMPLEMENTATION_SPEC.md` (this file)

**Modified Files (7)**:
1. `blender_blocking/config.py` (lines 95-136: +3 fields, +3 validation, +3 serialization)
2. `blender_blocking/main_integration.py` (lines 83: +3 imports, lines 626-710: +60 lines logic)
3. `configs/loft_profile-default.json` (+3 fields)
4. `configs/loft_profile-higher.json` (+3 fields)
5. `configs/loft_profile-ultra.json` (+3 fields)
6. `configs/loft_profile-extreme-ultra.json` (+3 fields)
7. `docs/LOFT_PROFILE_IMPROVEMENTS.md` (reference documentation)

**Total New Code**: ~560 lines
**Total Modified Code**: ~90 lines
**Total Test Code**: ~150 lines
