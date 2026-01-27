# Loft Profile Mode: Non-Circular Cross-Section Support

## Problem Statement

The current `loft_profile` reconstruction mode produces **circular/elliptical cross-sections** for all objects, causing significant IoU degradation for objects with non-circular profiles:

- **Cube test case**: Front and side views render correctly (squares), but top view is a circle
  - Expected: Square top view
  - Actual: Circular top view
  - IoU impact: ~0.93 overall (would be ~0.99+ with correct top view)

**Root cause** (lines 36-40 of `profile_loft_mesh.py`):
```python
for i in range(radial_segments):
    theta = (2.0 * math.pi * i) / radial_segments
    x = center_x + rx * math.cos(theta)  # ← Forced circular topology
    y = center_y + ry * math.sin(theta)  # ← Forced circular topology
```

## Proposed Solutions

### Solution 1: Top-View Constrained Lofting ⭐ RECOMMENDED

**Approach**: Use top-view silhouette contour as the cross-section template, scaled by front/side profiles.

#### Algorithm

```python
# 1. Extract and normalize top-view contour
top_mask = extract_binary_silhouette(self.views["top"])
top_contour = find_largest_contour(top_mask)
normalized_template = normalize_to_unit_square(top_contour)

# 2. Resample to consistent vertex count (critical for bridging)
template = resample_uniform(normalized_template, num_points=radial_segments)

# 3. At each height z, scale template by front/side profile factors
for z, rx, ry in profile_slices:
    scaled_contour = scale_2d(template, scale_x=rx, scale_y=ry)
    ring_verts = [(x, y, z) for x, y in scaled_contour]
    rings.append(ring_verts)

# 4. Bridge rings (existing logic unchanged)
```

#### Implementation Requirements

1. **New data model** (`geometry/profile_models.py`):
   ```python
   @dataclass(frozen=True)
   class ContourSlice:
       """Loft slice with arbitrary contour shape."""
       z: float
       contour_2d: np.ndarray  # (N, 2) normalized contour
       scale_x: float          # Width scale from front view
       scale_y: float          # Depth scale from side view
       cx: Optional[float] = None
       cy: Optional[float] = None
   ```

2. **Contour utilities** (`integration/shape_matching/contour_utils.py`):
   ```python
   def normalize_contour(contour: np.ndarray) -> np.ndarray:
       """Normalize contour to [-0.5, 0.5] unit square."""

   def resample_contour_uniform(contour: np.ndarray, num_points: int) -> np.ndarray:
       """Resample contour to exactly num_points uniformly spaced."""

   def scale_contour_2d(normalized: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
       """Scale normalized contour by profile factors."""
   ```

3. **Modified mesh generation** (`integration/blender_ops/contour_loft_mesh.py`):
   ```python
   def create_contour_loft_mesh(
       slices: Sequence[ContourSlice],
       top_contour: np.ndarray,
       radial_segments: int = 64,
       ...
   ) -> object:
       """Create loft mesh using top-view contour template."""
       template = resample_contour_uniform(
           normalize_contour(top_contour),
           num_points=radial_segments
       )

       rings = []
       for slice in slices:
           scaled_2d = scale_contour_2d(template, slice.scale_x, slice.scale_y)
           ring_3d = add_z_coordinate(scaled_2d, slice.z)
           rings.append(ring_3d)

       # Bridge rings (existing _bridge_rings logic works)
       for ring_a, ring_b in zip(rings[:-1], rings[1:]):
           _bridge_rings(bm, ring_a, ring_b)
   ```

4. **Integration** (`main_integration.py`):
   ```python
   def create_3d_blockout_loft(self, num_slices: int) -> object:
       # Extract top view if available
       top_mask = None
       if "top" in self.views:
           top_mask = extract_binary_silhouette(self.views["top"])

       # Build profile from front/side (existing logic)
       profile = build_elliptical_profile_from_views(...)

       # Choose mesh generation strategy
       if top_mask is not None and self.config.mesh_from_profile.use_top_contour:
           # New: Top-view constrained lofting
           top_contour = find_largest_contour(top_mask)
           slices = convert_to_contour_slices(profile, top_contour)
           mesh = create_contour_loft_mesh(slices, top_contour, ...)
       else:
           # Fallback: Elliptical lofting (existing)
           slices = sample_elliptical_slices(profile, ...)
           mesh = create_loft_mesh_from_slices(slices, ...)
   ```

#### Pros & Cons

**Pros**:
- ✅ Handles **any shape** with top-view reference (rectangles, stars, complex profiles)
- ✅ Preserves geometric accuracy for non-symmetric objects
- ✅ Uses all three views effectively
- ✅ Generalizes beyond specific shapes
- ✅ Fallback to elliptical mode when top view unavailable

**Cons**:
- ⚠️ More complex than elliptical math
- ⚠️ Requires robust contour correspondence (vertex ordering matters)
- ⚠️ Contour resampling needed for consistent ring topology
- ⚠️ Potential self-intersections if profile scales vary dramatically

#### Effort Estimate: **2-3 days**

- Day 1: Contour utilities and data models
- Day 2: Modified mesh generation
- Day 3: Integration, testing, and refinement

---

### Solution 2: Rectangular Profile Mode (Quick Fix)

**Approach**: Special-case axis-aligned rectangles with 4-vertex rings.

#### Implementation

```python
# Configuration
@dataclass
class LoftMeshOptions:
    profile_shape: str = "elliptical"  # "elliptical" | "rectangular"

# Mesh generation
def _ring_vertices_rectangular(slice_data, weld_degenerate):
    if weld_degenerate and (slice_data.rx < min_radius or slice_data.ry < min_radius):
        return [center_vert], True

    half_w = slice_data.rx
    half_d = slice_data.ry
    return [
        (-half_w, -half_d, slice_data.z),  # Bottom-left
        (+half_w, -half_d, slice_data.z),  # Bottom-right
        (+half_w, +half_d, slice_data.z),  # Top-right
        (-half_w, +half_d, slice_data.z),  # Top-left
    ], False
```

#### Pros & Cons

**Pros**:
- ✅ Simple implementation (< 50 lines)
- ✅ Fixes cube test case immediately
- ✅ Low risk (isolated change)
- ✅ Config-gated (easy to disable)

**Cons**:
- ❌ Only handles rectangles (not stars, hexagons, complex shapes)
- ❌ No automatic detection (user must specify)
- ❌ Doesn't generalize

#### Effort Estimate: **1 day**

---

### Solution 3: Polygonal Template Profiles

**Approach**: User-specified polygon template (rectangle, hexagon, octagon, N-gon).

#### Implementation

```python
# Config
profile_shape_template: str = "rectangle"  # or "hexagon", "octagon", "custom"

# Templates
def get_polygon_template(shape: str, num_sides: int = 4) -> np.ndarray:
    if shape == "rectangle":
        return np.array([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])
    elif shape == "hexagon":
        angles = np.linspace(0, 2*np.pi, 7)[:-1]
        return np.column_stack([np.cos(angles), np.sin(angles)]) * 0.5
    # ... etc
```

#### Pros & Cons

**Pros**:
- ✅ Supports multiple standard shapes
- ✅ User control over topology
- ✅ Bridges gap between elliptical and top-constrained

**Cons**:
- ⚠️ Requires user to specify shape (not automatic)
- ⚠️ Template library needed
- ⚠️ Limited to predefined shapes

#### Effort Estimate: **2 days**

---

## Recommended Implementation Strategy

### Phase 1: Quick Fix (1 day) - IMMEDIATE
Implement **Rectangular Mode** to validate approach and fix cube test case:
- Add `profile_shape: "rectangular"` config option
- 4-vertex ring generation
- Update test suite with cube using rectangular mode

**Deliverable**: Cube test case passes with IoU > 0.95

### Phase 2: General Solution (2-3 days) - SHORT TERM
Implement **Top-View Constrained Lofting**:
- Contour extraction and normalization utilities
- ContourSlice data model
- Modified mesh generation with contour template
- Automatic fallback to elliptical mode

**Deliverable**: All test cases (cube, star, car) achieve high IoU with top-view mode

### Phase 3: Hybrid Mode (1 day) - MEDIUM TERM
Automatic mode selection based on shape analysis:
```python
def select_loft_mode(front_mask, side_mask, top_mask) -> str:
    if top_mask is None:
        return "elliptical"

    circularity = compute_circularity(top_mask)
    if circularity > 0.85:  # Nearly circular
        return "elliptical"
    else:
        return "top_constrained"
```

**Deliverable**: Smart automatic mode selection, no user configuration needed

---

## Risk Analysis

### Low Risk
- ✅ Rectangular mode (isolated, fallback available)
- ✅ Config flags (can disable problematic features)
- ✅ Existing elliptical mode preserved

### Medium Risk
- ⚠️ Contour resampling (must maintain vertex correspondence)
- ⚠️ Self-intersections (dramatic scale changes between slices)
- ⚠️ Edge cases (degenerate contours, holes)

### Mitigation Strategies
1. **Extensive testing**: Add test cases for rectangles, stars, hexagons
2. **Fallback logic**: Revert to elliptical on errors
3. **Validation**: Check for self-intersections, warn user
4. **Progressive rollout**: Ship rectangular mode first, validate, then add contour mode

---

## Test Plan

### Unit Tests
```python
def test_normalize_contour_square():
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    normalized = normalize_contour(square)
    assert np.allclose(normalized, [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])

def test_resample_contour_uniform():
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    resampled = resample_contour_uniform(square, num_points=8)
    assert len(resampled) == 8
    # Verify uniform spacing along perimeter
```

### Integration Tests
```python
def test_cube_top_view_iou():
    """Verify cube top view is square, not circular."""
    workflow = BlockingWorkflow(
        front_path="test_images/cube_front.png",
        side_path="test_images/cube_side.png",
        top_path="test_images/cube_top.png",
        config=BlockingConfig(
            reconstruction_mode="loft_profile",
            mesh_from_profile=LoftMeshOptions(use_top_contour=True)
        )
    )
    mesh = workflow.run_full_workflow()

    # Render top view
    rendered = render_orthogonal_views(..., views=["top"])
    iou = compute_iou(reference="test_images/cube_top.png", rendered=rendered["top"])

    assert iou > 0.95, f"Cube top view IoU too low: {iou}"
```

### E2E Validation
Run multi-subject test suite with both modes:
- **Circular objects** (bottle, vase): Use elliptical mode
- **Non-circular objects** (cube, star, car): Use top-constrained mode
- Compare IoU scores before/after improvements

---

## Expected Results

### Cube Test Case
| Metric | Before | After (Rectangular) | After (Top-Constrained) |
|--------|--------|---------------------|-------------------------|
| Front IoU | 0.929 | 0.929 | 0.929 |
| Side IoU | 0.929 | 0.929 | 0.929 |
| Top IoU | **0.929** (circle) | **0.990+** (square) | **0.990+** (square) |
| Overall | 0.929 | 0.950+ | 0.950+ |

### Star Test Case
| Metric | Before | After |
|--------|--------|-------|
| Top IoU | ~0.70 (circular approximation) | **0.90+** (actual star shape) |

---

## Conclusion

**Top-View Constrained Lofting is highly feasible** and provides a general solution for non-circular profiles. The implementation complexity is manageable (2-3 days) and can be rolled out progressively:

1. **Start with rectangular mode** to validate the approach
2. **Implement top-constrained mode** for full generality
3. **Add automatic mode selection** for seamless UX

This will significantly improve IoU scores for objects like cubes, cars, and stars while preserving excellent results for bottles and vases.
