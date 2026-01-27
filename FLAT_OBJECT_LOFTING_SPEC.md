# Flat Object Contour Lofting: Implementation Specification

## Executive Summary

**Problem:** Top-view contour lofting fails for flat objects (star, dog, piss) because front/side profiles measure projected dimensions, not true cross-sectional dimensions. This causes incorrect squashing when scaling the contour.

**Root Cause:** For a flat star lying in XY plane:
- Top view shows 400×400px (true XY extent)
- Side view shows 400×20px (edge-on projection)
- Profile gives ry=0.1 world units
- Contour scaled by 2*ry=0.2 → squashed to 1/20th correct size!

**Solution:** Detect flat objects and use top-view contour's original pixel dimensions as the primary scale reference, adjusting only by height-based scaling factors.

**Impact:**
- Expected improvements: star 0.472 → 0.90+, dog 0.637 → 0.80+, piss 0.558 → 0.80+
- Risk: Must preserve cube's perfect 1.0 IoU (regression testing critical)

---

## Architecture Overview

### Current Flow (Works for Boxes, Fails for Flat Objects)
```
1. Extract top contour → Normalize to [-0.5, 0.5] (loses size info)
2. Extract front/side profiles → Get rx(z), ry(z) at each height
3. For each slice: Scale contour by (2*rx, 2*ry)
   ❌ For flat objects: rx correct, ry too small (projection != cross-section)
```

### Proposed Flow (Handles Both)
```
1. Extract top contour with original_bbox (w_px, h_px)
2. Extract front/side profiles → Get rx(z), ry(z) and world_height
3. Detect if object is flat: world_height << max(rx, ry)
4. Choose scaling strategy:
   - Box-like: Use profile dimensions (current behavior)
   - Flat: Use top-view dimensions with height-based scaling
5. Scale contour using chosen strategy
```

---

## Data Models

### ContourTemplate (Already Exists - No Changes)
**File:** `blender_blocking/geometry/contour_models.py:10-30`

Already includes `original_bbox: Optional[tuple]` with (x, y, w, h) in pixels.

### ContourSlice (Modified - Add Metadata)
**File:** `blender_blocking/geometry/contour_models.py:32-48`

**Before:**
```python
@dataclass(frozen=True)
class ContourSlice:
    """Single loft slice using contour template with scale factors."""

    z: float
    scale_x: float  # Width scale factor (from front view profile)
    scale_y: float  # Depth scale factor (from side view profile)
    cx: float = 0.0  # Center X offset
    cy: float = 0.0  # Center Y offset
```

**After:**
```python
@dataclass(frozen=True)
class ContourSlice:
    """Single loft slice using contour template with scale factors."""

    z: float
    scale_x: float  # Width scale factor (from front view profile)
    scale_y: float  # Depth scale factor (from side view profile)
    cx: float = 0.0  # Center X offset
    cy: float = 0.0  # Center Y offset
    scale_mode: str = "profile"  # "profile" or "contour_native"
    z_scale_factor: float = 1.0  # For height-based uniform scaling
```

**Changes:**
- Add `scale_mode: str = "profile"` to indicate scaling strategy
- Add `z_scale_factor: float = 1.0` for flat object height-based scaling

---

## Detection Algorithm

### New Function: `detect_flat_object()`
**File:** `blender_blocking/geometry/contour_utils.py` (NEW - Insert after imports)

**Location:** After line 8 (after imports, before normalize_contour)

```python
def detect_flat_object(
    max_rx: float,
    max_ry: float,
    world_height: float,
    flatness_threshold: float = 0.2,
) -> bool:
    """
    Detect if object is flat (disk-like) vs. volumetric (box-like).

    Flat objects have small height relative to XY extent, meaning their
    front/side projections don't match their true cross-sections.

    Args:
        max_rx: Maximum X radius from front profile (world units)
        max_ry: Maximum Y radius from side profile (world units)
        world_height: Object height from profile (world units)
        flatness_threshold: Height/XY ratio threshold for flat detection

    Returns:
        True if object is flat (disk-like), False if volumetric (box-like)

    Examples:
        - Cube (2×2×2): height=2, max_xy=1, ratio=2.0 → False (volumetric)
        - Flat star (4×4×0.2): height=0.2, max_xy=2, ratio=0.1 → True (flat)
        - Bottle (1×1×3): height=3, max_xy=0.5, ratio=6.0 → False (volumetric)
    """
    max_xy = max(max_rx, max_ry)

    # Avoid division by zero
    if max_xy == 0:
        return False

    # Compute height to XY ratio
    height_to_xy_ratio = world_height / max_xy

    # Object is flat if height is much smaller than XY extent
    return height_to_xy_ratio < flatness_threshold
```

**Why 0.2 threshold?**
- Cube (1:1 ratio) → 1.0 >> 0.2 → Volumetric ✓
- Flat disk (0.1:1 ratio) → 0.1 < 0.2 → Flat ✓
- Cylindrical bottle (3:1 ratio) → 3.0 >> 0.2 → Volumetric ✓

### New Function: `compute_contour_native_scale()`
**File:** `blender_blocking/geometry/contour_utils.py` (NEW - Insert after detect_flat_object)

**Location:** After detect_flat_object function

```python
def compute_contour_native_scale(
    bbox_w_px: int,
    bbox_h_px: int,
    unit_scale: float,
    z_scale_factor: float = 1.0,
) -> Tuple[float, float]:
    """
    Compute scale factors based on contour's native pixel dimensions.

    Used for flat objects where top-view defines true XY extent.

    Args:
        bbox_w_px: Contour bounding box width in pixels
        bbox_h_px: Contour bounding box height in pixels
        unit_scale: Pixel to world unit conversion (e.g., 0.01)
        z_scale_factor: Height-based scaling factor (0-1 range)

    Returns:
        (scale_x, scale_y) in world units (half-widths for compatibility)

    Example:
        bbox = (0, 0, 400, 400)  # 400×400px star
        unit_scale = 0.01
        z_scale_factor = 0.8  # 80% of max height

        → scale_x = (400 * 0.01 * 0.8) / 2 = 1.6
        → scale_y = (400 * 0.01 * 0.8) / 2 = 1.6

        Contour normalized to [-0.5, 0.5] scaled by 2*1.6 = 3.2 → 3.2×3.2 world units
    """
    # Convert pixel dimensions to world units
    w_world = bbox_w_px * unit_scale
    h_world = bbox_h_px * unit_scale

    # Apply height-based scaling uniformly
    w_scaled = w_world * z_scale_factor
    h_scaled = h_world * z_scale_factor

    # Return as half-widths (radii) for compatibility with existing code
    # (contour scaling multiplies by 2, so we need to divide by 2 here)
    return w_scaled / 2.0, h_scaled / 2.0
```

---

## Scaling Logic Modifications

### Modified Function: `scale_contour_2d()`
**File:** `blender_blocking/geometry/contour_utils.py:141-172`

**Current Implementation (Lines 141-172):**
```python
def scale_contour_2d(
    normalized_contour: np.ndarray,
    scale_x: float,
    scale_y: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
    original_bbox: Tuple[int, int, int, int] = None,
    unit_scale: float = None,
) -> np.ndarray:
    """..."""
    scaled = normalized_contour.copy()

    # Scale from [-0.5, 0.5] to actual dimensions
    # Multiply by 2 because rx/ry are radii and we need diameter
    scaled[:, 0] = scaled[:, 0] * scale_x * 2.0 + center_x
    scaled[:, 1] = scaled[:, 1] * scale_y * 2.0 + center_y

    return scaled
```

**No changes needed** - parameters already accept optional bbox and unit_scale.
The actual scaling logic remains the same because we'll pre-compute the correct scale_x/scale_y values before calling this function.

---

## Mesh Generation Modifications

### Modified Function: `_ring_vertices_from_contour()`
**File:** `blender_blocking/integration/blender_ops/contour_loft_mesh.py:19-60`

**Current Implementation:**
```python
def _ring_vertices_from_contour(
    bm: "bmesh.types.BMesh",
    slice_data: ContourSlice,
    template: ContourTemplate,
    min_radius_u: float,
    weld_degenerate_rings: bool,
) -> Tuple[List["bmesh.types.BMVert"], bool]:
    """Generate ring vertices using contour template scaled by profile factors."""
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
```

**Modified Implementation (Lines 19-70):**

Replace entire function:
```python
def _ring_vertices_from_contour(
    bm: "bmesh.types.BMesh",
    slice_data: ContourSlice,
    template: ContourTemplate,
    min_radius_u: float,
    weld_degenerate_rings: bool,
    unit_scale: float = 0.01,  # NEW PARAMETER
) -> Tuple[List["bmesh.types.BMVert"], bool]:
    """Generate ring vertices using contour template scaled by profile factors.

    Args:
        bm: BMesh instance
        slice_data: ContourSlice with z position and scale factors
        template: Normalized contour template (includes original_bbox)
        min_radius_u: Minimum radius for welding
        weld_degenerate_rings: Whether to collapse degenerate rings to points
        unit_scale: Pixel to world unit conversion (NEW)

    Returns:
        (vertices, is_degenerate)
    """
    # Check for degenerate slice
    if weld_degenerate_rings and (
        slice_data.scale_x <= min_radius_u or slice_data.scale_y <= min_radius_u
    ):
        vert = bm.verts.new((slice_data.cx, slice_data.cy, slice_data.z))
        return [vert], True

    # Compute actual scale factors based on scaling mode
    actual_scale_x = slice_data.scale_x
    actual_scale_y = slice_data.scale_y

    if slice_data.scale_mode == "contour_native" and template.original_bbox is not None:
        # For flat objects, use contour's native dimensions
        _, _, w_px, h_px = template.original_bbox
        from geometry.contour_utils import compute_contour_native_scale
        actual_scale_x, actual_scale_y = compute_contour_native_scale(
            w_px, h_px, unit_scale, slice_data.z_scale_factor
        )

    # Scale template contour by computed factors
    scaled_2d = scale_contour_2d(
        template.points,
        scale_x=actual_scale_x,
        scale_y=actual_scale_y,
        center_x=slice_data.cx,
        center_y=slice_data.cy,
    )

    # Create 3D vertices at height z
    verts = []
    for x, y in scaled_2d:
        verts.append(bm.verts.new((x, y, slice_data.z)))

    return verts, False
```

**Changes:**
- Line 24: Add `unit_scale: float = 0.01` parameter
- Lines 36-43: NEW - Compute actual scale factors based on scale_mode
- Lines 45-50: Use `actual_scale_x/y` instead of `slice_data.scale_x/y`

### Modified Function: `create_contour_loft_mesh()`
**File:** `blender_blocking/integration/blender_ops/contour_loft_mesh.py:105-223`

**Current signature (Line 105-120):**
```python
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
```

**Modified signature:**
```python
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
    unit_scale: float = 0.01,  # NEW PARAMETER
) -> Optional[object]:
```

**Find all calls to `_ring_vertices_from_contour` and add `unit_scale` parameter.**

**Line ~160 (in mesh generation loop):**

**Before:**
```python
verts, is_degen = _ring_vertices_from_contour(
    bm, slice_data, template, min_radius_u, weld_degenerate_rings
)
```

**After:**
```python
verts, is_degen = _ring_vertices_from_contour(
    bm, slice_data, template, min_radius_u, weld_degenerate_rings, unit_scale
)
```

---

## Integration Changes

### Modified Function: `create_3d_blockout_loft()`
**File:** `blender_blocking/main_integration.py:626-793`

**Changes at Lines 727-767:**

**Current code:**
```python
        max_rx = max(slice_data.rx for slice_data in slices)
        max_ry = max(slice_data.ry for slice_data in slices)
        bounds_min = (-max_rx, -max_ry, profile.z0)
        bounds_max = (max_rx, max_ry, profile.z0 + profile.world_height)

        # Choose mesh generation strategy based on top-view availability
        use_contour_lofting = (
            top_contour_template is not None
            and self.config.mesh_from_profile.use_top_contour
        )

        if use_contour_lofting:
            print("  Using top-view constrained lofting (contour-based)")
        else:
            print("  Using elliptical lofting (circular cross-sections)")

        if self.context.dry_run:
            # ... dry run handling ...
            return None

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
                # ... other params ...
            )
```

**Modified code:**
```python
        max_rx = max(slice_data.rx for slice_data in slices)
        max_ry = max(slice_data.ry for slice_data in slices)
        bounds_min = (-max_rx, -max_ry, profile.z0)
        bounds_max = (max_rx, max_ry, profile.z0 + profile.world_height)

        # Choose mesh generation strategy based on top-view availability
        use_contour_lofting = (
            top_contour_template is not None
            and self.config.mesh_from_profile.use_top_contour
        )

        # Detect flat objects for special handling
        is_flat = False
        scale_mode = "profile"
        if use_contour_lofting:
            from geometry.contour_utils import detect_flat_object
            is_flat = detect_flat_object(max_rx, max_ry, profile.world_height)
            if is_flat:
                scale_mode = "contour_native"
                print("  Detected flat object - using top-view native dimensions")
                print("  Using top-view constrained lofting (contour-based, flat-object mode)")
            else:
                print("  Using top-view constrained lofting (contour-based, profile-scaled mode)")
        else:
            print("  Using elliptical lofting (circular cross-sections)")

        if self.context.dry_run:
            # ... dry run handling (no changes) ...
            return None

        if use_contour_lofting:
            # Compute height-based scaling factors for flat objects
            if is_flat and profile.world_height > 0:
                # Normalize z to [0, 1] and use as scaling factor
                # At bottom (z=z0): scale_factor = 0 or min_scale
                # At top (z=z0+height): scale_factor = 1.0
                z_factors = [
                    max(0.1, (s.z - profile.z0) / profile.world_height)
                    for s in slices
                ]
            else:
                # For volumetric objects, no height-based scaling
                z_factors = [1.0 for _ in slices]

            # Convert elliptical slices to contour slices
            contour_slices = [
                ContourSlice(
                    z=s.z,
                    scale_x=s.rx,
                    scale_y=s.ry,
                    cx=s.cx if s.cx is not None else 0.0,
                    cy=s.cy if s.cy is not None else 0.0,
                    scale_mode=scale_mode,  # NEW
                    z_scale_factor=z_factors[i],  # NEW
                )
                for i, s in enumerate(slices)
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
                unit_scale=self.config.reconstruction.unit_scale,  # NEW
            )
```

**Changes:**
1. Lines 737-747: NEW - Detect flat objects and set scale_mode
2. Lines 756-766: NEW - Compute height-based scaling factors for flat objects
3. Lines 770-773: Add `scale_mode` and `z_scale_factor` to ContourSlice
4. Line 783: Pass `unit_scale` parameter to create_contour_loft_mesh

---

## Configuration Changes

### Optional: Add Flatness Threshold Config
**File:** `blender_blocking/config.py:95-140`

**Current LoftMeshOptions (Lines 95-140):**
```python
@dataclass
class LoftMeshOptions:
    # ... existing fields ...
    use_top_contour: bool = True
    contour_simplify_epsilon: float = 0.001
    fallback_to_elliptical: bool = True
```

**Modified (Add new field):**
```python
@dataclass
class LoftMeshOptions:
    # ... existing fields ...
    use_top_contour: bool = True
    contour_simplify_epsilon: float = 0.001
    fallback_to_elliptical: bool = True
    flat_object_threshold: float = 0.2  # NEW - Height/XY ratio threshold
```

**Update validation method (around line 130):**
```python
def validate(self):
    """Validate configuration parameters."""
    if self.radial_segments < 3:
        raise ValueError("radial_segments must be >= 3")
    # ... other validations ...
    if self.contour_simplify_epsilon < 0:
        raise ValueError("contour_simplify_epsilon must be >= 0")
    if self.flat_object_threshold <= 0:  # NEW
        raise ValueError("flat_object_threshold must be > 0")
```

**Update to_dict method:**
```python
def to_dict(self):
    return {
        # ... existing fields ...
        "use_top_contour": self.use_top_contour,
        "contour_simplify_epsilon": self.contour_simplify_epsilon,
        "fallback_to_elliptical": self.fallback_to_elliptical,
        "flat_object_threshold": self.flat_object_threshold,  # NEW
    }
```

**Update config files** (optional - can use default):
```json
"mesh_from_profile": {
  ...
  "use_top_contour": true,
  "contour_simplify_epsilon": 0.001,
  "fallback_to_elliptical": true,
  "flat_object_threshold": 0.2
}
```

---

## Implementation Timeline

### Phase 1: Detection & Utilities (30 min)
1. Add `detect_flat_object()` to `contour_utils.py`
2. Add `compute_contour_native_scale()` to `contour_utils.py`
3. Modify `ContourSlice` dataclass to add `scale_mode` and `z_scale_factor`

### Phase 2: Mesh Generation (30 min)
1. Modify `_ring_vertices_from_contour()` to handle native scaling
2. Modify `create_contour_loft_mesh()` to accept `unit_scale` parameter
3. Update call sites in mesh generation loop

### Phase 3: Integration (30 min)
1. Modify `create_3d_blockout_loft()` to detect flat objects
2. Compute z_scale_factors for flat objects
3. Pass scale_mode and z_scale_factor to ContourSlice
4. Pass unit_scale to create_contour_loft_mesh

### Phase 4: Configuration (15 min)
1. Add `flat_object_threshold` to LoftMeshOptions
2. Update validation and to_dict methods
3. (Optional) Update config JSON files

### Phase 5: Testing (30 min)
1. Run E2E tests on all subjects
2. Verify improvements: star, dog, piss
3. Verify no regression: cube must maintain 1.0 IoU
4. Visual inspection of renders

**Total estimated time: 2.25 hours**

---

## Success Metrics

### Primary Goals:
- ✅ Cube: Maintain IoU = 1.0 (no regression)
- 🎯 Star: 0.472 → 0.90+ IoU
- 🎯 Dog: 0.637 → 0.80+ IoU
- 🎯 Piss: 0.558 → 0.80+ IoU

### Verification Checklist:
- [ ] All 9 test subjects pass E2E validation (avg IoU > 0.7)
- [ ] Top views for flat objects show correct shape and size
- [ ] No visual artifacts (holes, self-intersections)
- [ ] Cube achieves perfect/near-perfect IoU (0.99+)
- [ ] Star top view is diamond-shaped, not squashed line
- [ ] Volumetric objects (bottle, vase) maintain performance

---

## Risk Mitigation

### Risk 1: Height-based scaling may over-compensate
**Mitigation:** Use conservative min scale (0.1) to prevent disappearing geometry

### Risk 2: Detection threshold may misclassify objects
**Mitigation:** Make threshold configurable; test on all 9 subjects; adjust if needed

### Risk 3: Native scaling may not account for camera perspective
**Mitigation:** Use original pixel bbox which captures perspective; adjust if tests fail

### Risk 4: Cube regression
**Mitigation:** Run cube test first; if fails, adjust detection logic to preserve current behavior

---

## Testing Strategy

### Unit Tests (Future Work):
```python
def test_detect_flat_object():
    # Flat disk
    assert detect_flat_object(2.0, 2.0, 0.2, 0.2) == True

    # Cube
    assert detect_flat_object(1.0, 1.0, 2.0, 0.2) == False

    # Tall cylinder
    assert detect_flat_object(0.5, 0.5, 3.0, 0.2) == False

def test_compute_contour_native_scale():
    scale_x, scale_y = compute_contour_native_scale(400, 400, 0.01, 0.8)
    assert scale_x == 1.6  # (400 * 0.01 * 0.8) / 2
    assert scale_y == 1.6
```

### Integration Tests:
1. Run full E2E suite with `loft_profile-ultra.json`
2. Compare before/after results
3. Visual inspection of problematic cases

### Regression Tests:
1. Cube must achieve IoU ≥ 0.99
2. Bottle/vase must maintain ±0.05 IoU
3. Car must maintain or improve IoU

---

## Rollback Plan

If tests fail:
1. Revert changes to `main_integration.py` (preserve detection logic for analysis)
2. Revert changes to `contour_loft_mesh.py`
3. Keep new functions in `contour_utils.py` for future investigation
4. Document failure modes for alternative approach

Alternative if flat detection is problematic:
- Add manual flag: `force_contour_native: bool` in config
- Let users opt-in for specific objects

---

## Files Summary

### Modified Files (5):
1. `blender_blocking/geometry/contour_models.py` - Add fields to ContourSlice
2. `blender_blocking/geometry/contour_utils.py` - Add detection and native scale functions
3. `blender_blocking/integration/blender_ops/contour_loft_mesh.py` - Handle native scaling
4. `blender_blocking/main_integration.py` - Detect flat objects and compute scale factors
5. `blender_blocking/config.py` - Add flatness_threshold config (optional)

### New Code (approx. lines):
- Detection function: ~30 lines
- Native scale function: ~25 lines
- Integration logic: ~30 lines
- Config updates: ~5 lines
- **Total: ~90 new lines, ~50 modified lines**

### Test Files:
- Re-run: `test_e2e_validation.py` with all 9 subjects

---

## Post-Implementation

### Documentation:
- Update `TOP_VIEW_LOFTING_DIAGNOSIS.md` with results
- Create `FLAT_OBJECT_HANDLING.md` explaining the solution
- Add comments explaining flatness detection in code

### Future Improvements:
- Consider alternative detection: aspect ratio of top vs side bbox
- Explore gradient-based scaling (smooth transition top→bottom)
- Add visualization mode to debug scaling factors
- Consider machine learning approach for optimal scale prediction
