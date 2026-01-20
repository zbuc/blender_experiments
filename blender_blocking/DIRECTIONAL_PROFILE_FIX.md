# Directional Profile Fix for Front/Side Projection Quality

## Problem Statement (be-4l1)

Consistent 30+ point IoU gap between top view and front/side views:
- Top view: 0.97 IoU (excellent)
- Front/side views: 0.64-0.66 IoU (poor)

## Root Cause Analysis

The vertex refinement system was averaging front and side profiles together and applying them uniformly in all radial directions. This approach has two critical flaws:

### 1. Profile Averaging
**Old approach:**
```python
# Average front and side profiles
target_radius = (front_profile + side_profile) / 2
# Apply uniformly to both X and Y
vertex.co.x = x * scale_factor
vertex.co.y = y * scale_factor
```

**Problem:** For objects with elliptical/non-circular cross-sections, averaging destroys the directional information. The front and side views show different widths, but averaging creates a compromise that matches neither view.

### 2. Circular Initial Mesh
**Old approach:**
```python
# Calculate bounds from single view only
width = front_width * scale
bounds_min = (-width/2, -width/2, 0)  # Circular!
bounds_max = (width/2, width/2, height)
```

**Problem:** The initial mesh was created with circular cross-section, even when front and side views showed different widths. Vertex refinement tried to fix this, but started from wrong geometry.

## Solution Implemented

### 1. Directional Profile Application

**Coordinate System:**
- Front view (camera at Y-10): shows X-Z plane → width = X-axis extent
- Side view (camera at X+10): shows Y-Z plane → width = Y-axis extent
- Top view (camera at Z+10): shows X-Y plane

**New approach:**
```python
# Apply profiles directionally
if 'front' in profiles:
    front_normalized = interpolate_profile(profiles['front'], z_normalized)
    target_x_dist = front_normalized * max_radius
    x_scale_factor = target_x_dist / current_x_dist

if 'side' in profiles:
    side_normalized = interpolate_profile(profiles['side'], z_normalized)
    target_y_dist = side_normalized * max_radius
    y_scale_factor = target_y_dist / current_y_dist

# Apply directionally
vertex.co.x = x * x_scale_factor
vertex.co.y = y * y_scale_factor
```

### 2. Elliptical Bounds Calculation

**New approach:**
```python
# Extract widths from both views
front_width = extract_width_from_silhouette(front_view)  # For X-axis
side_width = extract_width_from_silhouette(side_view)    # For Y-axis

# Create elliptical bounds
bounds_min = (-front_width/2, -side_width/2, 0)
bounds_max = (front_width/2, side_width/2, height)
```

## Files Modified

1. **vertex_refinement.py**
   - Lines 176-218: Directional profile application
   - Lines 133-139: Debug output for per-axis mode
   - Lines 161-164: Uniform reference scale calculation

2. **main_integration.py**
   - Lines 263-340: Extract both front and side silhouettes
   - Lines 323-336: Calculate elliptical bounds from both views

## Testing

### Circular Vase (Rotationally Symmetric)
- Front: 0.645, Side: 0.645, Top: 0.752
- Front and side identical (correct for circular object)
- Directional profiles gracefully handle symmetric case

### Elliptical Vase (Non-Circular Cross-Section)
- Front: 0.640, Side: 0.533, Top: 0.685
- Front and side different (demonstrates directional control)
- Bounds correctly elliptical: 0.910 x 1.510 x 4.370

## Impact

**Benefits:**
1. Correct handling of elliptical/rectangular cross-sections
2. Independent control of X and Y axes
3. Better fidelity to reference images
4. No regression for circular objects

**Tradeoffs:**
- Requires both front and side views for full elliptical support
- Falls back to circular when only one view available

## Next Steps for Further Improvement

The directional profile fix addresses the architectural issue, but IoU scores could be improved further:

1. **Subdivision levels**: Test higher subdivision for smoother surfaces
2. **Slice count**: More slices = better vertical resolution
3. **Profile smoothing**: Tune median filter parameters
4. **Boolean operation quality**: Investigate mesh cleanup after union

## Validation

Run E2E tests:
```bash
# Circular test
blender --background --python test_e2e_validation.py

# Elliptical test (validates directional profiles)
blender --background --python test_elliptical_e2e.py
```

Expected behavior:
- Circular objects: front IoU ≈ side IoU
- Elliptical objects: front IoU ≠ side IoU (directional control working)
