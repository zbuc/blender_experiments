# Blender Automated Blocking Tool

Automated tool for creating rough 3D blockouts from orthogonal reference images for sculpting workflows.

**Convoy:** hq-cv-7sdbk
**Status:** Modules created by polecats, integration pending

## Project Structure

```
blender_blocking/
├── primitives/          # Blender primitive spawning (chrome)
│   ├── primitives.py
│   └── example_primitives.py
├── shape_matching/      # Slice-based shape analysis (nitro)
│   └── slice_shape_matcher.py
├── placement/           # 3D placement & mesh joining (guzzle)
│   └── primitive_placement.py
└── integration/         # Full integration framework (witness)
    ├── image_processing/
    ├── shape_matching/
    └── blender_ops/
```

## Modules Created

### 1. Primitives Library (chrome - COMPLETE)
- **File:** `primitives/primitives.py`
- **Status:** ✅ Complete and committed
- **Functions:** spawn_cube, spawn_sphere, spawn_cylinder, spawn_cone, spawn_torus
- **Features:** Configurable size, position, rotation

### 2. Shape Matching (nitro - COMPLETE)
- **File:** `shape_matching/slice_shape_matcher.py`
- **Status:** ✅ Code complete, not committed
- **Algorithm:** Slice-based mesh comparison using cross-sectional profiles

### 3. Primitive Placement (guzzle - COMPLETE)
- **File:** `placement/primitive_placement.py`
- **Status:** ✅ Code complete, not committed
- **Features:** 3D positioning, scaling, boolean union operations

### 4. Integration Framework (witness - IN PROGRESS)
- **Directory:** `integration/`
- **Status:** ⚠️  Structure created, implementation incomplete
- **Modules:**
  - image_processing/ - Image loading and preprocessing
  - shape_matching/ - Contour analysis
  - blender_ops/ - Scene setup, mesh generation, rendering

### 5. Image Preprocessing (rust - BLOCKED)
- **Status:** ❌ Implemented in Rust instead of Python
- **Issue:** Not compatible with Python/Blender workflow
- **Action Required:** Reimplement in Python using opencv-python

## Next Steps

1. **Fix Image Preprocessing** - Reimplement be-w7q in Python
2. **Complete Integration** - Finish witness's integration framework
3. **Create End-to-End Script** - Main script that ties everything together
4. **Test with Sample Images** - Validate workflow with orthogonal reference photos
5. **Commit & Push** - Merge all work into main branch

## Technical Approach

**Slice-Based Reconstruction (MVP):**
1. Load three orthogonal reference images (front, side, top)
2. Extract silhouettes using edge detection/thresholding
3. Divide silhouettes into horizontal segments
4. Measure width/depth from front and side views for each slice
5. Select best-fit primitives based on aspect ratios and circularity
6. Position and scale primitives in 3D space
7. Boolean union all primitives into single mesh

## Dependencies

- Python 3.8+
- Blender 3.0+ with bpy
- OpenCV (opencv-python)
- NumPy
- SciPy (for shape analysis)
