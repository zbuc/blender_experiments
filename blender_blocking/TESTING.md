# Testing the Blender Automated Blocking Tool

This document provides testing instructions for the integration.

## Prerequisites

### Install Dependencies

For standalone Python testing:
```bash
pip install numpy opencv-python Pillow scipy
```

For Blender testing, install dependencies in Blender's Python:
```bash
# macOS example (adjust path for your Blender version)
/Applications/Blender.app/Contents/Resources/3.6/python/bin/python3.10 -m pip install numpy opencv-python Pillow scipy

# Linux example
~/.config/blender/3.6/python/bin/python3.10 -m pip install numpy opencv-python Pillow scipy

# Windows example
C:\Program Files\Blender Foundation\Blender 3.6\3.6\python\bin\python.exe -m pip install numpy opencv-python Pillow scipy
```

## Test Suite Overview

The integration includes three main test components:

1. **create_test_images.py** - Generates sample reference images
2. **test_integration.py** - Validates workflow without Blender
3. **main_integration.py** - Complete workflow (requires Blender)

## 1. Generate Test Images

Create sample orthogonal reference images:

```bash
cd blender_blocking
python3 create_test_images.py
```

This generates images in `test_images/` directory:
- Bottle shape (bottle_front.png, bottle_side.png, bottle_top.png)
- Vase shape (vase_front.png, vase_side.png, vase_top.png)
- Cube shape (cube_front.png, cube_side.png, cube_top.png)

## 2. Standalone Tests (No Blender)

Run the test suite without Blender to validate image processing:

```bash
cd blender_blocking
python3 test_integration.py
```

This tests:
- Image loading
- Edge detection and normalization
- Contour detection
- Shape analysis (area, perimeter, circularity, aspect ratio)

Expected output:
```
==================================================
BLENDER BLOCKING INTEGRATION - TEST SUITE
==================================================

[TEST 1/3] Image Loading and Processing
------------------------------------------------
✓ Loaded 3 views
✓ All images are numpy arrays
✓ Image dimensions are correct

[TEST 2/3] Edge Detection
------------------------------------------------
✓ Edge extraction works
✓ Edges are binary
✓ Edge map has expected shape

[TEST 3/3] Shape Analysis
------------------------------------------------
✓ Contours detected
✓ Shape properties extracted
✓ Circularity calculated correctly

==================================================
ALL TESTS PASSED (3/3)
==================================================
```

## 3. Blender Integration Tests

### Interactive Testing in Blender

1. Open Blender
2. Switch to Scripting workspace
3. Open Blender's Python console
4. Run the following:

```python
import sys
from pathlib import Path

# Add repository to Python path
repo_path = "/path/to/blender_experiments/crew/sculptor"
sys.path.insert(0, repo_path)

# Import test suite
from blender_blocking.test_integration import run_all_tests

# Run tests
run_all_tests(blender_available=True)
```

### Command Line Testing

Run Blender in background mode with test script:

```bash
blender --background --python blender_blocking/test_integration.py
```

### Example Workflow Tests

#### Test with Sample Images

```python
import sys
sys.path.insert(0, "/path/to/blender_experiments/crew/sculptor")

from blender_blocking.main_integration import example_workflow_with_images

# Run with vase test images
result = example_workflow_with_images(
    front_path="/path/to/blender_blocking/test_images/vase_front.png",
    side_path="/path/to/blender_blocking/test_images/vase_side.png",
    top_path="/path/to/blender_blocking/test_images/vase_top.png"
)
```

#### Test Procedural Generation (No Images)

```python
import sys
sys.path.insert(0, "/path/to/blender_experiments/crew/sculptor")

from blender_blocking.main_integration import example_workflow_no_images

# Create procedural blockout
mesh = example_workflow_no_images(num_slices=12)
```

## Expected Results

### Standalone Tests
- All 3 test suites should pass
- No import errors
- Shape properties should be within reasonable ranges

### Blender Tests
- Creates mesh object named "Blockout_Mesh"
- Mesh should have vertices and faces
- Camera and lighting added to scene
- Boolean operations complete successfully

## Troubleshooting

### "No module named 'numpy'"
Install dependencies:
```bash
pip install numpy opencv-python Pillow scipy
```

### "No module named 'cv2'"
Install OpenCV:
```bash
pip install opencv-python
```

### "Blender API not available"
This is expected when running standalone tests. The test suite gracefully skips Blender-specific tests when run outside Blender.

### "No shapes detected in images"
- Verify test images were generated correctly
- Check that images have good contrast (black shapes on white background)
- Try adjusting edge detection thresholds in `integration/image_processing/image_processor.py`

### "Boolean union failed"
- Too many primitives can cause issues - reduce `num_slices`
- Try using `join_simple()` instead of `join_with_boolean_union()`
- Check for degenerate primitives (zero or negative scale)

### Import errors in Blender
Blender uses its own Python interpreter. You must install dependencies in Blender's Python:
```bash
# Find Blender's Python
blender --background --python-expr "import sys; print(sys.executable)"

# Use that path to install packages
/path/to/blender/python -m pip install numpy opencv-python
```

## Manual Verification

After running the workflow in Blender, verify:

1. **Mesh Created**
   - Object named "Blockout_Mesh" exists in scene
   - Has geometry (vertices and faces)
   - Positioned at origin

2. **Scene Setup**
   - Camera present and positioned
   - Lights added to scene
   - Viewport is in solid shading mode

3. **Mesh Quality**
   - Mesh approximates the reference shape
   - No gaps or overlaps (boolean union worked)
   - Suitable as base for sculpting

## Performance Notes

- **Image Processing**: Fast (<1 second for 512x512 images)
- **Shape Analysis**: Fast (<1 second)
- **3D Generation**: Depends on num_slices
  - 10 slices: ~2-3 seconds
  - 20 slices: ~5-8 seconds
  - Boolean operations are the bottleneck

## Continuous Testing

When making changes to the codebase:

1. Run standalone tests first: `python3 test_integration.py`
2. If standalone tests pass, test in Blender
3. Verify mesh output visually
4. Check console for any warnings or errors

## Test Coverage

Current tests cover:
- ✅ Image loading (all formats)
- ✅ Image processing (normalization, edge detection)
- ✅ Contour detection
- ✅ Shape analysis (area, perimeter, circularity, aspect ratio)
- ✅ 3D primitive creation
- ✅ Mesh joining
- ✅ Scene setup

Not yet covered:
- ⚠️ Advanced shape matching
- ⚠️ Slice-based comparison
- ⚠️ Adaptive primitive selection
- ⚠️ Error handling for malformed images
- ⚠️ Performance benchmarks

## Future Test Enhancements

1. Add unit tests for individual modules
2. Create benchmarks for performance regression testing
3. Add visual regression tests (compare render output)
4. Test with variety of real-world reference images
5. Add stress tests (large images, many slices)
6. Test error handling and edge cases
