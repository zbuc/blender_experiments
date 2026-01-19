# Blender Blocking Tool - Quick Start

Get started with the Blender automated blocking tool in minutes.

## Prerequisites

- Blender 3.0 or later
- Python 3.8+

## Important: Blender Python Setup

**Before you start**, you need to configure Blender's Python to access the tool's dependencies.

👉 **See [BLENDER_SETUP.md](BLENDER_SETUP.md) for complete setup instructions**

**Quick option (Recommended):** Install dependencies directly into Blender's Python:

```bash
# macOS (adjust version as needed)
/Applications/Blender.app/Contents/Resources/4.0/python/bin/python3.11 -m pip install numpy opencv-python Pillow scipy

# Find your Blender Python path: Open Blender → Python Console → Run:
# import sys; print(sys.executable)
```

See BLENDER_SETUP.md for other methods (venv integration, PYTHONPATH, startup scripts).

## Step 1: Install Python Dependencies (For Testing Outside Blender)

From the `blender_blocking` directory:

```bash
cd blender_blocking
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 2: Run Tests (Optional)

Verify everything works:

```bash
# Create test images
./venv/bin/python3 create_test_images.py

# Run tests (without Blender)
./venv/bin/python3 test_integration.py
```

You should see:
```
✓ Image loading successful
✓ Image processing successful
✓ Shape analysis successful
```

## Step 3: Use in Blender

### Quick Test (Procedural - No Images)

1. Open Blender
2. Switch to **Scripting** workspace (top menu bar)
3. In the Python console at bottom:

```python
import sys
sys.path.insert(0, "/path/to/blender_experiments/crew/sculptor")

from blender_blocking.main_integration import example_workflow_no_images
mesh = example_workflow_no_images()
```

You should see a mesh named "Procedural_Blockout" in your scene.

### With Reference Images

1. Create test images (if you haven't):
```bash
cd blender_blocking
./venv/bin/python3 create_test_images.py
```

2. Open Blender and run:

```python
import sys
sys.path.insert(0, "/path/to/blender_experiments/crew/sculptor")

from blender_blocking.main_integration import example_workflow_with_images

# Replace paths with your actual paths
workflow = example_workflow_with_images(
    front_path="/path/to/test_images/vase_front.png",
    side_path="/path/to/test_images/vase_side.png",
    top_path="/path/to/test_images/vase_top.png"
)
```

### With Your Own Images

Your images should:
- Be orthogonal views (front, side, top)
- Have black shapes on white background
- Be PNG or JPG format

```python
import sys
sys.path.insert(0, "/path/to/blender_experiments/crew/sculptor")

from blender_blocking.main_integration import BlockingWorkflow

workflow = BlockingWorkflow(
    front_path="/path/to/your/front.png",
    side_path="/path/to/your/side.png",
    top_path="/path/to/your/top.png"
)

# Run the workflow
mesh = workflow.run_full_workflow(num_slices=12)
```

## What You Get

After running the workflow:
- **Mesh**: "Blockout_Mesh" object in your scene
- **Camera**: Positioned for 3D view
- **Lighting**: Basic lighting setup

The mesh is ready for sculpting!

## Adjusting Parameters

Control the detail level with `num_slices`:

```python
# More slices = more detail (but slower)
mesh = workflow.run_full_workflow(num_slices=20)

# Fewer slices = faster (but less detail)
mesh = workflow.run_full_workflow(num_slices=8)
```

## Troubleshooting

### "No module named numpy" or dependency errors

See [BLENDER_SETUP.md](BLENDER_SETUP.md) for complete Blender Python configuration guide with multiple setup options.

**Quick fix:**
```python
# In Blender, add your venv to path before importing
import sys
sys.path.insert(0, "/path/to/blender_blocking/venv/lib/python3.X/site-packages")
```

### "No shapes detected"
- Ensure images have good contrast (black on white)
- Try inverting your images if they're white on black

### Boolean union fails
- Reduce `num_slices` to 8 or 10
- Check that your images have clear silhouettes

### Import errors

Make sure you use the correct path in `sys.path.insert(0, "...")` - it should point to the directory containing `blender_blocking/`

For persistent setup across Blender sessions, see the startup script option in [BLENDER_SETUP.md](BLENDER_SETUP.md)

## Next Steps

- See **[BLENDER_SETUP.md](BLENDER_SETUP.md)** for Blender Python configuration options
- See **[INTEGRATION.md](INTEGRATION.md)** for detailed API documentation
- See **[TESTING.md](TESTING.md)** for comprehensive testing guide
- Experiment with different reference images
- Adjust `num_slices` for your needs

## Example Session

Complete example from scratch:

```bash
# 1. Setup
cd /path/to/blender_experiments/crew/sculptor/blender_blocking
python3 -m venv venv
./venv/bin/python3 -m pip install -r requirements.txt

# 2. Create test images
./venv/bin/python3 create_test_images.py

# 3. Open Blender, then in Python console:
```

```python
import sys
sys.path.insert(0, "/path/to/blender_experiments/crew/sculptor")

from blender_blocking.main_integration import example_workflow_with_images

workflow = example_workflow_with_images(
    front_path="/path/to/blender_blocking/test_images/vase_front.png",
    side_path="/path/to/blender_blocking/test_images/vase_side.png",
    top_path="/path/to/blender_blocking/test_images/vase_top.png"
)
```

Done! You now have a blockout mesh ready for sculpting.
