# Blender Automated Blocking Tool

Automated tool for creating rough 3D blockouts from orthogonal reference images for sculpting workflows.

**Status:** ✅ Complete and tested

## Quick Start

**New users start here:**

1. **[BLENDER_SETUP.md](BLENDER_SETUP.md)** - Configure Blender's Python environment (one-time setup)
2. **[QUICKSTART.md](QUICKSTART.md)** - 5-minute guide to your first blockout

Already configured Blender? Jump straight to the [QUICKSTART](QUICKSTART.md).

## Documentation

- **[BLENDER_SETUP.md](BLENDER_SETUP.md)** - Blender Python configuration guide ⭐ Start here
- **[QUICKSTART.md](QUICKSTART.md)** - Get started in minutes
- **[INTEGRATION.md](INTEGRATION.md)** - Detailed API and usage guide
- **[TESTING.md](TESTING.md)** - Testing instructions and troubleshooting
- **[E2E_VALIDATION_SUMMARY.md](E2E_VALIDATION_SUMMARY.md)** - Validation framework details

## What It Does

Takes orthogonal reference images (front, side, top views) and automatically generates a 3D blockout mesh ready for sculpting in Blender.

**Input:** Reference images (PNG/JPG with black silhouettes on white background)

**Output:** Joined mesh object in Blender, ready for sculpting

## Quick Example

```python
import sys
sys.path.insert(0, "/path/to/blender_experiments/crew/sculptor")

from blender_blocking.main_integration import example_workflow_with_images

# Create blockout from reference images
workflow = example_workflow_with_images(
    front_path="images/front.png",
    side_path="images/side.png",
    top_path="images/top.png"
)
```

## How It Works

1. **Load** orthogonal reference images
2. **Process** images (edge detection, normalization)
3. **Analyze** shapes (contours, circularity, aspect ratio)
4. **Generate** 3D primitives using slice-based reconstruction
5. **Join** primitives into single mesh using boolean operations
6. **Setup** camera and lighting

## Project Structure

```
blender_blocking/
├── README.md                   # This file
├── BLENDER_SETUP.md            # ⭐ Blender Python setup guide
├── QUICKSTART.md               # Quick start guide
├── INTEGRATION.md              # Detailed API guide
├── TESTING.md                  # Testing guide
├── main_integration.py         # Main workflow
├── create_test_images.py       # Test image generator
├── test_integration.py         # Test suite
├── test_e2e_validation.py      # E2E validation with IoU
├── requirements.txt            # Dependencies
├── primitives/                 # Blender primitive spawning
│   └── primitives.py
├── shape_matching/             # Slice-based shape analysis
│   └── slice_shape_matcher.py
├── placement/                  # 3D placement & mesh joining
│   └── primitive_placement.py
└── integration/                # Integration modules
    ├── image_processing/       # Image loading & processing
    ├── shape_matching/         # Contour analysis
    └── blender_ops/            # Scene setup & mesh generation
```

## Requirements

- Python 3.8+
- Blender 3.0+
- Dependencies: numpy, opencv-python, Pillow, scipy

See [QUICKSTART.md](QUICKSTART.md) for installation instructions.

## Development

### Created By

This tool was built through Gas Town's multi-agent workflow:
- **chrome** - Primitives library
- **nitro** - Shape matching algorithm
- **guzzle** - Primitive placement
- **witness** - Integration framework
- **sculptor** - Final integration and testing

**Convoy:** hq-cv-7sdbk

### Module Status

- ✅ Primitives library - Complete
- ✅ Shape matching - Complete
- ✅ Primitive placement - Complete
- ✅ Integration framework - Complete
- ✅ Main integration - Complete
- ✅ Test suite - Complete
- ✅ Documentation - Complete

## License

See repository root for license information.
