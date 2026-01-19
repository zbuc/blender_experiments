# Blender Python Setup Guide

Complete guide to configuring Blender's Python environment to use the blocking tool dependencies.

## The Challenge

Blender bundles its own Python interpreter, which doesn't have access to your virtual environment by default. We need to make the dependencies (numpy, opencv-python, Pillow, scipy) available to Blender's Python.

## Solution Options

Choose the approach that works best for your workflow:

### Option 1: Install Dependencies into Blender's Python (Recommended)

Install packages directly into Blender's bundled Python interpreter.

**macOS:**
```bash
# Find Blender's Python (adjust version numbers as needed)
BLENDER_PYTHON="/Applications/Blender.app/Contents/Resources/4.0/python/bin/python3.11"

# Install dependencies
$BLENDER_PYTHON -m pip install numpy opencv-python Pillow scipy
```

**Linux:**
```bash
# Common Blender Python location
BLENDER_PYTHON="/usr/share/blender/4.0/python/bin/python3.11"

# Or if installed via snap
BLENDER_PYTHON="/snap/blender/current/4.0/python/bin/python3.11"

# Install dependencies
$BLENDER_PYTHON -m pip install numpy opencv-python Pillow scipy
```

**Windows:**
```powershell
# Common Blender Python location
$BLENDER_PYTHON = "C:\Program Files\Blender Foundation\Blender 4.0\4.0\python\bin\python.exe"

# Install dependencies
& $BLENDER_PYTHON -m pip install numpy opencv-python Pillow scipy
```

**Finding Blender's Python:**
If the paths above don't work, open Blender's Python console and run:
```python
import sys
print(sys.executable)
```

### Option 2: Add Virtual Environment to Blender's sys.path

Make Blender aware of your virtual environment's packages.

**In Blender's Python console or script:**
```python
import sys

# Add venv site-packages to Blender's path
venv_path = "/path/to/blender_blocking/venv/lib/python3.13/site-packages"
sys.path.insert(0, venv_path)

# Now you can import the modules
import numpy
import cv2
from blender_blocking.main_integration import BlockingWorkflow
```

**To find your venv site-packages path:**
```bash
cd blender_blocking
source venv/bin/activate  # On Windows: venv\Scripts\activate
python -c "import site; print(site.getsitepackages()[0])"
deactivate
```

### Option 3: Launch Blender with PYTHONPATH

Set the PYTHONPATH environment variable when launching Blender.

**macOS/Linux:**
```bash
cd blender_blocking
VENV_PACKAGES="$(pwd)/venv/lib/python3.13/site-packages"

# Launch Blender with custom PYTHONPATH
PYTHONPATH="$VENV_PACKAGES:$PYTHONPATH" /Applications/Blender.app/Contents/MacOS/Blender

# Or create a launch script
cat > launch_blender.sh << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PACKAGES="$SCRIPT_DIR/venv/lib/python3.13/site-packages"
export PYTHONPATH="$VENV_PACKAGES:$PYTHONPATH"
/Applications/Blender.app/Contents/MacOS/Blender "$@"
EOF

chmod +x launch_blender.sh
./launch_blender.sh
```

**Windows (PowerShell):**
```powershell
cd blender_blocking
$VENV_PACKAGES = "$(pwd)\venv\Lib\site-packages"
$env:PYTHONPATH = "$VENV_PACKAGES;$env:PYTHONPATH"
& "C:\Program Files\Blender Foundation\Blender 4.0\blender.exe"
```

### Option 4: Startup Script (Persistent)

Add a startup script to Blender that automatically configures paths.

1. Create a startup script:
```bash
# macOS/Linux
mkdir -p ~/.config/blender/4.0/scripts/startup
cat > ~/.config/blender/4.0/scripts/startup/setup_blocking_tool.py << 'EOF'
import sys
import os

# Add your venv site-packages
VENV_PATH = "/absolute/path/to/blender_blocking/venv/lib/python3.13/site-packages"
if os.path.exists(VENV_PATH) and VENV_PATH not in sys.path:
    sys.path.insert(0, VENV_PATH)
    print(f"[Blocking Tool] Added venv to path: {VENV_PATH}")

# Add blocking tool parent directory
BLOCKING_TOOL_PATH = "/absolute/path/to/crew/sculptor"
if os.path.exists(BLOCKING_TOOL_PATH) and BLOCKING_TOOL_PATH not in sys.path:
    sys.path.insert(0, BLOCKING_TOOL_PATH)
    print(f"[Blocking Tool] Added blocking tool to path: {BLOCKING_TOOL_PATH}")
EOF
```

2. Restart Blender - the script runs automatically on startup

## Verification

Test your setup in Blender's Python console:

```python
# Test imports
import numpy as np
import cv2
from PIL import Image
import scipy

print("✓ All dependencies available!")

# Test blocking tool import
import sys
sys.path.insert(0, "/path/to/crew/sculptor")  # If not using startup script
from blender_blocking.main_integration import BlockingWorkflow

print("✓ Blocking tool ready!")
```

## Quick Start After Setup

Once dependencies are available, using the tool is simple:

```python
import sys
sys.path.insert(0, "/path/to/crew/sculptor")  # If needed

from blender_blocking.main_integration import example_workflow_with_images

# Run with test images
workflow = example_workflow_with_images(
    front_path="/path/to/test_images/vase_front.png",
    side_path="/path/to/test_images/vase_side.png",
    top_path="/path/to/test_images/vase_top.png"
)
```

## Troubleshooting

### "No module named 'cv2'" or "No module named 'numpy'"

Your Blender Python doesn't have access to the dependencies.

**Quick fix:**
```python
import sys
# Add your venv site-packages (update path!)
sys.path.insert(0, "/path/to/blender_blocking/venv/lib/python3.13/site-packages")
```

### "Permission denied" when installing to Blender's Python

On macOS/Linux, you may need admin permissions:
```bash
sudo /Applications/Blender.app/Contents/Resources/4.0/python/bin/python3.11 -m pip install numpy opencv-python Pillow scipy
```

Or install for user only:
```bash
/Applications/Blender.app/Contents/Resources/4.0/python/bin/python3.11 -m pip install --user numpy opencv-python Pillow scipy
```

### "ImportError: numpy.core.multiarray failed to import"

Version mismatch between numpy and Blender's Python. Install a compatible version:
```bash
$BLENDER_PYTHON -m pip install "numpy<2.0" opencv-python Pillow scipy
```

### Different Python versions

If your venv uses Python 3.13 but Blender uses 3.11, the packages may not be compatible. Use Option 1 (install directly into Blender's Python) instead.

### opencv-python build errors on macOS ARM (M1/M2/M3)

Use prebuilt wheels:
```bash
$BLENDER_PYTHON -m pip install --only-binary :all: opencv-python
```

## Recommended Workflow

**For development and testing:**
- Use Option 1 (install into Blender's Python)
- Most reliable, no path juggling needed

**For production/deployment:**
- Use Option 4 (startup script)
- Automated, consistent across sessions
- Easy to update paths centrally

**For quick testing:**
- Use Option 2 (sys.path.insert)
- No installation needed
- Good for trying things out

## Creating a Helper Script

Create a convenience script that handles everything:

```bash
cat > setup_blender_blocking.sh << 'EOF'
#!/bin/bash

# Configuration
BLENDER_APP="/Applications/Blender.app"
BLENDER_PYTHON="$BLENDER_APP/Contents/Resources/4.0/python/bin/python3.11"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔧 Setting up Blender Blocking Tool"
echo "=================================="

# Check if Blender exists
if [ ! -f "$BLENDER_PYTHON" ]; then
    echo "❌ Blender Python not found at: $BLENDER_PYTHON"
    echo "Please update BLENDER_PYTHON in this script"
    exit 1
fi

echo "📦 Installing dependencies to Blender's Python..."
$BLENDER_PYTHON -m pip install numpy opencv-python Pillow scipy

echo "✅ Setup complete!"
echo ""
echo "To use in Blender:"
echo "  import sys"
echo "  sys.path.insert(0, \"$SCRIPT_DIR/..\")"
echo "  from blender_blocking.main_integration import BlockingWorkflow"
EOF

chmod +x setup_blender_blocking.sh
./setup_blender_blocking.sh
```

## Platform-Specific Notes

### macOS
- Blender app is typically at `/Applications/Blender.app`
- Python is in `Blender.app/Contents/Resources/[version]/python/bin/`
- May need to allow Blender in Security & Privacy settings

### Linux
- Installed via package manager: `/usr/share/blender/`
- Installed via snap: `/snap/blender/current/`
- Installed manually: Check `~/blender/` or `/opt/blender/`

### Windows
- Default: `C:\Program Files\Blender Foundation\Blender [version]\`
- Python: `[blender]\[version]\python\bin\python.exe`
- Use PowerShell (not cmd) for better path handling

## See Also

- [QUICKSTART.md](QUICKSTART.md) - Quick start guide for using the tool
- [INTEGRATION.md](INTEGRATION.md) - Full API documentation
- [TESTING.md](TESTING.md) - Testing guide
