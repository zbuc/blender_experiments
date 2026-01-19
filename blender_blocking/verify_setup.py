#!/usr/bin/env python3
"""
Setup Verification Script for Blender Blocking Tool

Run this script in Blender's Python console to verify your setup is correct.

Usage in Blender:
    import sys
    sys.path.insert(0, "/path/to/crew/sculptor")
    exec(open("/path/to/crew/sculptor/blender_blocking/verify_setup.py").read())
"""

import sys


def verify_setup():
    """Verify that all dependencies are properly installed and compatible."""
    print("\n" + "="*70)
    print("🔍 Blender Blocking Tool - Setup Verification")
    print("="*70 + "\n")

    # Check Python version
    print(f"✓ Python version: {sys.version}")
    print(f"✓ Python executable: {sys.executable}\n")

    errors = []
    warnings = []

    # Check numpy
    print("Checking numpy...")
    try:
        import numpy as np
        print(f"  ✓ numpy {np.__version__} installed")
        print(f"    Location: {np.__file__}")
    except ImportError as e:
        errors.append(f"numpy: {e}")
        print(f"  ❌ numpy not found")

    # Check opencv-python
    print("\nChecking opencv-python...")
    try:
        import cv2
        print(f"  ✓ opencv-python {cv2.__version__} installed")
        print(f"    Location: {cv2.__file__}")
    except ImportError as e:
        errors.append(f"opencv-python: {e}")
        print(f"  ❌ opencv-python not found")

    # Check Pillow (with C extension verification)
    print("\nChecking Pillow...")
    try:
        from PIL import Image
        print(f"  ✓ Pillow {Image.__version__} installed")
        print(f"    Location: {Image.__file__}")

        # Critical: check C extensions
        try:
            from PIL import _imaging
            print(f"  ✓ Pillow C extensions (_imaging) working")
        except ImportError as e:
            errors.append(f"Pillow C extensions: {e}")
            print(f"  ❌ Pillow C extensions not compatible!")
            print(f"     This means Pillow was installed for a different Python version.")
            warnings.append("Pillow C extension incompatibility detected")

    except ImportError as e:
        errors.append(f"Pillow: {e}")
        print(f"  ❌ Pillow not found")

    # Check scipy
    print("\nChecking scipy...")
    try:
        import scipy
        print(f"  ✓ scipy {scipy.__version__} installed")
        print(f"    Location: {scipy.__file__}")
    except ImportError as e:
        errors.append(f"scipy: {e}")
        print(f"  ❌ scipy not found")

    # Try importing Blender (if available)
    print("\nChecking Blender availability...")
    try:
        import bpy
        print(f"  ✓ Running in Blender {bpy.app.version_string}")
    except ImportError:
        warnings.append("Not running in Blender")
        print(f"  ⚠️  Not running in Blender (this is OK if testing outside Blender)")

    # Summary
    print("\n" + "="*70)
    if not errors and not warnings:
        print("✅ SETUP COMPLETE - All dependencies verified!")
        print("\nYou're ready to use the Blender Blocking Tool.")
        print("\nQuick start:")
        print("  from blender_blocking.main_integration import example_workflow_no_images")
        print("  example_workflow_no_images()")
    elif errors:
        print("❌ SETUP INCOMPLETE - Missing or incompatible dependencies")
        print("\nErrors found:")
        for error in errors:
            print(f"  • {error}")

        print("\n🔧 FIX:")
        print("Install dependencies into Blender's Python:")
        print("\n  # Find Blender's Python:")
        print("  # In Blender console: import sys; print(sys.executable)")
        print("\n  # Then run:")
        print("  /path/to/blender/python -m pip install numpy opencv-python Pillow scipy")
        print("\n📖 See BLENDER_SETUP.md for complete instructions")

        if "Pillow C extension" in str(errors):
            print("\n⚠️  PILLOW C EXTENSION ERROR DETECTED")
            print("This means you installed Pillow in a venv with a different Python")
            print("version than Blender uses. You MUST install directly into Blender's")
            print("Python for C extensions to work.")
    else:
        print("⚠️  SETUP MOSTLY COMPLETE - Minor warnings")
        print("\nWarnings:")
        for warning in warnings:
            print(f"  • {warning}")

        print("\nYou should be able to proceed, but check the warnings above.")

    print("="*70 + "\n")

    return len(errors) == 0


if __name__ == "__main__":
    # When run directly (or exec'd in Blender console)
    verify_setup()
