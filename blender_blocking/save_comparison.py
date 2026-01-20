"""
Run the ellipsoid vs cylinder test and save the result to a .blend file.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import bpy
from test_ellipsoid_vs_cylinder import test_with_procedural_shape

# Run the test
cylinder_mesh, ellipsoid_mesh = test_with_procedural_shape()

# Save the comparison
output_path = Path(__file__).parent / "comparison_ellipsoid_vs_cylinder.blend"
bpy.ops.wm.save_as_mainfile(filepath=str(output_path))

print(f"\n✓ Saved comparison to: {output_path}")
print("\nTo view:")
print(f"  blender {output_path}")
