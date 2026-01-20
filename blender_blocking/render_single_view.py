"""
Render a single view of the ellipsoid vs cylinder comparison.
"""

import sys
from pathlib import Path
import math

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import bpy
from test_ellipsoid_vs_cylinder import test_with_procedural_shape

# Run the test
print("\nGenerating comparison meshes...")
cylinder_mesh, ellipsoid_mesh = test_with_procedural_shape()

# Setup rendering
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.render.resolution_x = 1280
scene.render.resolution_y = 720

# Get camera
camera = bpy.data.objects.get('Camera')
if camera:
    camera.location = (15, -15, 8)
    camera.rotation_euler = (math.radians(70), 0, math.radians(35))

# Render
output_path = Path(__file__).parent / "comparison_angle_view.png"
scene.render.filepath = str(output_path)

print(f"\nRendering to: {output_path}")
bpy.ops.render.render(write_still=True)
print(f"✓ Rendered successfully")
