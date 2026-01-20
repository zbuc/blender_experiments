"""
Render the ellipsoid vs cylinder comparison from multiple angles.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import bpy
import math
from test_ellipsoid_vs_cylinder import test_with_procedural_shape

# Run the test
print("\nGenerating comparison meshes...")
cylinder_mesh, ellipsoid_mesh = test_with_procedural_shape()

# Setup rendering
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE_NEXT'  # Use EEVEE for fast rendering
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.film_transparent = False

# Get camera
camera = bpy.data.objects.get('Camera')
if not camera:
    print("ERROR: No camera found in scene")
    sys.exit(1)

# Define camera positions for different views
views = [
    ("front", (0, -20, 5), (math.radians(80), 0, 0)),
    ("side", (20, 0, 5), (math.radians(80), 0, math.radians(90))),
    ("angle", (15, -15, 8), (math.radians(70), 0, math.radians(35))),
]

output_dir = Path(__file__).parent / "comparison_renders"
output_dir.mkdir(exist_ok=True)

print(f"\nRendering {len(views)} views...")
print(f"Output directory: {output_dir}")

for view_name, location, rotation in views:
    print(f"\n  Rendering {view_name} view...")

    # Position camera
    camera.location = location
    camera.rotation_euler = rotation

    # Render
    output_path = output_dir / f"comparison_{view_name}.png"
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)

    print(f"    ✓ Saved: {output_path}")

print("\n" + "="*70)
print("RENDERING COMPLETE")
print("="*70)
print(f"\nRendered {len(views)} views to: {output_dir}")
print("\nFiles:")
for view_name, _, _ in views:
    print(f"  - comparison_{view_name}.png")
print("\n" + "="*70)
