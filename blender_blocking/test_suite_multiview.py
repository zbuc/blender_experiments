"""
Automated test suite for multi-view Visual Hull validation.

Generates ~20 diverse test meshes and validates 3-view vs 12-view
reconstruction across different object categories.

Categories:
- Simple shapes (cube, sphere, cylinder, cone, torus)
- Furniture (chair, table, lamp)
- Animals (simplified models)
- Vehicles (simplified models)
- Organic (vase, bottle, bowl)

Usage:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python test_suite_multiview.py
"""

import sys
from pathlib import Path
import time
import json
import pickle
from datetime import datetime

# Add to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path.home() / 'blender_python_packages'))

import bpy
import numpy as np
from mathutils import Vector
import math

from integration.multi_view.visual_hull import MultiViewVisualHull


# Test suite configuration
TEST_OBJECTS = [
    # Simple shapes
    {"name": "cube", "category": "simple", "type": "cube", "params": {"size": 1.5}},
    {"name": "sphere", "category": "simple", "type": "sphere", "params": {"radius": 0.75}},
    {"name": "cylinder", "category": "simple", "type": "cylinder", "params": {"radius": 0.5, "depth": 2.0}},
    {"name": "cone", "category": "simple", "type": "cone", "params": {"radius": 0.75, "depth": 1.5}},
    {"name": "torus", "category": "simple", "type": "torus", "params": {"major_radius": 0.75, "minor_radius": 0.25}},

    # Organic shapes
    {"name": "vase", "category": "organic", "type": "vase", "params": {}},
    {"name": "bottle", "category": "organic", "type": "bottle", "params": {}},
    {"name": "bowl", "category": "organic", "type": "bowl", "params": {}},
    {"name": "cup", "category": "organic", "type": "cup", "params": {}},

    # Furniture (simplified)
    {"name": "table", "category": "furniture", "type": "table", "params": {}},
    {"name": "chair", "category": "furniture", "type": "chair", "params": {}},
    {"name": "lamp", "category": "furniture", "type": "lamp", "params": {}},

    # Vehicles (very simplified)
    {"name": "car_simple", "category": "vehicle", "type": "car", "params": {}},

    # Animals (very simplified)
    {"name": "dog_simple", "category": "animal", "type": "dog", "params": {}},

    # Complex shapes
    {"name": "elongated_cylinder", "category": "complex", "type": "cylinder", "params": {"radius": 0.3, "depth": 3.0}},
    {"name": "wide_cylinder", "category": "complex", "type": "cylinder", "params": {"radius": 1.0, "depth": 0.5}},
    {"name": "small_sphere", "category": "complex", "type": "sphere", "params": {"radius": 0.4}},
    {"name": "large_cube", "category": "complex", "type": "cube", "params": {"size": 2.0}},
    {"name": "thin_cone", "category": "complex", "type": "cone", "params": {"radius": 0.3, "depth": 2.5}},
    {"name": "thick_torus", "category": "complex", "type": "torus", "params": {"major_radius": 0.6, "minor_radius": 0.35}},
]


def clear_scene():
    """Remove all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def create_mesh(obj_type, params):
    """Create a mesh based on type and parameters."""
    if obj_type == "cube":
        bpy.ops.mesh.primitive_cube_add(size=params.get("size", 1.5))
        return bpy.context.active_object

    elif obj_type == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=params.get("radius", 0.75))
        return bpy.context.active_object

    elif obj_type == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(
            radius=params.get("radius", 0.5),
            depth=params.get("depth", 2.0)
        )
        return bpy.context.active_object

    elif obj_type == "cone":
        bpy.ops.mesh.primitive_cone_add(
            radius1=params.get("radius", 0.75),
            depth=params.get("depth", 1.5)
        )
        return bpy.context.active_object

    elif obj_type == "torus":
        bpy.ops.mesh.primitive_torus_add(
            major_radius=params.get("major_radius", 0.75),
            minor_radius=params.get("minor_radius", 0.25)
        )
        return bpy.context.active_object

    elif obj_type == "vase":
        return create_vase()

    elif obj_type == "bottle":
        return create_bottle()

    elif obj_type == "bowl":
        return create_bowl()

    elif obj_type == "cup":
        return create_cup()

    elif obj_type == "table":
        return create_table()

    elif obj_type == "chair":
        return create_chair()

    elif obj_type == "lamp":
        return create_lamp()

    elif obj_type == "car":
        return create_car()

    elif obj_type == "dog":
        return create_dog()

    else:
        raise ValueError(f"Unknown object type: {obj_type}")


def create_vase():
    """Create vase shape."""
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=2.0)
    vase = bpy.context.active_object
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.transform.resize(value=(1.2, 1.2, 1.0), constraint_axis=(True, True, False))
    bpy.ops.object.mode_set(mode='OBJECT')
    return vase


def create_bottle():
    """Create bottle shape."""
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1.5, location=(0, 0, 0.75))
    body = bpy.context.active_object
    bpy.ops.mesh.primitive_cylinder_add(radius=0.2, depth=0.5, location=(0, 0, 1.75))
    neck = bpy.context.active_object
    modifier = body.modifiers.new(name='Union', type='BOOLEAN')
    modifier.operation = 'UNION'
    modifier.object = neck
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.modifier_apply(modifier='Union')
    bpy.data.objects.remove(neck)
    return body


def create_bowl():
    """Create bowl shape."""
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.75)
    bowl = bpy.context.active_object
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.transform.resize(value=(1.0, 1.0, 0.5))
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')
    return bowl


def create_cup():
    """Create cup shape."""
    bpy.ops.mesh.primitive_cylinder_add(radius=0.4, depth=1.0)
    cup = bpy.context.active_object
    return cup


def create_table():
    """Create simplified table."""
    # Tabletop
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.75))
    tabletop = bpy.context.active_object
    tabletop.scale = (1.5, 1.0, 0.1)

    # Legs
    for x in [-0.6, 0.6]:
        for y in [-0.4, 0.4]:
            bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=0.7, location=(x, y, 0.35))
            leg = bpy.context.active_object
            modifier = tabletop.modifiers.new(name='Union', type='BOOLEAN')
            modifier.operation = 'UNION'
            modifier.object = leg

    bpy.context.view_layer.objects.active = tabletop
    for modifier in tabletop.modifiers:
        bpy.ops.object.modifier_apply(modifier=modifier.name)

    return tabletop


def create_chair():
    """Create simplified chair."""
    # Seat
    bpy.ops.mesh.primitive_cube_add(size=0.8, location=(0, 0, 0.5))
    seat = bpy.context.active_object
    seat.scale = (1.0, 1.0, 0.1)

    # Back
    bpy.ops.mesh.primitive_cube_add(size=0.8, location=(0, -0.35, 1.0))
    back = bpy.context.active_object
    back.scale = (1.0, 0.1, 0.8)

    modifier = seat.modifiers.new(name='Union', type='BOOLEAN')
    modifier.operation = 'UNION'
    modifier.object = back
    bpy.context.view_layer.objects.active = seat
    bpy.ops.object.modifier_apply(modifier='Union')
    bpy.data.objects.remove(back)

    return seat


def create_lamp():
    """Create simplified lamp."""
    # Base
    bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=0.2, location=(0, 0, 0.1))
    base = bpy.context.active_object

    # Pole
    bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=1.5, location=(0, 0, 0.85))
    pole = bpy.context.active_object

    # Shade
    bpy.ops.mesh.primitive_cone_add(radius1=0.4, depth=0.5, location=(0, 0, 1.75))
    shade = bpy.context.active_object

    modifier = base.modifiers.new(name='Union1', type='BOOLEAN')
    modifier.operation = 'UNION'
    modifier.object = pole

    modifier = base.modifiers.new(name='Union2', type='BOOLEAN')
    modifier.operation = 'UNION'
    modifier.object = shade

    bpy.context.view_layer.objects.active = base
    for modifier in base.modifiers:
        bpy.ops.object.modifier_apply(modifier=modifier.name)

    bpy.data.objects.remove(pole)
    bpy.data.objects.remove(shade)

    return base


def create_car():
    """Create very simplified car."""
    # Body
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.3))
    body = bpy.context.active_object
    body.scale = (1.5, 0.8, 0.5)

    # Cabin
    bpy.ops.mesh.primitive_cube_add(size=0.7, location=(0, 0, 0.75))
    cabin = bpy.context.active_object
    cabin.scale = (0.8, 0.7, 0.6)

    modifier = body.modifiers.new(name='Union', type='BOOLEAN')
    modifier.operation = 'UNION'
    modifier.object = cabin
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.modifier_apply(modifier='Union')
    bpy.data.objects.remove(cabin)

    return body


def create_dog():
    """Create very simplified dog."""
    # Body
    bpy.ops.mesh.primitive_cube_add(size=0.8, location=(0, 0, 0.5))
    body = bpy.context.active_object
    body.scale = (1.2, 0.6, 0.6)

    # Head
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=(0.7, 0, 0.6))
    head = bpy.context.active_object

    modifier = body.modifiers.new(name='Union', type='BOOLEAN')
    modifier.operation = 'UNION'
    modifier.object = head
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.modifier_apply(modifier='Union')
    bpy.data.objects.remove(head)

    return body


def voxelize_mesh(mesh_obj, resolution=128, bounds_min=None, bounds_max=None):
    """Voxelize mesh for ground truth."""
    if bounds_min is None:
        bounds_min = np.array([-2.0, -2.0, -2.0])
    if bounds_max is None:
        bounds_max = np.array([2.0, 2.0, 2.0])

    voxel_grid = np.zeros((resolution, resolution, resolution), dtype=bool)
    voxel_size = (bounds_max - bounds_min) / resolution

    voxels_inside = 0
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                voxel_pos = bounds_min + (np.array([i, j, k]) + 0.5) * voxel_size
                point = Vector((voxel_pos[0], voxel_pos[1], voxel_pos[2]))

                hit_count = 0
                for ray_dir in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                    direction = Vector(ray_dir)
                    result, location, normal, index = mesh_obj.ray_cast(
                        point - direction * 10,
                        direction,
                        distance=20.0
                    )
                    if result:
                        hit_count += 1

                if hit_count >= 2:
                    voxel_grid[i, j, k] = True
                    voxels_inside += 1

    return voxel_grid


def render_turntable(obj, output_dir, num_views=12, resolution=512):
    """Render turntable sequence for object."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup camera
    bpy.ops.object.camera_add(location=(5, 0, 0))
    camera = bpy.context.active_object
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 4.0
    bpy.context.scene.camera = camera

    # Setup lighting
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 5))

    # Setup render
    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'BW'
    scene.render.engine = 'BLENDER_EEVEE'
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)

    # Set object to black
    if not obj.data.materials:
        mat = bpy.data.materials.new(name="Black")
        mat.use_nodes = True
        mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0, 0, 0, 1)
        obj.data.materials.append(mat)

    # Render lateral views
    angle_step = 360.0 / num_views
    for i in range(num_views):
        angle = i * angle_step
        angle_rad = math.radians(angle)

        distance = 5.0
        camera.location.x = distance * math.cos(angle_rad)
        camera.location.y = distance * math.sin(angle_rad)
        camera.location.z = 0.0

        direction = Vector((0, 0, 0)) - camera.location
        camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

        scene.render.filepath = str(output_dir / f"view_{int(angle):03d}.png")
        bpy.ops.render.render(write_still=True)

    # Render top view
    camera.location = (0, 0, 5.0)
    camera.rotation_euler = (0, 0, 0)
    scene.render.filepath = str(output_dir / "top.png")
    bpy.ops.render.render(write_still=True)


def load_image_as_silhouette(image_path):
    """Load image and convert to silhouette."""
    from integration.image_processing.image_loader import load_image
    img = load_image(str(image_path))
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)
    return img < 128


def reconstruct_multiview(turntable_dir, num_views, resolution=128):
    """Reconstruct using multi-view Visual Hull."""
    from integration.image_processing.image_loader import load_multi_view_auto

    views_dict = load_multi_view_auto(str(turntable_dir), num_views=num_views, include_top=True)

    hull = MultiViewVisualHull(
        resolution=resolution,
        bounds_min=np.array([-2.0, -2.0, -2.0]),
        bounds_max=np.array([2.0, 2.0, 2.0])
    )

    for view_name, (img, angle, view_type) in views_dict.items():
        if len(img.shape) == 3:
            img_gray = np.mean(img, axis=2)
        else:
            img_gray = img
        silhouette = img_gray < 128
        hull.add_view_from_silhouette(silhouette, angle=angle, view_type=view_type)

    start_time = time.time()
    voxel_grid = hull.reconstruct(verbose=False)
    processing_time = time.time() - start_time

    return voxel_grid, processing_time


def compute_iou(grid_a, grid_b):
    """Compute IoU between two voxel grids."""
    intersection = np.logical_and(grid_a, grid_b).sum()
    union = np.logical_or(grid_a, grid_b).sum()
    return intersection / union if union > 0 else 0.0


def test_object(obj_config, resolution=128):
    """Test a single object through the pipeline."""
    print(f"\n{'='*70}")
    print(f"Testing: {obj_config['name']} ({obj_config['category']})")
    print(f"{'='*70}")

    results = {
        "name": obj_config["name"],
        "category": obj_config["category"],
        "type": obj_config["type"],
    }

    # Create object
    clear_scene()
    print("Creating mesh...")
    obj = create_mesh(obj_config["type"], obj_config["params"])

    # Voxelize ground truth
    print("Voxelizing ground truth...")
    start_vox = time.time()
    ground_truth = voxelize_mesh(obj, resolution=resolution)
    vox_time = time.time() - start_vox
    results["voxelization_time"] = vox_time
    results["ground_truth_voxels"] = int(ground_truth.sum())
    print(f"  Ground truth: {ground_truth.sum():,} voxels in {vox_time:.1f}s")

    # Render turntable
    turntable_dir = Path(f"test_output/suite/{obj_config['name']}")
    print(f"Rendering turntable to {turntable_dir}...")
    render_turntable(obj, turntable_dir, num_views=12)

    # 3-view reconstruction
    print("Reconstructing with 3 views...")
    voxel_3view, time_3view = reconstruct_multiview(turntable_dir, num_views=3, resolution=resolution)
    iou_3view = compute_iou(voxel_3view, ground_truth)
    results["3view_iou"] = float(iou_3view)
    results["3view_time"] = float(time_3view)
    results["3view_voxels"] = int(voxel_3view.sum())
    print(f"  3-view: IoU={iou_3view:.4f}, time={time_3view:.2f}s, voxels={voxel_3view.sum():,}")

    # 12-view reconstruction
    print("Reconstructing with 12 views...")
    voxel_12view, time_12view = reconstruct_multiview(turntable_dir, num_views=12, resolution=resolution)
    iou_12view = compute_iou(voxel_12view, ground_truth)
    results["12view_iou"] = float(iou_12view)
    results["12view_time"] = float(time_12view)
    results["12view_voxels"] = int(voxel_12view.sum())
    print(f"  12-view: IoU={iou_12view:.4f}, time={time_12view:.2f}s, voxels={voxel_12view.sum():,}")

    # Improvement
    iou_improvement = iou_12view - iou_3view
    results["iou_improvement"] = float(iou_improvement)
    results["improvement_percent"] = float(iou_improvement / iou_3view * 100) if iou_3view > 0 else 0
    print(f"  Improvement: {iou_improvement:+.4f} ({results['improvement_percent']:+.1f}%)")

    return results


def main():
    """Run test suite on all objects."""
    print("\n" + "="*70)
    print("MULTI-VIEW VISUAL HULL TEST SUITE")
    print("="*70)
    print(f"\nTest objects: {len(TEST_OBJECTS)}")
    print(f"Resolution: 128³")
    print(f"Started: {datetime.now().isoformat()}")

    all_results = []

    for i, obj_config in enumerate(TEST_OBJECTS, 1):
        print(f"\n[{i}/{len(TEST_OBJECTS)}]")
        try:
            results = test_object(obj_config, resolution=128)
            all_results.append(results)
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "name": obj_config["name"],
                "category": obj_config["category"],
                "error": str(e)
            })

    # Save results
    results_path = Path("test_output/suite/test_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUITE SUMMARY")
    print("="*70)

    successful = [r for r in all_results if "error" not in r]
    failed = [r for r in all_results if "error" in r]

    print(f"\nCompleted: {len(successful)}/{len(TEST_OBJECTS)}")
    print(f"Failed: {len(failed)}")

    if successful:
        avg_3view = np.mean([r["3view_iou"] for r in successful])
        avg_12view = np.mean([r["12view_iou"] for r in successful])
        avg_improvement = np.mean([r["iou_improvement"] for r in successful])

        print(f"\nAverage IoU:")
        print(f"  3-view:  {avg_3view:.4f}")
        print(f"  12-view: {avg_12view:.4f}")
        print(f"  Improvement: {avg_improvement:+.4f} ({avg_improvement/avg_3view*100:+.1f}%)")

        print(f"\nBy category:")
        categories = set(r["category"] for r in successful)
        for cat in sorted(categories):
            cat_results = [r for r in successful if r["category"] == cat]
            cat_avg_12view = np.mean([r["12view_iou"] for r in cat_results])
            cat_avg_improvement = np.mean([r["iou_improvement"] for r in cat_results])
            print(f"  {cat}: {cat_avg_12view:.4f} IoU ({cat_avg_improvement:+.4f} improvement)")

    print(f"\nResults saved to: {results_path}")
    print("\n" + "="*70)
    print("TEST SUITE COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
