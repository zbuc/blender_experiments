"""
Phase 2 Integration Test: Visual Hull → Multi-Profile → Primitives

Tests the complete Phase 2 pipeline:
1. Generate Visual Hull mesh from multi-view images
2. Extract multi-angle profiles from mesh
3. Combine profiles (median)
4. Feed into SliceAnalyzer + primitive placement
5. Measure IoU improvement over Visual Hull alone

Expected results:
- Visual Hull alone: ~0.78 IoU
- Visual Hull + primitives: ~0.85-0.88 IoU (target)

Usage:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python test_phase2_integration.py
"""

import sys
from pathlib import Path
import time

# Add to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path.home() / 'blender_python_packages'))

import bpy
import numpy as np
from mathutils import Vector

# Phase 1: Visual Hull
from integration.multi_view.visual_hull import MultiViewVisualHull

# Phase 2: Multi-profile extraction
from integration.shape_matching.mesh_profile_extractor import (
    extract_multi_angle_profiles,
    combine_profiles
)

# Existing primitive placement
from placement.primitive_placement import SliceAnalyzer, PrimitivePlacer, MeshJoiner


def clear_scene():
    """Remove all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def create_test_mesh_vase():
    """Create vase for testing (same as in test suite)."""
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=2.0, location=(0, 0, 0))
    vase = bpy.context.active_object
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.transform.resize(value=(1.2, 1.2, 1.0), constraint_axis=(True, True, False))
    bpy.ops.object.mode_set(mode='OBJECT')
    return vase


def voxelize_mesh_ground_truth(mesh_obj, resolution=128):
    """Voxelize mesh for ground truth IoU measurement."""
    # Get bounds
    bbox = [mesh_obj.matrix_world @ Vector(corner) for corner in mesh_obj.bound_box]
    bounds_min = Vector((
        min(v.x for v in bbox),
        min(v.y for v in bbox),
        min(v.z for v in bbox)
    ))
    bounds_max = Vector((
        max(v.x for v in bbox),
        max(v.y for v in bbox),
        max(v.z for v in bbox)
    ))

    # Create voxel grid
    voxel_grid = np.zeros((resolution, resolution, resolution), dtype=bool)

    # Convert to numpy for vectorized operations
    bounds_min_np = np.array([bounds_min.x, bounds_min.y, bounds_min.z])
    bounds_max_np = np.array([bounds_max.x, bounds_max.y, bounds_max.z])
    voxel_size = (bounds_max_np - bounds_min_np) / resolution

    print(f"  Voxelizing at {resolution}³ resolution...")
    voxels_inside = 0

    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                voxel_pos = bounds_min_np + (np.array([i, j, k]) + 0.5) * voxel_size
                point = Vector((voxel_pos[0], voxel_pos[1], voxel_pos[2]))

                # Tri-directional raycast
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

    return voxel_grid, bounds_min, bounds_max


def compute_iou(grid_a, grid_b):
    """Compute IoU between two voxel grids."""
    intersection = np.logical_and(grid_a, grid_b).sum()
    union = np.logical_or(grid_a, grid_b).sum()
    return intersection / union if union > 0 else 0.0


def render_silhouette_at_angle(mesh_obj, angle_degrees, output_path, resolution=512):
    """Render silhouette from specific angle."""
    import math

    # Setup camera
    angle_rad = math.radians(angle_degrees)
    distance = 5.0
    camera_loc = (distance * math.cos(angle_rad), distance * math.sin(angle_rad), 0)

    bpy.ops.object.camera_add(location=camera_loc)
    camera = bpy.context.active_object
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 4.0

    # Point at origin
    direction = Vector((0, 0, 0)) - Vector(camera_loc)
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

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

    # Set mesh to black
    if not mesh_obj.data.materials:
        mat = bpy.data.materials.new(name="Black")
        mat.use_nodes = True
        mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0, 0, 0, 1)
        mesh_obj.data.materials.append(mat)

    # Render
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


def render_turntable_silhouettes(mesh_obj, output_dir, num_views=12):
    """Render turntable silhouettes for Visual Hull."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    angle_step = 360.0 / num_views

    for i in range(num_views):
        angle = i * angle_step
        output_path = output_dir / f"view_{int(angle):03d}.png"
        render_silhouette_at_angle(mesh_obj, angle, output_path)

    # Top view
    bpy.ops.object.camera_add(location=(0, 0, 5.0))
    camera = bpy.context.active_object
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 4.0
    camera.rotation_euler = (0, 0, 0)
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.filepath = str(output_dir / "top.png")
    bpy.ops.render.render(write_still=True)


def create_visual_hull_mesh(turntable_dir, resolution=128):
    """Create Visual Hull mesh from turntable images."""
    from integration.image_processing.image_loader import load_multi_view_auto

    # Load views
    views_dict = load_multi_view_auto(str(turntable_dir), num_views=12, include_top=True)

    # Create Visual Hull
    hull = MultiViewVisualHull(
        resolution=resolution,
        bounds_min=np.array([-2.0, -2.0, -2.0]),
        bounds_max=np.array([2.0, 2.0, 2.0])
    )

    # Add views
    for view_name, (img, angle, view_type) in views_dict.items():
        if len(img.shape) == 3:
            img_gray = np.mean(img, axis=2)
        else:
            img_gray = img
        silhouette = img_gray < 128
        hull.add_view_from_silhouette(silhouette, angle=angle, view_type=view_type)

    # Reconstruct
    voxel_grid = hull.reconstruct(verbose=False)

    # Extract surface points
    points = hull.extract_mesh_points(surface_only=True)

    # Create Blender mesh from points
    mesh = bpy.data.meshes.new("VisualHull_Mesh")
    obj = bpy.data.objects.new("VisualHull_Mesh", mesh)
    bpy.context.collection.objects.link(obj)

    mesh.from_pydata(points.tolist(), [], [])
    mesh.update()

    return obj, voxel_grid


def test_phase2_pipeline():
    """Test complete Phase 2 pipeline."""
    print("\n" + "="*70)
    print("PHASE 2 INTEGRATION TEST")
    print("="*70)
    print("\nPipeline: Visual Hull → Multi-Profile → Primitives")
    print("Target: Boost from ~0.78 IoU to ~0.85-0.88 IoU")

    resolution = 128
    turntable_dir = Path("test_output/phase2_vase")

    # Step 1: Create ground truth vase
    print("\n" + "="*70)
    print("STEP 1: Create Ground Truth Vase")
    print("="*70)

    clear_scene()
    ground_truth_vase = create_test_mesh_vase()
    print("✓ Created vase mesh")

    # Voxelize for ground truth
    ground_truth_voxels, gt_bounds_min, gt_bounds_max = voxelize_mesh_ground_truth(
        ground_truth_vase, resolution=resolution
    )
    print(f"✓ Ground truth: {ground_truth_voxels.sum():,} voxels")

    # Step 2: Render turntable silhouettes
    print("\n" + "="*70)
    print("STEP 2: Render Turntable Silhouettes")
    print("="*70)

    render_turntable_silhouettes(ground_truth_vase, turntable_dir, num_views=12)
    print(f"✓ Rendered 13 views to {turntable_dir}")

    # Keep ground truth for later, create new scene for reconstruction
    ground_truth_copy = ground_truth_voxels.copy()

    # Step 3: Visual Hull reconstruction
    print("\n" + "="*70)
    print("STEP 3: Visual Hull Reconstruction (Phase 1)")
    print("="*70)

    clear_scene()
    visual_hull_mesh, visual_hull_voxels = create_visual_hull_mesh(turntable_dir, resolution=resolution)
    print(f"✓ Visual Hull mesh: {len(visual_hull_mesh.data.vertices):,} vertices")
    print(f"✓ Visual Hull voxels: {visual_hull_voxels.sum():,}")

    # Measure Visual Hull IoU
    iou_visual_hull = compute_iou(visual_hull_voxels, ground_truth_copy)
    print(f"✓ Visual Hull IoU vs ground truth: {iou_visual_hull:.4f}")

    # Step 4: Extract multi-angle profiles from Visual Hull mesh
    print("\n" + "="*70)
    print("STEP 4: Extract Multi-Angle Profiles (Phase 2)")
    print("="*70)

    profiles = extract_multi_angle_profiles(
        visual_hull_mesh,
        num_angles=12,
        num_heights=20,
        bounds_min=gt_bounds_min,
        bounds_max=gt_bounds_max
    )
    print(f"✓ Extracted {len(profiles)} profiles at different angles")

    # Combine profiles
    combined_profile = combine_profiles(profiles, method='median')
    print(f"✓ Combined using median: {len(combined_profile)} height samples")

    radii = [r for h, r in combined_profile]
    print(f"  Radius range: {min(radii):.3f} to {max(radii):.3f}")

    # Step 5: Place primitives using combined profile
    print("\n" + "="*70)
    print("STEP 5: Place Primitives from Multi-Profile")
    print("="*70)

    # Create SliceAnalyzer with combined profile
    analyzer = SliceAnalyzer(
        gt_bounds_min,
        gt_bounds_max,
        num_slices=20,
        vertical_profile=combined_profile
    )
    slice_data = analyzer.get_all_slice_data()
    print(f"✓ Analyzed {len(slice_data)} slices")

    # Place primitives
    placer = PrimitivePlacer()
    objects = placer.place_primitives_from_slices(slice_data, primitive_type='CYLINDER')
    print(f"✓ Placed {len(objects)} cylinder primitives")

    # Join primitives
    if objects:
        joiner = MeshJoiner()
        final_mesh = joiner.join_with_boolean_union(objects, target_name="Phase2_Primitives")
        print(f"✓ Joined into single mesh: {final_mesh.name}")

        # Step 6: Measure IoU of primitive mesh
        print("\n" + "="*70)
        print("STEP 6: Measure IoU Improvement")
        print("="*70)

        # Voxelize primitive mesh
        primitive_voxels, _, _ = voxelize_mesh_ground_truth(final_mesh, resolution=resolution)
        print(f"✓ Primitive mesh voxels: {primitive_voxels.sum():,}")

        # Measure IoU
        iou_primitives = compute_iou(primitive_voxels, ground_truth_copy)
        print(f"✓ Primitives IoU vs ground truth: {iou_primitives:.4f}")

        # Results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)

        improvement = iou_primitives - iou_visual_hull
        improvement_pct = (improvement / iou_visual_hull * 100) if iou_visual_hull > 0 else 0

        print(f"\nGround Truth: {ground_truth_copy.sum():,} voxels")
        print(f"\nVisual Hull (Phase 1):")
        print(f"  Voxels: {visual_hull_voxels.sum():,}")
        print(f"  IoU: {iou_visual_hull:.4f}")

        print(f"\nPrimitives (Phase 2):")
        print(f"  Voxels: {primitive_voxels.sum():,}")
        print(f"  IoU: {iou_primitives:.4f}")

        print(f"\nImprovement:")
        print(f"  Absolute: {improvement:+.4f}")
        print(f"  Relative: {improvement_pct:+.1f}%")

        print(f"\nTarget Assessment:")
        if iou_primitives >= 0.85:
            print(f"  ✓ SUCCESS: Achieved target (≥0.85)")
        else:
            gap = 0.85 - iou_primitives
            print(f"  ⚠ Below target by {gap:.4f}")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_phase2_pipeline()
