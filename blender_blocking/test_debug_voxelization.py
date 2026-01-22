"""
Debug voxelization issues in Phase 2 integration test.

Tests voxelization in isolation to identify why all voxels are marked as inside.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import bpy
import numpy as np
from mathutils import Vector


def clear_scene():
    """Remove all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def create_simple_vase():
    """Create simple vase mesh."""
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=2.0, location=(0, 0, 0))
    vase = bpy.context.active_object
    vase.name = "TestVase"
    return vase


def voxelize_mesh_debug(mesh_obj, resolution=32, use_fixed_bounds=False):
    """Voxelize with detailed debugging."""
    print(f"\n{'='*70}")
    print("VOXELIZATION DEBUG")
    print(f"{'='*70}")

    # Get bounds
    if use_fixed_bounds:
        # Use fixed large bounds like test_ground_truth_iou.py
        bounds_min = Vector((-2.0, -2.0, -2.0))
        bounds_max = Vector((2.0, 2.0, 2.0))
        print("Using FIXED bounds: [-2, -2, -2] to [2, 2, 2]")
    else:
        # Use mesh bounding box
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
        print("Using MESH bounds from bounding box")

    print(f"\nMesh: {mesh_obj.name}")
    print(f"  Vertices: {len(mesh_obj.data.vertices)}")
    print(f"  Faces: {len(mesh_obj.data.polygons)}")
    print(f"  Bounds: {bounds_min} to {bounds_max}")

    # Create voxel grid
    voxel_grid = np.zeros((resolution, resolution, resolution), dtype=bool)

    # Use numpy arrays for bounds
    bounds_min_arr = np.array([bounds_min.x, bounds_min.y, bounds_min.z])
    bounds_max_arr = np.array([bounds_max.x, bounds_max.y, bounds_max.z])
    voxel_size = (bounds_max_arr - bounds_min_arr) / resolution

    print(f"\nVoxel size: {voxel_size}")
    print(f"Resolution: {resolution}³ = {resolution**3:,} voxels")

    # Test a few specific voxels
    test_voxels = [
        (0, 0, 0),  # Corner
        (resolution//2, resolution//2, resolution//2),  # Center
        (resolution//2, resolution//2, 0),  # Bottom center
        (0, 0, resolution-1),  # Top corner
    ]

    print(f"\nTesting specific voxels:")
    for idx in test_voxels:
        i, j, k = idx
        voxel_pos = bounds_min_arr + (np.array([i, j, k]) + 0.5) * voxel_size
        point = Vector((voxel_pos[0], voxel_pos[1], voxel_pos[2]))

        print(f"\n  Voxel ({i}, {j}, {k}) at position {point}:")

        # Test raycasts in each direction
        hit_count = 0
        for ray_dir in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            direction = Vector(ray_dir)
            origin = point - direction * 10

            result, location, normal, index = mesh_obj.ray_cast(
                origin,
                direction,
                distance=20.0
            )

            print(f"    Ray from {origin} dir {ray_dir}: hit={result}")
            if result:
                print(f"      Hit at {location}, distance={(location-origin).length:.3f}")
                hit_count += 1

        inside = hit_count >= 2
        print(f"    → Hit count: {hit_count}/3, Inside: {inside}")

    # Now do full voxelization with sampling
    print(f"\nFull voxelization (every 4th voxel for speed):")
    voxels_inside = 0
    voxels_tested = 0

    for i in range(0, resolution, 4):
        for j in range(0, resolution, 4):
            for k in range(0, resolution, 4):
                voxel_pos = bounds_min_arr + (np.array([i, j, k]) + 0.5) * voxel_size
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
                    voxels_inside += 1

                voxels_tested += 1

    occupancy = voxels_inside / voxels_tested if voxels_tested > 0 else 0
    print(f"\n  Tested: {voxels_tested:,} voxels")
    print(f"  Inside: {voxels_inside:,} voxels")
    print(f"  Occupancy: {occupancy*100:.2f}%")
    print(f"\n  Expected for cylinder: ~20-30%")

    if occupancy > 0.9:
        print(f"\n  ✗ ERROR: {occupancy*100:.0f}% occupancy - raycast likely broken")
    elif occupancy < 0.1:
        print(f"\n  ✗ ERROR: {occupancy*100:.0f}% occupancy - mesh likely empty or raycast missing")
    else:
        print(f"\n  ✓ OK: {occupancy*100:.0f}% occupancy looks reasonable")


def test_voxelization():
    """Test voxelization debug."""
    print("="*70)
    print("VOXELIZATION DEBUG TEST")
    print("="*70)

    clear_scene()

    # Test 1: Simple cylinder
    print("\nTEST 1: Simple Cylinder")
    print("-"*70)
    vase = create_simple_vase()

    print("\nVase mesh info:")
    print(f"  Location: {vase.location}")
    print(f"  Rotation: {vase.rotation_euler}")
    print(f"  Scale: {vase.scale}")
    print(f"  Matrix world:\n{vase.matrix_world}")

    voxelize_mesh_debug(vase, resolution=32)

    # Test 2: Same mesh with FIXED bounds
    print("\n\nTEST 2: Same Mesh with FIXED Bounds [-2,-2,-2] to [2,2,2]")
    print("-"*70)
    voxelize_mesh_debug(vase, resolution=32, use_fixed_bounds=True)

    # Test 3: After scene modifications
    print("\n\nTEST 3: After Adding Camera (simulating rendering setup)")
    print("-"*70)
    bpy.ops.object.camera_add(location=(5, 0, 0))
    voxelize_mesh_debug(vase, resolution=32, use_fixed_bounds=True)

    print("\n" + "="*70)
    print("DEBUG COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_voxelization()
