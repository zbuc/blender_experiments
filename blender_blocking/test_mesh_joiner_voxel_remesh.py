"""Blender-only test for MeshJoiner voxel remesh mode."""

from __future__ import annotations

import sys
from pathlib import Path

# Add blender_blocking directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_voxel_remesh_join() -> bool:
    """Verify voxel remesh join creates a non-empty mesh."""
    try:
        import bpy
    except ImportError:
        print("Blender not available, skipping voxel remesh test")
        return False

    from placement.primitive_placement import MeshJoiner

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
    obj1 = bpy.context.active_object
    bpy.ops.mesh.primitive_cube_add(location=(0.5, 0, 0))
    obj2 = bpy.context.active_object

    joiner = MeshJoiner()
    result = joiner.join_with_voxel_remesh([obj1, obj2], target_name="VoxelRemeshTest")

    if not result:
        print("Voxel remesh join failed")
        return False

    if len(result.data.vertices) == 0:
        print("Voxel remesh result has no vertices")
        return False

    print("✓ Voxel remesh join produced a valid mesh")
    return True


if __name__ == "__main__":
    success = test_voxel_remesh_join()
    sys.exit(0 if success else 1)
