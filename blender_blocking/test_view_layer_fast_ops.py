"""Blender-only tests for view layer reminder suppression helper."""

from __future__ import annotations

try:
    import bpy

    BLENDER_AVAILABLE = True
except ImportError:
    bpy = None
    BLENDER_AVAILABLE = False

from integration.blender_ops.ops_fastpath import suppress_view_layer_updates


def test_view_layer_update_suppression() -> bool:
    """Ensure suppressed view layer updates still allow object creation."""
    if not BLENDER_AVAILABLE:
        raise RuntimeError("Blender API not available")

    before_names = {obj.name for obj in bpy.data.objects}

    def add_cubes() -> None:
        for i in range(5):
            bpy.ops.mesh.primitive_cube_add(location=(i * 2.0, 0.0, 0.0))

    with suppress_view_layer_updates(enabled=True) as active:
        add_cubes()

    after_names = {obj.name for obj in bpy.data.objects}
    new_names = sorted(after_names - before_names)

    if len(new_names) < 5:
        raise AssertionError("Expected at least 5 new objects")

    # Cleanup created objects without using bpy.ops.
    for name in new_names:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)

    return True


if __name__ == "__main__":
    test_view_layer_update_suppression()
