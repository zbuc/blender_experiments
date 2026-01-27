"""Blender-only tests for multi-angle mesh profile extraction."""

from __future__ import annotations

import unittest

try:
    import bpy

    BLENDER_AVAILABLE = True
except ImportError:
    bpy = None
    BLENDER_AVAILABLE = False

from integration.shape_matching.mesh_profile_extractor import (
    combine_profiles,
    extract_multi_angle_profiles,
)


@unittest.skipUnless(BLENDER_AVAILABLE, "Requires Blender bpy")
class TestMeshProfileExtractor(unittest.TestCase):
    def tearDown(self) -> None:
        if not BLENDER_AVAILABLE:
            return
        for obj in list(bpy.context.scene.objects):
            if obj.name.startswith("Test_Profile_"):
                bpy.data.objects.remove(obj, do_unlink=True)

    def test_cylinder_profiles(self) -> None:
        bpy.ops.mesh.primitive_cylinder_add(radius=1.0, depth=2.0, location=(0, 0, 0))
        cylinder = bpy.context.active_object
        cylinder.name = "Test_Profile_Cylinder"

        profiles = extract_multi_angle_profiles(cylinder, num_angles=8, num_heights=6)
        self.assertEqual(len(profiles), 8)
        self.assertTrue(all(len(profile) == 6 for profile in profiles))

        combined = combine_profiles(profiles, method="median")
        self.assertEqual(len(combined), 6)
        radii = [radius for _, radius in combined]
        self.assertTrue(all(radius >= 0.0 for radius in radii))


if __name__ == "__main__":
    unittest.main()
