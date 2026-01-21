"""
Integration test for SuperFrustum with primitive placement system.

This test verifies that SuperFrustum can be used with the PrimitivePlacer
class and MeshJoiner for boolean operations.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import bpy
    from mathutils import Vector
    from placement.primitive_placement import PrimitivePlacer, MeshJoiner, clear_scene

    def test_superfrustum_with_placer():
        """Test SuperFrustum integration with PrimitivePlacer."""
        print("\n" + "="*70)
        print("INTEGRATION TEST: SuperFrustum with PrimitivePlacer")
        print("="*70)

        # Clear scene
        clear_scene()

        # Create placer
        placer = PrimitivePlacer()

        # Create multiple SuperFrustum primitives
        print("\n  Creating SuperFrustum primitives...")

        # Bottom frustum (wider at base)
        obj1 = placer.create_primitive(
            primitive_type='SUPERFRUSTUM',
            location=(0, 0, 0),
            scale=(1, 1, 1),
            radius_bottom=2.0,
            radius_top=1.5,
            height=2.0
        )
        obj1.name = "SuperFrustum_Base"
        print(f"    ✓ Created {obj1.name}")

        # Middle frustum
        obj2 = placer.create_primitive(
            primitive_type='SUPERFRUSTUM',
            location=(0, 0, 2),
            scale=(1, 1, 1),
            radius_bottom=1.5,
            radius_top=1.0,
            height=2.0
        )
        obj2.name = "SuperFrustum_Middle"
        print(f"    ✓ Created {obj2.name}")

        # Top cone (radius_top = 0)
        obj3 = placer.create_primitive(
            primitive_type='SUPERFRUSTUM',
            location=(0, 0, 4),
            scale=(1, 1, 1),
            radius_bottom=1.0,
            radius_top=0.0,
            height=2.0
        )
        obj3.name = "SuperFrustum_Top"
        print(f"    ✓ Created {obj3.name}")

        print(f"\n  Created {len(placer.placed_objects)} SuperFrustum objects")

        # Join with boolean union
        print("\n  Joining with boolean union...")
        joiner = MeshJoiner()
        final_mesh = joiner.join_with_boolean_union(
            placer.placed_objects,
            target_name="SuperFrustum_Stack"
        )

        print(f"    ✓ Created final mesh: {final_mesh.name}")
        print(f"    ✓ Vertices: {len(final_mesh.data.vertices)}")
        print(f"    ✓ Faces: {len(final_mesh.data.polygons)}")

        print("\n✓ SuperFrustum integration test PASSED")
        return True

    def test_mixed_primitives():
        """Test mixing SuperFrustum with traditional primitives."""
        print("\n" + "="*70)
        print("INTEGRATION TEST: Mixed Primitives (SuperFrustum + Cylinder)")
        print("="*70)

        # Clear scene
        clear_scene()

        # Create placer
        placer = PrimitivePlacer()

        # Create base cylinder
        obj1 = placer.create_primitive(
            primitive_type='CYLINDER',
            location=(0, 0, 0),
            scale=(1.5, 1.5, 1.0)
        )
        obj1.name = "Base_Cylinder"
        print(f"  ✓ Created {obj1.name}")

        # Create SuperFrustum on top
        obj2 = placer.create_primitive(
            primitive_type='SUPERFRUSTUM',
            location=(0, 0, 2),
            scale=(1, 1, 1),
            radius_bottom=1.5,
            radius_top=0.5,
            height=3.0
        )
        obj2.name = "Top_SuperFrustum"
        print(f"  ✓ Created {obj2.name}")

        # Join
        joiner = MeshJoiner()
        final_mesh = joiner.join_with_boolean_union(
            [obj1, obj2],
            target_name="Mixed_Shape"
        )

        print(f"  ✓ Created final mesh: {final_mesh.name}")
        print(f"  ✓ Vertices: {len(final_mesh.data.vertices)}")

        print("\n✓ Mixed primitives test PASSED")
        return True

    def run_integration_tests():
        """Run all integration tests."""
        print("\n" + "="*70)
        print("SUPERFRUSTUM INTEGRATION TEST SUITE")
        print("="*70)

        tests = [
            ("SuperFrustum with PrimitivePlacer", test_superfrustum_with_placer),
            ("Mixed Primitives", test_mixed_primitives),
        ]

        results = {}
        for name, test_func in tests:
            try:
                result = test_func()
                results[name] = result
            except Exception as e:
                print(f"\n❌ {name} FAILED with exception:")
                print(f"   {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                results[name] = False

        # Print summary
        print("\n" + "="*70)
        print("INTEGRATION TEST SUMMARY")
        print("="*70)

        passed = sum(1 for v in results.values() if v is True)
        failed = sum(1 for v in results.values() if v is False)

        for name, result in results.items():
            status = "✓ PASS" if result else "❌ FAIL"
            print(f"  {name:.<50} {status}")

        print("="*70)
        print(f"\nResults: {passed} passed, {failed} failed")

        if failed > 0:
            print("\n❌ INTEGRATION TESTS FAILED")
            return 1
        else:
            print("\n✓ ALL INTEGRATION TESTS PASSED")
            return 0

    if __name__ == "__main__":
        exit_code = run_integration_tests()
        sys.exit(exit_code)

except ImportError:
    print("❌ ERROR: Must run in Blender environment")
    print("\nUsage:")
    print("  blender --background --python test_superfrustum_integration.py")
    sys.exit(2)
