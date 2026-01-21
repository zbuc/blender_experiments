"""
Test Multi-View Silhouette Consistency Validation (EXP-J)

Tests geometric consistency between orthogonal views of 3D reconstructions.

This complements the existing E2E validation (test_e2e_validation.py) which
compares rendered views to reference images. This test checks that the
rendered views are geometrically consistent with EACH OTHER.

Usage:
    # In Blender (with GUI)
    Run this script in Blender's scripting workspace

    # Headless (for CI/CD)
    blender --background --python test_multiview_consistency.py

    # Combined E2E + Consistency validation
    blender --background --python test_full_validation.py
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add ~/blender_python_packages for user-installed dependencies
sys.path.insert(0, str(Path.home() / 'blender_python_packages'))

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    print("ERROR: This test must be run inside Blender")
    sys.exit(1)

import numpy as np
from blender_blocking.main_integration import BlockingWorkflow
from blender_blocking.integration.blender_ops.render_utils import render_orthogonal_views
from blender_blocking.multi_view_consistency import MultiViewConsistencyValidator


def test_consistency_with_sample_images():
    """
    Test multi-view consistency with built-in sample images.

    Returns:
        bool: Test passed
    """
    base_dir = Path(__file__).parent
    test_images_dir = base_dir / 'test_images'

    # Check if test images exist
    if not test_images_dir.exists():
        print("Creating test images...")
        import subprocess
        result = subprocess.run(
            [sys.executable, str(base_dir / 'create_test_images.py')],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("ERROR: Failed to create test images")
            print(result.stderr)
            return False

    # Use vase test images (good for testing consistency)
    reference_paths = {
        'front': str(test_images_dir / 'vase_front.png'),
        'side': str(test_images_dir / 'vase_side.png'),
        'top': str(test_images_dir / 'vase_top.png')
    }

    print("="*60)
    print("MULTI-VIEW CONSISTENCY TEST - SAMPLE IMAGES")
    print("="*60)

    # Step 1: Generate 3D model
    print("\n[1/3] Generating 3D model from reference images...")
    workflow = BlockingWorkflow(
        front_path=reference_paths.get('front'),
        side_path=reference_paths.get('side'),
        top_path=reference_paths.get('top')
    )
    mesh = workflow.run_full_workflow(num_slices=120)

    if not mesh:
        print("ERROR: Failed to generate 3D model")
        return False

    print(f"✓ Generated mesh: {mesh.name}")

    # Step 2: Setup and render orthogonal views
    print("\n[2/3] Rendering orthogonal views...")

    # Configure render settings
    scene = bpy.context.scene
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.taa_render_samples = 1

    # Render views
    output_dir = Path(__file__).parent / 'test_output' / 'consistency_renders'
    rendered_paths = render_orthogonal_views(str(output_dir))

    if not rendered_paths:
        print("ERROR: Failed to render views")
        return False

    for view, path in rendered_paths.items():
        print(f"✓ Rendered {view}: {path}")

    # Step 3: Validate consistency
    print("\n[3/3] Validating multi-view consistency...")
    validator = MultiViewConsistencyValidator(tolerance=0.05)
    passed, results = validator.validate_consistency(rendered_paths)

    # Print detailed results
    validator.print_detailed_results()

    return passed


def test_consistency_with_simple_shape():
    """
    Test multi-view consistency with a simple cylinder (should pass easily).

    Returns:
        bool: Test passed
    """
    print("="*60)
    print("MULTI-VIEW CONSISTENCY TEST - SIMPLE CYLINDER")
    print("="*60)

    # Create a simple cylinder in Blender
    print("\n[1/3] Creating simple cylinder...")
    bpy.ops.mesh.primitive_cylinder_add(radius=1, depth=2, location=(0, 0, 0))
    mesh = bpy.context.active_object
    print(f"✓ Created cylinder: {mesh.name}")

    # Setup and render
    print("\n[2/3] Rendering orthogonal views...")
    scene = bpy.context.scene
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.taa_render_samples = 1

    output_dir = Path(__file__).parent / 'test_output' / 'cylinder_renders'
    rendered_paths = render_orthogonal_views(str(output_dir))

    if not rendered_paths:
        print("ERROR: Failed to render views")
        return False

    for view, path in rendered_paths.items():
        print(f"✓ Rendered {view}: {path}")

    # Validate consistency
    print("\n[3/3] Validating multi-view consistency...")
    validator = MultiViewConsistencyValidator(tolerance=0.05)
    passed, results = validator.validate_consistency(rendered_paths)

    validator.print_detailed_results()

    # Clean up
    bpy.data.objects.remove(mesh, do_unlink=True)

    return passed


def test_consistency_with_custom_images(front, side, top):
    """
    Test multi-view consistency with custom reference images.

    Args:
        front: Path to front view image
        side: Path to side view image
        top: Path to top view image

    Returns:
        bool: Test passed
    """
    reference_paths = {
        'front': front,
        'side': side,
        'top': top
    }

    print("="*60)
    print("MULTI-VIEW CONSISTENCY TEST - CUSTOM IMAGES")
    print("="*60)

    # Generate 3D model
    print("\n[1/3] Generating 3D model from reference images...")
    workflow = BlockingWorkflow(
        front_path=reference_paths.get('front'),
        side_path=reference_paths.get('side'),
        top_path=reference_paths.get('top')
    )
    mesh = workflow.run_full_workflow(num_slices=12)

    if not mesh:
        print("ERROR: Failed to generate 3D model")
        return False

    print(f"✓ Generated mesh: {mesh.name}")

    # Render
    print("\n[2/3] Rendering orthogonal views...")
    scene = bpy.context.scene
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.taa_render_samples = 1

    output_dir = Path(__file__).parent / 'test_output' / 'custom_renders'
    rendered_paths = render_orthogonal_views(str(output_dir))

    if not rendered_paths:
        print("ERROR: Failed to render views")
        return False

    for view, path in rendered_paths.items():
        print(f"✓ Rendered {view}: {path}")

    # Validate
    print("\n[3/3] Validating multi-view consistency...")
    validator = MultiViewConsistencyValidator(tolerance=0.05)
    passed, results = validator.validate_consistency(rendered_paths)

    validator.print_detailed_results()

    return passed


def run_all_tests():
    """
    Run all consistency tests.

    Returns:
        bool: All tests passed
    """
    results = []

    print("\n" + "="*60)
    print("RUNNING ALL MULTI-VIEW CONSISTENCY TESTS")
    print("="*60 + "\n")

    # Test 1: Simple cylinder (should easily pass)
    print("\n### TEST 1: Simple Cylinder ###")
    try:
        result = test_consistency_with_simple_shape()
        results.append(('Simple Cylinder', result))
        print(f"\nTest 1 Result: {'✓ PASSED' if result else '✗ FAILED'}\n")
    except Exception as e:
        print(f"\nTest 1 ERROR: {e}\n")
        results.append(('Simple Cylinder', False))

    # Test 2: Sample images (vase)
    print("\n### TEST 2: Sample Images (Vase) ###")
    try:
        result = test_consistency_with_sample_images()
        results.append(('Sample Images', result))
        print(f"\nTest 2 Result: {'✓ PASSED' if result else '✗ FAILED'}\n")
    except Exception as e:
        print(f"\nTest 2 ERROR: {e}\n")
        results.append(('Sample Images', False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results:
        status = '✓ PASSED' if passed else '✗ FAILED'
        print(f"{test_name:30s} {status}")
    print("="*60)

    all_passed = all(result for _, result in results)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")

    return all_passed


if __name__ == "__main__":
    # Run all tests
    success = run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
