#!/usr/bin/env python3
"""
Blender Test Runner - Runs all tests in headless Blender

This is the main test runner for CI/CD and local testing.
MUST be run inside Blender to execute actual Blender API tests.

Usage:
    blender --background --python test_runner.py
    blender --background --python test_runner.py -- --verbose
    blender --background --python test_runner.py -- --quick  # Skip slow tests

Exit codes:
    0: All tests passed
    1: One or more tests failed
    2: Test runner error (missing Blender, etc.)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_blender_available():
    """Verify we're running inside Blender."""
    try:
        import bpy
        return True, bpy.app.version_string
    except ImportError:
        return False, None


def run_test_suite(verbose=False, quick=False):
    """
    Run complete Blender test suite.

    Args:
        verbose: Enable verbose output
        quick: Skip slow tests

    Returns:
        dict: Test results with pass/fail status
    """
    results = {}

    print("\n" + "="*70)
    print("BLENDER BLOCKING TOOL - TEST SUITE")
    print("="*70)

    # Check Blender availability
    blender_ok, version = check_blender_available()
    if not blender_ok:
        print("\n❌ CRITICAL: Must run inside Blender")
        print("\nUsage:")
        print("  blender --background --python test_runner.py")
        print("="*70)
        sys.exit(2)

    print(f"\n✓ Running in Blender {version}")
    print(f"✓ Python {sys.version.split()[0]}")

    # Test 1: Version compatibility
    print("\n" + "-"*70)
    print("[1/6] Version Compatibility")
    print("-"*70)
    try:
        from test_version_compatibility import test_version_detection, test_boolean_solver_compatibility
        version_ok = test_version_detection()
        solver_ok = test_boolean_solver_compatibility()
        results['version_compatibility'] = version_ok and solver_ok
    except Exception as e:
        print(f"❌ Test crashed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        results['version_compatibility'] = False

    # Test 2: Boolean solver enum
    print("\n" + "-"*70)
    print("[2/6] Boolean Solver Enum")
    print("-"*70)
    try:
        from test_blender_boolean import test_boolean_solver_enum
        results['boolean_solver'] = test_boolean_solver_enum()
    except Exception as e:
        print(f"❌ Test crashed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        results['boolean_solver'] = False

    # Test 3: MeshJoiner integration
    print("\n" + "-"*70)
    print("[3/6] MeshJoiner Integration")
    print("-"*70)
    try:
        from test_blender_boolean import test_mesh_joiner
        results['mesh_joiner'] = test_mesh_joiner()
    except Exception as e:
        print(f"❌ Test crashed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        results['mesh_joiner'] = False

    # Test 4: Full workflow test
    if not quick:
        print("\n" + "-"*70)
        print("[4/6] Full Workflow (Procedural)")
        print("-"*70)
        try:
            from test_integration import test_procedural_generation
            results['procedural_workflow'] = test_procedural_generation()
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            results['procedural_workflow'] = False
    else:
        print("\n" + "-"*70)
        print("[4/6] Full Workflow (SKIPPED - quick mode)")
        print("-"*70)
        results['procedural_workflow'] = None

    # Test 5: E2E Validation (Reference → 3D → Render → Compare)
    if not quick:
        print("\n" + "-"*70)
        print("[5/6] E2E Validation (Image → Mesh → Render → IoU)")
        print("-"*70)
        try:
            from test_e2e_validation import test_with_sample_images
            results['e2e_validation'] = test_with_sample_images()
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            results['e2e_validation'] = False
    else:
        print("\n" + "-"*70)
        print("[5/6] E2E Validation (SKIPPED - quick mode)")
        print("-"*70)
        results['e2e_validation'] = None

    # Test 6: Dependency verification
    print("\n" + "-"*70)
    print("[6/6] Dependency Check")
    print("-"*70)
    try:
        # sys already imported at module level - no need to reimport
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from blender_blocking.verify_setup import verify_setup
        results['dependencies'] = verify_setup()
    except Exception as e:
        print(f"❌ Test crashed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        results['dependencies'] = False

    return results


def print_summary(results):
    """Print test summary and return exit code."""
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = 0
    failed = 0
    skipped = 0

    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
            passed += 1
        elif result is False:
            status = "❌ FAIL"
            failed += 1
        else:
            status = "- SKIP"
            skipped += 1

        print(f"  {test_name:.<50} {status}")

    print("="*70)
    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\n❌ TESTS FAILED")
        print("="*70)
        return 1
    else:
        print("\n✓ ALL TESTS PASSED")
        print("="*70)
        return 0


def main():
    """Main entry point for test runner."""
    # Parse arguments (simple manual parsing since we're in Blender)
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    quick = '--quick' in sys.argv or '-q' in sys.argv

    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        sys.exit(0)

    # Run tests
    results = run_test_suite(verbose=verbose, quick=quick)

    # Print summary and exit with appropriate code
    exit_code = print_summary(results)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
