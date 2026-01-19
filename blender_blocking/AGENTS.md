# Agent Quality Gates for Blender Blocking Tool

This document defines quality gates and testing requirements for agents and crews working on the Blender Blocking Tool.

## Overview

**Critical principle**: All tests MUST run in actual Blender. Mocked tests give false confidence and miss real API issues (like the boolean solver enum bug that made it to production).

## Pre-Commit Quality Gates

Before committing code changes, agents MUST:

### 1. Run Quick Test Suite

```bash
cd blender_blocking
blender --background --python test_runner.py -- --quick
```

**Exit codes:**
- `0` = All tests passed ✓ → Safe to commit
- `1` = Tests failed ❌ → Fix before committing
- `2` = Runner error ⚠️ → Check Blender installation

**What quick mode tests:**
- Version compatibility detection
- Boolean solver enum compatibility
- MeshJoiner integration
- Dependency verification

**Skipped in quick mode** (too slow for pre-commit):
- Full workflow test
- E2E validation with rendering

### 2. Verify No Breaking Changes

If you modified Blender API usage (modifiers, boolean operations, rendering):
```bash
# Run full test suite including E2E validation
blender --background --python test_runner.py
```

### 3. Code Quality Checks

- No new Blender API calls without corresponding tests
- Boolean operations must use valid Blender 5.0 enums (`EXACT`, `FLOAT`, `MANIFOLD`)
- All new Blender-dependent code must have test coverage

## Pre-Merge Quality Gates

Before merging to main, the following MUST pass:

### Required: Multi-Version Testing

```bash
# Test with Blender 5.0
blender-5.0 --background --python test_runner.py

# Test with Blender 4.2 (LTS)
blender-4.2 --background --python test_runner.py
```

Both must exit with code `0`.

### Required: Full Test Suite

All 6 test suites must pass:
1. ✅ Version Compatibility
2. ✅ Boolean Solver Enum
3. ✅ MeshJoiner Integration
4. ✅ Full Workflow (Procedural)
5. ✅ E2E Validation (Image → Mesh → Render → IoU)
6. ✅ Dependency Check

### Required: No Regressions

- IoU scores in E2E validation must be ≥ 0.7
- Mesh creation must complete without errors
- No boolean operation failures
- No orphan data blocks in scene

## Common Failures and Fixes

### "enum 'FAST' not found"

**Cause**: Using deprecated Blender API enum

**Fix**: Update to Blender 5.0 compatible enum:
```python
# Wrong (Blender < 5.0)
modifier.solver = 'FAST'

# Correct (Blender 5.0+)
modifier.solver = 'EXACT'  # or 'FLOAT', 'MANIFOLD'
```

### "ImportError: cannot import name '_imaging' from 'PIL'"

**Cause**: Pillow installed in venv, not Blender's Python

**Fix**: Install directly to Blender's Python:
```bash
# Find Blender's Python
blender --background --python-expr "import sys; print(sys.executable)"

# Install to Blender's Python
/path/to/blender/python -m pip install numpy opencv-python Pillow scipy
```

See [BLENDER_SETUP.md](BLENDER_SETUP.md) for complete setup guide.

### "Test timed out"

**Cause**: E2E validation rendering takes too long in CI

**Solutions:**
- Use `--quick` mode for pre-commit
- Reduce `num_slices` in test cases to 8-10
- Skip E2E validation in quick mode (already implemented)

## CI/CD Integration

### GitHub Actions Workflow

See [.github/workflows/blender-tests.yml](.github/workflows/blender-tests.yml) for complete workflow.

**Key points:**
- Tests run in Docker containers with real Blender
- Multi-version testing: Blender 5.0 and 4.2 (LTS)
- Artifacts uploaded on failure for debugging

### Local Docker Testing

Test locally before pushing:

```bash
# Pull Blender container
docker pull nytimes/blender:5.0-cpu-ubuntu22.04

# Run tests in container
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace/blender_blocking \
  nytimes/blender:5.0-cpu-ubuntu22.04 \
  bash -c "
    BLENDER_PYTHON=\$(blender --background --python-expr 'import sys; print(sys.executable)' 2>&1 | grep -oP '/[^ ]+python[0-9.]*')
    \$BLENDER_PYTHON -m pip install numpy opencv-python Pillow scipy
    blender --background --python test_runner.py
  "
```

## Test Coverage Requirements

When adding new features:

### Blender API Usage

**Any code that calls Blender API must have tests:**

```python
# Example: Adding new boolean operation
def join_meshes_with_custom_solver(meshes, solver='EXACT'):
    # Implementation...
    modifier.solver = solver  # ← Blender API call
```

**Required test:**
```python
def test_custom_solver():
    """Test that custom solver enum is valid."""
    # Create test objects
    # Call join_meshes_with_custom_solver
    # Assert no TypeError about enum
    # Assert mesh created successfully
```

### Rendering Operations

Any rendering code must have IoU validation:

```python
def test_new_render_feature():
    """Test new rendering feature."""
    # Render with new feature
    # Compare to reference with IoU
    # Assert IoU ≥ 0.7
```

### Shape Matching

New shape analysis must be validated:

```python
def test_new_shape_analysis():
    """Test new shape analysis algorithm."""
    # Create test shapes
    # Run analysis
    # Assert expected properties
    # Validate against known good results
```

## When Tests Fail

### Step 1: Identify Which Test Failed

```bash
# Run with verbose output
blender --background --python test_runner.py -- --verbose
```

Look for the test that crashed or failed.

### Step 2: Run Individual Test

```bash
# Run specific test file
blender --background --python test_blender_boolean.py

# Or specific function
blender --background --python -c "
from test_blender_boolean import test_boolean_solver_enum
test_boolean_solver_enum()
"
```

### Step 3: Debug

Add print statements or use Blender's logging:

```python
import bpy
print(f"Blender version: {bpy.app.version_string}")
print(f"Available solvers: {bpy.types.BooleanModifier.bl_rna.properties['solver'].enum_items.keys()}")
```

### Step 4: Fix and Verify

After fixing:
```bash
# Re-run full suite
blender --background --python test_runner.py

# If passes, commit
git add <files>
git commit -m "Fix: <description>"
git push
```

## Quality Standards

### Code Changes

- ✅ All tests pass in Blender 5.0
- ✅ All tests pass in Blender 4.2 (LTS)
- ✅ No new deprecation warnings
- ✅ Exit code 0 from test_runner.py
- ✅ IoU scores ≥ 0.7 in E2E validation

### Documentation Changes

- ✅ Updated README.md if adding new features
- ✅ Updated TESTING.md if changing test structure
- ✅ Updated CI_CD.md if changing CI configuration
- ✅ Code examples tested in actual Blender

### Test Changes

- ✅ New tests run in actual Blender (not mocked)
- ✅ Tests validate real API behavior
- ✅ Tests have clear failure messages
- ✅ Tests clean up after themselves (no orphan data blocks)

## Emergency Procedures

### Production Bug Detected

1. **Stop work on current task**
2. **Reproduce locally** with actual Blender
3. **Write failing test** that catches the bug
4. **Fix the bug**
5. **Verify test now passes**
6. **Run full test suite**
7. **Commit with detailed message explaining root cause**

### Test Suite Broken

If test_runner.py itself is broken:

1. **Check Blender version**: `blender --version`
2. **Verify dependencies**: `blender --background --python verify_setup.py`
3. **Check for obvious errors**: Import failures, syntax errors
4. **Bisect to find breaking commit**: `git bisect`
5. **Fix and verify**: Re-run test suite

## Contact

For questions about testing or quality gates:
- See [TESTING.md](TESTING.md) for detailed test documentation
- See [CI_CD.md](CI_CD.md) for CI/CD configuration
- Check [test_runner.py](test_runner.py) source for test implementation details

## Summary

**Key Principle**: If you modify code that uses Blender API, you MUST test it in actual Blender.

**Minimum Quality Gate**: `blender --background --python test_runner.py -- --quick` exits with code 0

**Full Quality Gate**: `blender --background --python test_runner.py` exits with code 0 in both Blender 5.0 and 4.2
