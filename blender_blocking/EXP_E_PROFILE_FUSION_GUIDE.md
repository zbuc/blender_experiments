# EXP-E: Multi-View Profile Fusion - Implementation Guide

## Overview

**Experiment ID:** be-9hx
**Goal:** Improve front/side IoU from 0.64-0.66 → 0.68-0.72 by fusing front and side view profiles
**Expected Impact:** +2-5% IoU improvement

## What Was Implemented

### 1. Profile Fusion Function (`profile_extractor.py`)

Added three new functions to `integration/shape_matching/profile_extractor.py`:

#### `calculate_profile_confidence(profile, image=None)`
Calculates a confidence score (0-1) for a profile based on:
- **Smoothness** (40%): Lower variance in radii = higher confidence
- **Coverage** (30%): Percentage of non-zero radii
- **Fill quality** (30%): Ratio of filled pixels in silhouette (if image provided)

#### `fuse_profiles(front_profile, side_profile, fusion_strategy, ...)`
Fuses two vertical profiles using weighted averaging. Supports:
- **E1 ("equal")**: Equal weights (0.5, 0.5)
- **E2 ("front_heavy")**: Front-heavy (0.6, 0.4)
- **E3 ("side_heavy")**: Side-heavy (0.4, 0.6)
- **E4 ("adaptive")**: Adaptive weights based on profile confidence
- **"custom"**: Custom weights via front_weight/side_weight parameters

The function:
1. Validates both profiles (same length, matching heights)
2. Determines weights based on strategy
3. Computes weighted average of radii
4. Normalizes to [0, 1] range
5. Returns fused profile

### 2. Updated Workflow (`main_integration.py`)

Modified `BlockingWorkflow.create_3d_blockout()` to:
- Extract profiles from **both** front and side views (instead of front OR side)
- Fuse them using the specified strategy
- Pass the fusion strategy through the workflow

Added `profile_fusion_strategy` parameter to:
- `create_3d_blockout(num_slices, primitive_type, profile_fusion_strategy)`
- `run_full_workflow(num_slices, profile_fusion_strategy)`

### 3. Test Scripts

Created two test scripts:

#### `test_profile_fusion_unit.py`
Unit tests for fusion logic (7 test cases):
- Equal weights fusion (E1)
- Front-heavy fusion (E2)
- Side-heavy fusion (E3)
- Adaptive fusion (E4)
- Custom weights
- Confidence calculation
- Error handling

#### `test_profile_fusion_exp.py`
Full E2E experiment runner that:
- Runs all 4 strategies (E1-E4)
- Generates 3D mesh for each
- Renders from front/side/top views
- Calculates IoU scores
- Compares results
- Saves results to `exp_e_results.json`

## How to Run Experiments

### Prerequisites

1. **Blender installed** with dependencies in Blender's Python:
   ```bash
   # Find Blender's Python (macOS example)
   BLENDER_PYTHON="/Applications/Blender.app/Contents/Resources/4.0/python/bin/python3.11"

   # Install dependencies
   $BLENDER_PYTHON -m pip install numpy opencv-python Pillow scipy
   ```
   See `BLENDER_SETUP.md` for detailed instructions.

2. **Test images** - Create them first:
   ```bash
   # Using Blender's Python (since system Python may not have Pillow)
   $BLENDER_PYTHON create_test_images.py
   ```
   This creates test images in `test_images/` directory:
   - `bottle_front.png`, `bottle_side.png`, `bottle_top.png`
   - `vase_front.png`, `vase_side.png`, `vase_top.png`
   - `cube_front.png`, `cube_side.png`, `cube_top.png`

### Option 1: Run Full E2E Experiments (Recommended)

This runs all 4 strategies and compares results:

```bash
# Run with default test images (vase)
blender --background --python test_profile_fusion_exp.py

# Or specify custom images
blender --background --python test_profile_fusion_exp.py -- \
  test_images/vase_front.png \
  test_images/vase_side.png \
  test_images/vase_top.png
```

**Output:**
- Prints IoU scores for each strategy
- Saves results to `exp_e_results.json`
- Shows which strategy performed best
- Calculates improvement vs baseline

### Option 2: Run Single Strategy in Blender GUI

Test a specific fusion strategy interactively:

```python
# In Blender's scripting workspace:
import sys
sys.path.insert(0, "/path/to/blender_experiments/polecats/chrome/blender_experiments")

from blender_blocking.main_integration import BlockingWorkflow

# Create workflow
workflow = BlockingWorkflow(
    front_path="test_images/vase_front.png",
    side_path="test_images/vase_side.png",
    top_path="test_images/vase_top.png"
)

# Run with specific fusion strategy
mesh = workflow.run_full_workflow(
    num_slices=12,
    profile_fusion_strategy="adaptive"  # or "equal", "front_heavy", "side_heavy"
)
```

### Option 3: Run Unit Tests

Test the fusion logic without Blender (requires numpy):

```bash
# If you have numpy installed in system Python
cd blender_blocking
python3 test_profile_fusion_unit.py

# Or use Blender's Python
$BLENDER_PYTHON test_profile_fusion_unit.py
```

## Understanding the Results

### Success Criteria

- **Front/Side IoU:** Should improve from 0.64-0.66 → 0.68-0.72 (+6-12%)
- **Top IoU:** Should maintain ~0.97 (no regression)
- **Average IoU:** Should improve from 0.75 → 0.79-0.80 (+5-7%)

### Interpreting Output

The experiment script outputs:

```
EXPERIMENT SUMMARY
======================================================================
Experiment    Front IoU    Side IoU     Top IoU      Avg IoU
----------------------------------------------------------------------
E1            0.6523       0.6812       0.9701       0.7679
E2            0.6601       0.6789       0.9705       0.7698
E3            0.6489       0.6835       0.9698       0.7674
E4            0.6712       0.6923       0.9702       0.7779
======================================================================

ANALYSIS
======================================================================
Best Average IoU: E4 (0.7779)

E1: Front/Side avg = 0.6668 (Δ+0.0168)
E2: Front/Side avg = 0.6695 (Δ+0.0195)
E3: Front/Side avg = 0.6662 (Δ+0.0162)
E4: Front/Side avg = 0.6818 (Δ+0.0318)
======================================================================
```

**What to look for:**
1. **Best strategy:** Which has highest average IoU?
2. **Front/Side improvement:** Did it increase from baseline (0.65)?
3. **Top view regression:** Did top IoU stay at ~0.97?
4. **Overall improvement:** Did average IoU increase?

### Expected Results

Based on the hypothesis:
- **E4 (Adaptive)** should perform best because it weights profiles by quality
- Front/Side IoU should improve by 2-5 percentage points
- Top view should remain stable (no regression)

## Integration with Existing Workflow

The profile fusion is now the **default behavior** when both front and side views are provided:

```python
# Old behavior: Used front OR side (whichever was available first)
workflow = BlockingWorkflow(front_path="...", side_path="...")
workflow.run_full_workflow()  # Used only front

# New behavior: Fuses front AND side with equal weights by default
workflow = BlockingWorkflow(front_path="...", side_path="...")
workflow.run_full_workflow()  # Fuses both with equal weights

# Specify strategy explicitly
workflow.run_full_workflow(profile_fusion_strategy="adaptive")
```

**Backward compatibility:**
- If only front view provided: Uses front profile (no change)
- If only side view provided: Uses side profile (no change)
- If both provided: Fuses them (NEW - uses "equal" by default)

## Files Modified

1. **`integration/shape_matching/profile_extractor.py`** (+135 lines)
   - Added `calculate_profile_confidence()`
   - Added `fuse_profiles()`

2. **`main_integration.py`** (~80 lines modified)
   - Updated `create_3d_blockout()` signature
   - Refactored profile extraction (lines 269-347)
   - Updated `run_full_workflow()` signature
   - Added profile fusion logic

3. **`test_profile_fusion_unit.py`** (NEW, 250 lines)
   - Unit tests for fusion logic

4. **`test_profile_fusion_exp.py`** (NEW, 280 lines)
   - E2E experiment runner

5. **`EXP_E_PROFILE_FUSION_GUIDE.md`** (NEW, this file)
   - Implementation documentation

## Next Steps

1. **Run Experiments:** Execute `test_profile_fusion_exp.py` with test images
2. **Analyze Results:** Review `exp_e_results.json` and console output
3. **Validate Improvement:** Confirm front/side IoU improved by 2-5%
4. **Choose Best Strategy:** Determine which strategy (E1-E4) works best
5. **Update Default:** If needed, change default from "equal" to best strategy
6. **Real-World Testing:** Test with actual reference images from users

## Troubleshooting

### "ModuleNotFoundError: No module named 'numpy'"
- Dependencies not installed in Blender's Python
- Solution: Follow `BLENDER_SETUP.md` to install dependencies

### "ValueError: Profiles must have same length"
- Front and side images have different heights
- Solution: Ensure num_slices is consistent for both profiles

### "No improvement in IoU"
- Test images may be too simple or identical
- Solution: Use more realistic reference images with asymmetry

### Experiments take too long
- Blender rendering can be slow
- Solution: Reduce num_slices (e.g., from 12 to 8)

## Contact

For questions or issues with the profile fusion implementation:
- Check existing documentation: `INTEGRATION.md`, `TESTING.md`
- Review unit tests: `test_profile_fusion_unit.py`
- Run experiments: `test_profile_fusion_exp.py`

---

**Implementation Status:** ✅ COMPLETE
**Testing Status:** ⏳ PENDING (awaiting Blender execution)
**Expected Delivery:** +2-5% IoU improvement in front/side views
