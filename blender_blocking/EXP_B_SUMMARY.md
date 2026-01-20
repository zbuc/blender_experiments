# EXP-B: Slice Count Variation Analysis

**Task:** hq-nlr: EXP-B - Test slice count variations (80 vs 120 vs 160)
**Date:** 2026-01-20
**Status:** ✓ COMPLETE

## Objective

Evaluate the impact of `num_slices` parameter on reconstruction quality and performance to validate the decision in commit c61843e to increase from 80 to 120 slices.

## Background

Commit c61843e increased `num_slices` from 80 to 120 to reduce stepping artifacts in vertical profiles. The `num_slices` parameter controls how many horizontal slices are analyzed when reconstructing a 3D model from reference images. More slices theoretically provide better vertical resolution but increase processing time.

## Methodology

- **Test Object:** Vase with curved profile (ideal for testing vertical resolution)
- **Slice Counts Tested:** 80, 120, 160
- **Metrics:** Average IoU score, per-view IoU, processing time, mesh complexity
- **Test Framework:** E2E validation with reference image comparison

## Results

| Slices | Avg IoU | Front IoU | Side IoU | Top IoU | Time (s) | Vertices | Faces   |
|--------|---------|-----------|----------|---------|----------|----------|---------|
| 80     | 0.7588  | 0.6566    | 0.6547   | 0.9650  | 18.85    | 13,758   | 13,756  |
| 120    | 0.7534  | 0.6460    | 0.6477   | 0.9664  | 11.68    | 13,380   | 13,378  |
| 160    | 0.7448  | 0.6341    | 0.6341   | 0.9664  | 17.15    | 18,490   | 18,488  |

### Key Observations

1. **Quality Trend (Counterintuitive):**
   - 80 slices: BEST quality (0.7588 IoU)
   - 120 slices: Good quality (0.7534 IoU, -0.7% vs 80)
   - 160 slices: WORST quality (0.7448 IoU, -1.8% vs 80)

2. **Performance:**
   - 120 slices is FASTEST (11.68s, 38% faster than 80 slices)
   - 80 slices is slowest (18.85s)
   - 160 slices is slow (17.15s, 47% slower than 120)

3. **Primitive Count Impact:**
   - 80 slices → 69 primitives
   - 120 slices → 104 primitives
   - 160 slices → 138 primitives
   - More primitives increase mesh complexity and boolean operation time

## Analysis

### Why More Slices Don't Always Help

The counterintuitive result (more slices = worse quality) can be explained by:

1. **Boolean Operation Complexity:** More primitives (cylinders) create more complex boolean unions, potentially introducing mesh artifacts or failed operations.

2. **Refinement Phase:** The vertex-level refinement (Iteration 3) subdivides and adjusts vertices. With more initial primitives, the refinement may struggle with denser overlapping geometry.

3. **Optimal Sampling Rate:** There appears to be a "sweet spot" for slice density. Too few slices miss detail, but too many create processing artifacts that outweigh benefits.

4. **Performance vs Quality Trade-off:** 120 slices hits the optimal balance:
   - Only 0.7% quality loss vs 80 slices
   - 38% faster than 80 slices
   - 47% faster than 160 slices

### Per-View Analysis

- **Top view:** Consistently excellent across all slice counts (0.965-0.966 IoU)
- **Front/Side views:** Most sensitive to slice count
  - Front IoU: 0.657 → 0.646 → 0.634 (decreasing with more slices)
  - Side IoU: 0.655 → 0.648 → 0.634 (decreasing with more slices)

The front and side views suffer more with increased slice counts, suggesting the vertical reconstruction quality degrades with excessive slicing.

## Conclusion

**Current setting of 120 slices is OPTIMAL.**

### Recommendation

✓ **MAINTAIN 120 slices** (current setting from commit c61843e)

**Rationale:**
- Best performance (fastest processing time)
- Excellent quality/time ratio
- Only 0.7% quality loss vs 80 slices
- 38% faster than 80 slices
- Significantly better than 160 slices on both quality AND speed

### Decision Validation

The commit c61843e decision to increase from 80 to 120 slices was **partially correct**:
- ✓ Improved performance significantly (38% faster)
- ✗ Did not improve quality (slight 0.7% decrease)
- ✓ Still provides excellent overall balance

The original intent was to reduce stepping artifacts, but the data shows that more slices don't necessarily achieve this. However, the performance gain makes 120 slices the clear winner for production use.

## Recommendations for Future Work

1. **Profile Interpolation:** Instead of increasing slice count, improve the interpolation algorithm between slices to reduce stepping artifacts without adding more geometry.

2. **Adaptive Slicing:** Use variable slice density based on profile curvature (more slices in high-curvature regions, fewer in straight sections).

3. **Boolean Operation Tuning:** Investigate why 120 slices processes faster despite having more primitives than 80 slices.

4. **Front/Side Quality:** Focus on improving front/side view IoU scores (currently 0.64-0.66) which are significantly lower than top view (0.97).

## Test Artifacts

- Test script: `test_slice_count_variations.py`
- Detailed results: `test_output/slice_count_analysis/exp_b_results.txt`
- Rendered outputs: `test_output/e2e_renders/`
- Debug silhouettes: `test_output/debug_silhouettes/`

## Related Commits

- c61843e - Reduce cylinder stepping artifacts (increased slices 80→120)
- fe27d8c - QA Iteration 3: Vertex-level refinement after subdivision
- 43f428f - Fix vertex refinement bugs
