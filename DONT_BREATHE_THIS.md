# DONT_BREATHE_THIS - Unified Roadmap and Implementation Blueprint (Expanded)

## Purpose
Provide a single, exhaustive, sequenced plan that merges all roadmaps/specs under `docs/` into a foundation-first implementation guide. The dual-profile + loft + canonical IoU pipeline remains optional behind a flag so the legacy pipeline continues to work even if the new path is incomplete.

## Locked Decisions and Constraints
- Foundations first. Complete Phase 0 before any new reconstruction features.
- Parallel optional path. New dual-profile/loft/IoU pipeline is opt-in; legacy remains default.
- Agent environment cannot run Blender and will not have Blender available at all. Tests must be designed clearly and then validated by a human run.
- Avoid duplicate module trees. Use `blender_blocking/geometry/` and `blender_blocking/validation/` as the canonical locations for new logic.
- Do not delete or rewrite the legacy pipeline. Keep it available and tested.

## Global Implementation Rules
- No `bpy` imports in `geometry/` or `validation/` modules.
- Use deterministic behavior everywhere (fixed seeds, no random without context).
- Default config values must preserve current behavior until a phase explicitly changes them.
- Any new API must be introduced behind a compatibility wrapper or re-export to avoid breaking old imports.
- Any new rendering or mesh generation must avoid `bpy.ops` where a bmesh or direct data API exists.
- Every phase includes tests and updates to the test runner.

## Current Entry Points (Do Not Break)
These functions are used by tests and examples. Keep their behavior stable unless the phase explicitly says otherwise.
- `blender_blocking/main_integration.py`
  - `BlockingWorkflow.__init__` (near top of class)
  - `load_images`, `process_images`, `analyze_shapes`
  - `calculate_bounds_from_shapes` (mid-file)
  - `create_3d_blockout` (legacy path, mid-file)
  - `run_full_workflow` (near bottom)
- `blender_blocking/placement/primitive_placement.py`
  - `SliceAnalyzer.__init__`, `SliceAnalyzer.get_all_slice_data`, `_interpolate_profile`
  - `PrimitivePlacer.place_primitives_from_slices`
  - `MeshJoiner.join_with_boolean_union`, `MeshJoiner.join_simple`
- `blender_blocking/integration/shape_matching/profile_extractor.py`
  - `extract_silhouette_from_image`, `extract_vertical_profile`
- `blender_blocking/integration/blender_ops/render_utils.py`
  - `render_orthogonal_views`
- `blender_blocking/integration/shape_matching/shape_matcher.py`
  - `compare_silhouettes`
- `blender_blocking/test_e2e_validation.py`
  - `E2EValidator.extract_silhouette`, `E2EValidator.validate_reconstruction`

## Canonical Data Contracts (Single Source of Truth)
Create these types in `blender_blocking/geometry/profile_models.py`:
- `BBox2D`: `x0,y0` inclusive; `x1,y1` exclusive. Add `w`, `h` properties.
- `PixelScale(unit_per_px)` with `from_target_height(target_height_units, silhouette_height_px)`.
- `VerticalWidthProfilePx`: arrays for `heights_t`, `left_x`, `right_x`, `width_px`, `center_x`, `valid`, plus `bbox` and `source_view`.
- `EllipticalProfileU`: `heights_t`, `rx`, `ry`, `height_u`, `z0`, optional `cx/cy`, optional `meta`.
- `EllipticalSlice`: `z`, `rx`, `ry`, optional `cx/cy`.

Optional compatibility alias (no duplicate definitions):
- `blender_blocking/integration/shape_matching/contracts.py` re-exports these types and names used in older docs.
Status: ✅ Implemented in `blender_blocking/geometry/profile_models.py` and `blender_blocking/integration/shape_matching/contracts.py`.

## Phase 0 - Foundations (Must Finish Before Anything Else)
### 0.1 GenerationContext (Determinism + Run Lifecycle)
Tasks and subtasks:
- 0.1.0 Fix `main_integration.check_setup()` import side-effects: ✅ Done
  - Do not run setup checks at module import time in non-Blender contexts.
  - Gate setup checks behind an explicit call or only when Blender is available.
- 0.1.1 Create `blender_blocking/utils/generation_context.py` (pure Python): ✅ Done
  - Dataclass with fields from `docs/IMPLEMENTATION_SPEC.md` lines 251-260 (Section 15: Generation Context + Manifest).
  - Enforce defaults that match current behavior: `unit_scale=0.01`, `num_slices` default existing value.
  - Provide `apply_seed()` that seeds `random` and `numpy` if available.
- 0.1.2 Implement logging and timing: ✅ Done
  - `log(level, message, **fields)` emits consistent, grep-friendly lines that include `run_id`.
  - `time_block(stage)` context manager records elapsed ms in an internal list for the manifest.
- 0.1.3 Add serialization: ✅ Done
  - `to_dict()` must return JSON-serializable values only.
  - Include all config and toggle fields that affect output.
- 0.1.4 Integrate with workflow: ✅ Done
  - Update `BlockingWorkflow.__init__` to accept `context` (optional). If not provided, construct a default context.
  - At start of `run_full_workflow`, call `context.apply_seed()`.
  - Wrap stages: load_images, process_images, analyze_shapes, create_3d_blockout in `time_block()`.
- 0.1.5 Edge cases: ✅ Done
  - Handle missing numpy gracefully in `apply_seed()`.
  - Do not import `bpy` in `generation_context.py`.

Tests (pure Python):
- `blender_blocking/utils/test_generation_context.py`: ✅ Done
  - `apply_seed` determinism: two runs with same seed produce same random numbers.
  - `to_dict` JSON-safe: `json.dumps(context.to_dict())` works.
  - `time_block` writes stage names and elapsed time.

### 0.2 BlockingConfig (Centralized Configuration)
Tasks and subtasks:
- 0.2.1 Create `blender_blocking/config.py` with dataclasses: ✅ Done
  - `SilhouetteExtractConfig`, `ProfileSamplingConfig`, `LoftMeshOptions`, `RenderConfig`, `CanonicalizeConfig`, `MeshJoinConfig`, `BlockingConfig`.
- 0.2.2 Defaults must match current behavior: ✅ Done
  - `unit_scale=0.01`, `num_slices` default as in `main_integration.py`.
  - `mesh_join_mode="boolean"` or equivalent to preserve current boolean unions.
- 0.2.3 Validation helper: ✅ Done
  - `validate()` (or `__post_init__`) enforces non-negative values and mutually exclusive scale policies.
- 0.2.4 Integration points: ✅ Done
  - `BlockingWorkflow.__init__` accepts `config` and stores it on `self.config`.
  - If `config is None`, construct defaults.
  - Replace local constants in `main_integration.py` with config values (scale, num_slices, etc.).
- 0.2.5 Fix hard-coded canonical sizing in mesh generation: ✅ Done
  - In `integration/blender_ops/mesh_generator.create_mesh_from_contours`, replace hard-coded `256` normalization with `config.canonical_size` or contour-derived width/height (passed in explicitly).
  - Update any call sites to pass the size source so mesh generation stays correct when canonical size is adjusted.
- 0.2.6 Harden contour-to-mesh creation: ✅ Done
  - In `create_mesh_from_contours`, de-duplicate consecutive contour points and drop near-identical vertices before building a face.
  - If fewer than 3 unique vertices remain, return a clear error instead of creating a zero-geometry mesh.
  - Catch `bm.faces.new(...)` failures for self-intersecting contours and fall back to a triangulated fill (or return a clear error).

Tests (pure Python):
- `blender_blocking/test_config_defaults.py`: ✅ Done
  - Defaults align with legacy behavior.
  - Invalid values raise clear errors.

### 0.3 Metadata Tags + Manifest
Tasks and subtasks:
- 0.3.1 Create `blender_blocking/utils/manifest.py`: ✅ Done
  - `apply_object_tags(obj, role, context, index=None, params=None)`.
  - `build_manifest(context, outputs, warnings, errors)`.
  - `write_manifest(scene, manifest)` storing to `scene["blocktool_manifest"]`.
- 0.3.2 Update `main_integration.create_3d_blockout`: ✅ Done
  - Tag each primitive with `blocktool_role="primitive"` and `blocktool_index`.
  - Tag final mesh with `blocktool_role="final"`.
- 0.3.3 If `context.dry_run=True`: ✅ Done
  - Do not create Blender objects but still build a manifest with zero counts.
- 0.3.4 Add minimal outputs schema: ✅ Done
  - Output counts, object names, bounds.

Tests (pure Python):
- `blender_blocking/utils/test_manifest_schema.py`: ✅ Done
  - Validate required manifest keys exist.
  - Validate `run_id` consistency.

### 0.4 Heuristics and Scaling
Tasks and subtasks:
- 0.4.0 Fix legacy silhouette-based bounds logic in `main_integration.create_3d_blockout`: ✅ Done
  - If both front and side views exist, set width from front and depth from side (do not force square depth).
  - If top view exists and one of front/side is missing, use top to derive the missing axis.
  - Centralize bounds computation so silhouette-derived bounds and shape-derived bounds follow the same scale rules.
- 0.4.1 Update `SliceAnalyzer` to use `min_radius_ratio` and `z_overlap_ratio`: ✅ Done
  - Apply `min_radius = min_radius_ratio * max_dim`.
  - Guardrail: fix current divide-by-zero in `SliceAnalyzer.get_all_slice_data` when `num_slices == 1` and place a single slice at mid-height.
  - Reject `num_slices <= 0` with a clear error before any math.
  - Use a consistent `slice_thickness` (derived from bounds and `num_slices`) for Z scale, not mixed formulas.
- 0.4.2 Update `calculate_bounds_from_shapes`: ✅ Done
  - Replace fixed `scale=0.01` with `config.unit_scale`.
- 0.4.3 Add warnings to manifest for degenerate bounds. ✅ Done
 - 0.4.4 Replace hard-coded `radius > 0.01` in `PrimitivePlacer.place_primitives_from_slices` with scale-aware `min_radius`. ✅ Done

Tests (pure Python):
- `blender_blocking/placement/test_primitive_placement_math.py`: ✅ Done
  - `num_slices==1` does not crash and uses sensible z.
  - `min_radius_ratio` and `z_overlap_ratio` yield expected heights and skips.

### 0.5 Slice-Based Shape Matcher Correctness
Tasks and subtasks:
- 0.5.0 Fix `shape_matching/slice_shape_matcher.py` profile geometry: ✅ Done
  - Order slice intersection points around the centroid (angle sort) before computing area/perimeter.
  - De-duplicate near-identical intersection points to avoid zero-length edges.
  - If fewer than 3 ordered points remain, treat area/perimeter as 0 and skip in comparisons.
- 0.5.1 Add a deterministic test: ✅ Done
  - Create a known circular cross-section, verify computed area/perimeter are within tolerance.
  - Add a non-convex slice case to ensure ordering still produces sane results.
- 0.5.2 Input validation: ✅ Done
  - Reject `num_slices <= 0` in `SliceBasedShapeMatcher.__init__` or `slice_mesh` with a clear error.

### 0.6 Documentation Integrity (Required)
Tasks and subtasks:
- 0.6.0 Fix missing referenced docs under `blender_blocking/`: ✅ Done
  - Create `blender_blocking/TESTING.md` and `blender_blocking/AGENTS.md`, or update all references to point to root docs.
  - Ensure `blender_blocking/README.md`, `QUICKSTART.md`, and `BLENDER_SETUP.md` no longer reference missing files.
- 0.6.1 Resolve contradictory setup guidance: ✅ Done
  - In `BLENDER_SETUP.md`, remove or explicitly mark as unsupported the “sys.path insert venv site-packages” workaround (conflicts with the “only supported method” guidance).
- 0.6.2 Update project status claims: ✅ Done
  - In `blender_blocking/README.md`, replace “✅ Complete and tested” with an accurate status that reflects ongoing fixes and roadmap work.
- 0.6.3 Align E2E docs with current implementation: ✅ Done
  - In `E2E_VALIDATION_SUMMARY.md` and `E2E_TESTING_FEASIBILITY.md`, remove “production-ready / all components work” claims and update for canonical IoU + framing changes (Phase 5).
- 0.6.4 Align supported Blender versions across docs: ✅ Done
  - Update README/QUICKSTART/INTEGRATION to match the versions actually tested in CI (4.2 LTS + 5.0) and clarify older versions are unverified.
- 0.6.5 Normalize example paths: ✅ Done
  - Replace hard-coded `blender_experiments/crew/sculptor` paths with repo-root placeholders and ensure all examples are consistent.

## Phase 1 - Robust Image Processing and Silhouettes
### 1.1 RGBA-Safe Image Processing
Tasks and subtasks:
- 1.1.0 Fix current RGBA edge extraction crash: ✅ Done
  - `extract_edges` should never call `cv2.cvtColor` on 4-channel input.
  - `normalize_image` should not pass RGBA through unmodified.
- 1.1.0a Preserve alpha when loading via OpenCV fallback: ✅ Done
  - In `image_loader.load_image`, use `cv2.imread(..., cv2.IMREAD_UNCHANGED)` so PNG alpha isn’t discarded.
  - Convert BGR/BGRA to RGB/RGBA consistently so silhouette extraction sees correct channels.
- 1.1.1 Add `_to_gray_uint8(image, prefer_alpha)` to `image_processor.py`. ✅ Done
  - RGBA: if alpha varies, return alpha as grayscale.
  - RGB: convert to gray with cv2.
  - Grayscale: pass through.
- 1.1.2 Update `normalize_image` to use `_to_gray_uint8`. ✅ Done
- 1.1.3 Update `extract_edges` to ensure `uint8` gray input and avoid shape mismatches. ✅ Done

Tests:
- Add `blender_blocking/test_image_processor_rgba.py`: ✅ Done
  - Use synthetic RGBA silhouette, ensure `process_image` returns `uint8` and has edges.

### 1.2 Canonical Silhouette Extraction (Pure Python)
Tasks and subtasks:
- 1.2.1 Add `blender_blocking/geometry/silhouette.py`: ✅ Done
  - `extract_binary_silhouette` with `mode="auto"` polarity detection.
  - `bbox_from_mask` returns `BBox2D` or raises if empty.
- 1.2.2 Implement morphological clean (close then open) and fill holes. ✅ Done
- 1.2.3 Ensure output dtype is `bool`. ✅ Done
- 1.2.4 Provide a legacy wrapper in `profile_extractor.py`: ✅ Done
  - `extract_silhouette_from_image` returns 0/255 uint8 mask by calling the canonical function.

### 1.3 Vertical Profile Extraction (BBox-Aware)
Tasks and subtasks:
- 1.3.0 Fix `extract_vertical_profile` division by zero when `num_samples == 1`: ✅ Done
  - Clamp `num_samples` to >=2 or handle single-sample case explicitly.
- 1.3.0b Reject `num_samples <= 0` with a clear error before sampling. ✅ Done
- 1.3.0a Fix off-by-one width measurement in `profile_extractor.extract_vertical_profile`: ✅ Done
  - When computing width from `left_edge` and `right_edge`, use `(right_edge - left_edge + 1)` so widths reflect pixel counts.
- 1.3.0c Reject empty silhouettes: ✅ Done
  - If the filled mask has zero foreground pixels, raise a clear error (do not synthesize a fake profile from width).
  - Add a test that passes an all-background image and asserts the error path.
- 1.3.1 Update `extract_vertical_profile` to accept `bbox` and `already_silhouette`. ✅ Done
- 1.3.2 Crop to bbox before scanning rows. ✅ Done
- 1.3.3 Interpolate NaNs, clamp, then normalize. ✅ Done
- 1.3.4 Preserve output format (List[(height_norm, width_norm)]) for legacy callers. ✅ Done

Tests:
- `blender_blocking/test_silhouette_extraction.py`: ✅ Done
  - Black-on-white and white-on-black produce same mask after auto-invert.
  - RGBA alpha is handled and bbox is tight.
  - Profile extraction is invariant to padding.

### 1.4 Contour + Shape Matching Guardrails
Tasks and subtasks:
- 1.4.1 Harden `contour_analyzer.find_contours`: ✅ Done
  - If `edge_image` is None or empty, return `[]` with a warning instead of calling OpenCV.
  - If `edge_image` is 3-channel, convert to grayscale before `cv2.findContours`.
- 1.4.2 Harden `shape_matcher.match_shapes` and `compare_silhouettes`: ✅ Done
  - If either contour is empty or has < 3 points, raise a clear error (or return a sentinel score) instead of calling `cv2.matchShapes`.
  - If either image is empty, raise a clear error or return IoU=0 with a warning.

## Phase 2 - Dual-Profile Extraction (Pure Python)
### 2.1 Vertical Width Profile Sampling
Tasks and subtasks:
- 2.1.1 Add `extract_vertical_width_profile_px` in `geometry/dual_profile.py`. ✅ Done
  - For each sampled row, compute `left`, `right`, `width_px`, `center_x`.
  - Support `sample_policy` (`endpoints` vs `cell_centers`).
  - Missing rows: fill using linear or nearest interpolation, or constant fallback.
- 2.1.2 Apply smoothing (median filter) on widths. ✅ Done

### 2.2 Build Elliptical Profile
Tasks and subtasks:
- 2.2.1 Implement `build_elliptical_profile_from_views`: ✅ Done
  - Build front and side width profiles if masks provided.
  - Compute `height_u` via `height_strategy` (front/side/max/mean).
  - Convert widths to radii using `PixelScale`.
  - Apply `fallback_policy` if only one view present.
  - Clamp radii to `min_radius_u`.
- 2.2.2 Optional center offsets: ✅ Done
  - Compute `cx/cy` from `center_x` and scale into world units if enabled.

### 2.3 Slice Sampling
Tasks and subtasks:
- 2.3.1 Implement `sample_elliptical_slices` in `geometry/slicing.py`. ✅ Done
  - Interpolate `rx/ry` at each slice `t`.
  - Compute `z = z0 + t * height_u`.

Tests:
- `blender_blocking/test_profile_sampling.py`: ✅ Done
  - Rectangle mask yields constant width profile.
- `blender_blocking/test_elliptical_profile.py`: ✅ Done
  - Front width and side width map to `rx` and `ry` correctly.
  - Single-view fallback sets both axes equal.
- `blender_blocking/test_slice_sampling.py`: ✅ Done
  - Endpoints vs cell centers produce expected `z` values.

## Phase 3 - Mesh-From-Profile Loft (Blender bmesh)
### 3.1 Loft Mesh Builder
Tasks and subtasks:
- 3.1.1 Add `blender_blocking/integration/blender_ops/profile_loft_mesh.py`. ✅ Done
- 3.1.2 Implement bmesh-only loft algorithm: ✅ Done
  - Build rings for each slice (or point rings if degenerate).
  - Create quad strips between rings (triangles if ring->point).
  - Add caps if `cap_ends=True`.
- 3.1.3 Apply optional remove doubles and recalc normals. ✅ Done
- 3.1.4 Link to collection or scene, return object. ✅ Done

Tests (Blender):
- `blender_blocking/test_profile_loft_mesh.py`: ✅ Done
  - Cylinder bounds test.
  - Elliptical cylinder bounds test.
  - Manifoldness heuristic.
  - Determinism check.

## Phase 4 - Optional New Reconstruction Path (Parallel)
### 4.1 Loft Pipeline Integration
Tasks and subtasks:
- 4.1.1 Add `create_3d_blockout_loft(self, num_slices)` in `main_integration.py`. ✅ Done
- 4.1.2 Pipeline steps: ✅ Done
  1. Load views and extract silhouettes (front/side).
  2. Build `PixelScale` from config policy.
  3. Build `EllipticalProfileU`.
  4. Sample slices and create mesh via loft builder.
  5. Setup scene, camera, lighting.
- 4.1.3 Add config flag `reconstruction_mode`: ✅ Done
  - `legacy` uses existing `create_3d_blockout`.
  - `loft_profile` uses new path.
- 4.1.4 Fallbacks: ✅ Done
  - If no front or side, log warning and fall back to legacy procedural or raise error.

Tests:
- Update `test_integration.py` to allow running legacy by default. ✅ Done (already default)
- Add a new Blender test `test_loft_workflow.py` that sets `reconstruction_mode="loft_profile"`. ✅ Done

## Phase 5 - Robust IoU and Render Framing
### 5.1 Camera Framing
Tasks and subtasks:
- 5.1.0 Fix current fixed `ortho_scale=5.0` in `render_utils.render_orthogonal_views`: ✅ Done
  - Replace with bounds-based framing to avoid cropped renders.
- 5.1.1 Add `camera_framing.py`: ✅ Done
  - `compute_bounds_world(objects)` using bound_box and matrix_world.
  - `configure_ortho_camera_for_view(...)` per spec.
- 5.1.2 Update `render_utils.render_orthogonal_views` to use camera framing: ✅ Done
  - Pass `fit_to_bounds`, `margin_frac`, `resolution`.
- 5.1.3 Harden orthogonal rendering outputs: ✅ Done
  - Set `scene.render.image_settings.file_format='PNG'` and explicit `color_mode` (RGBA or BW) inside `render_orthogonal_views` to match output filenames.
  - Guard world background node access (`scene.world` or `Background` node may be missing); create if needed.
  - Allow passing explicit target objects (or filter by `blocktool_role="final"`) so bounds and framing ignore unrelated meshes.

### 5.2 Canonical IoU
Tasks and subtasks:
- 5.2.0 Fix current `compare_silhouettes` interpolation artifacts: ✅ Done
  - Stop using default `cv2.resize` on binary masks; use canonicalization and nearest-neighbor.
- 5.2.1 Add `validation/silhouette_iou.py`: ✅ Done
  - `mask_from_image_array` (auto alpha/luma).
  - `canonicalize_mask` (crop, pad, resize nearest, bottom-center).
  - `compute_mask_iou` returns IoUResult.
- 5.2.2 Update `shape_matcher.compare_silhouettes` to call new IoU and preserve detail keys. ✅ Done
- 5.2.3 Validate empty-mask edge cases: ✅ Done
  - If either canonicalized mask has zero foreground pixels, return IoU=0 and include a warning in the details payload.
  - Add a unit test that compares two empty masks and asserts IoU=0 with warning metadata.

### 5.3 E2E Validation Upgrade
Tasks and subtasks:
- 5.3.1 Update `test_e2e_validation.py`: ✅ Done
  - Use canonical IoU.
  - Save debug images when failing.
  - Use per-view thresholds.

Tests:
- `blender_blocking/test_silhouette_iou.py` (pure Python): ✅ Done
  - Resolution invariance.
  - Padding invariance.
  - Bottom-center alignment.

## Phase 6 - Mesh Join Modes (Legacy Performance)
### 6.1 MeshJoiner Enhancements
Tasks and subtasks:
- 6.1.1 Add `MeshJoiner.join` with mode switching. ✅ Done
- 6.1.2 Implement `join_with_voxel_remesh` with voxel size heuristic. ✅ Done
- 6.1.3 Update legacy `create_3d_blockout` to call `join(mode=config.mesh_join_mode)`. ✅ Done

Tests (Blender):
- `test_mesh_joiner_voxel_remesh.py`: ✅ Done
  - Remesh mode creates a non-empty mesh.
- Extend `test_version_compatibility.py` to validate `join(mode="auto")`. ✅ Done

## Phase 7 - Test Runner and Quality Gates
Tasks and subtasks:
- 7.1 Update `test_runner.py`: ✅ Done
  - Add pure-Python tests before Blender-only tests.
  - Keep `--quick` to skip E2E and heavy mesh tests.
- 7.2 Update documentation references (TESTING.md) for new tests and flags. ✅ Done
- 7.3 Harden voxelization progress logging in tests: ✅ Done
  - In `test_ground_truth_iou.py`, `test_phase2_integration.py`, and `test_phase2_groundtruth_profiles.py`, guard `total_voxels // 10` so modulo never divides by zero for small resolutions.
  - Use `progress_every = max(total_voxels // 10, 1)` or similar to avoid `ZeroDivisionError`.
- 7.4 Fix example script import reliability: ✅ Done
  - In `primitives/example_primitives.py`, add a sys.path insert for the repo root (or switch to `from primitives.primitives import ...`) so the script runs from Blender without manual path setup.
- 7.5 Expand pure-Python test coverage: ✅ Done
  - Add tests for profile model contracts, config validation edge cases, contour analysis guards, shape matching errors, and silhouette IoU utilities.
  - Register new pure-Python tests in `test_runner.py`.

Test design rules:
- Pure Python tests: synthetic masks, no IO, no Blender.
- Blender tests: minimal geometry, explicit cleanup, assert bounds and vertex counts.
- E2E tests: save debug artifacts on failure and log IoU details.

## Phase 8 - ResFit Hygiene (Required)
Tasks and subtasks:
- 8.0 Fix class naming typo and import stability: ✅ Done
  - Rename `ResiduaFitter` to `ResidualFitter` in `placement/resfitting.py`.
  - Provide backward-compatible alias so tests/imports using `ResiduaFitter` still work.
- 8.1 Guard empty inputs and zero counts: ✅ Done
  - In `ResidualFitter.fit`, raise a clear error if `target_points` is empty.
  - In `optimize_primitives` and `compute_residual_error`, handle empty `target_points` without division by zero.
  - In `initialize_from_voxels` and `fit` initialization, validate `num_initial >= 1` and handle `num_initial == 0` gracefully.
  - In `initialize_from_slices`, validate `num_initial >= 1` before computing `slices_per_primitive`.
  - In `compute_residual_error`, if `primitives` is empty, return a clear error instead of `inf` errors.
  - In `add_primitive_at_error_region`, handle empty `per_point_errors` (avoid `np.percentile` on empty arrays).
  - In `optimize_primitives`, return early if `primitives` is empty or `steps <= 0`.
- 8.2 Tests: ✅ Done
  - Extend `placement/test_resfitting.py` and `placement/test_resfitting_quick.py` to cover empty `target_points` and `num_initial==0` error paths.
- 8.3 Fix ResFit test correctness + determinism: ✅ Done
  - In `placement/test_resfitting.py`, compute `initial_error` BEFORE calling `optimize_primitives` (it mutates the primitive in place).
  - Seed numpy RNG at test start (or per test) in `placement/test_resfitting.py` and `placement/test_resfitting_quick.py` to avoid flakey assertions.

## Phase 9 - Mesh Profile Extraction / Multi-View Hygiene (Required)
Tasks and subtasks:
- 9.0 Fix ray-cast coordinate space in `integration/shape_matching/mesh_profile_extractor.py`: ✅ Done
  - `mesh_obj.ray_cast` expects local-space coordinates; current code uses world-space.
  - Transform origin/direction into local space (or use `scene.ray_cast` with world space) to get correct hits for transformed objects.
  - Add a small test case where the mesh is translated/rotated and verify the extracted profile matches the untransformed case.
- 9.1 Fix visual-hull projection center in `integration/multi_view/visual_hull.py`: ✅ Done
  - Rotate points around the bounds center, not the world origin, before projecting to a silhouette.
  - Add a sanity test with non-centered bounds to verify projections stay within silhouette bounds.
- 9.2 Fix ray-cast usage in voxelization tests: ✅ Done
  - Update `test_ground_truth_iou.py`, `test_suite_multiview.py`, `test_phase2_integration.py`, `test_phase2_groundtruth_profiles.py`, and `test_debug_voxelization.py` to ray-cast in the correct coordinate space for scaled/rotated objects.
  - Prefer a shared helper that accepts world-space points and handles object transforms safely.
- 9.3 Fix multi-view test mesh construction and rendering isolation: ✅ Done
  - `test_suite_multiview.create_table` leaves leg objects in the scene; delete or join them after boolean apply to avoid rendering extra geometry.
  - Ensure render helpers only include the target object (hide or delete other objects) before rendering silhouettes.
  - Replace fixed `ortho_scale` in `render_turntable` / `render_silhouette_at_angle` / `generate_turntable_sequence.setup_camera` with bounds-based framing to avoid clipped silhouettes.
  - Avoid accumulating lights/cameras in per-view renders (reuse single light/camera or delete between renders); otherwise later views get over-lit silhouettes.
  - Force a truly unlit black material for silhouettes (emission or Principled with specular=0, roughness=1) and override any existing materials.
- 9.4 Guard invalid sampling counts in mesh profile extraction: ✅ Done
  - In `integration/shape_matching/mesh_profile_extractor.extract_multi_angle_profiles`, reject `num_angles <= 0` with a clear error.
  - In `extract_profile_at_angle`, reject `num_samples <= 0` and document minimum valid values.
  - Add tests that pass `num_angles=0` and `num_heights=0` and assert a clear `ValueError`.
- 9.4b Validate profile list integrity: ✅ Done
  - In `combine_profiles`, assert all profiles have the same length and raise a clear error if they do not.
  - In `visualize_profiles`, return early with a warning if `profiles` is empty (avoid division by zero).
- 9.4c Validate weighted profile extraction inputs: ✅ Done
  - In `weighted_profile_extractor.extract_weighted_profiles`, assert `num_angles == 12` (or `len(all_profiles) >= 12`) before indexing fixed angle slots.
  - If the profile count is insufficient, raise a clear error rather than indexing out of range.
- 9.5 Robust view selection in multi-view reconstruction helpers: ✅ Done
  - In `test_suite_multiview.reconstruct_multiview`, use the `num_views` argument when calling `load_multi_view_auto` (currently hard-coded to 12).
  - Guard `num_views <= 0` and `num_views > available_lateral_views`.
  - If `num_views` is invalid, raise a clear error instead of duplicating views with `step=0`.
- 9.6 Guard `num_views` in turntable render helpers: ✅ Done
  - In `generate_turntable_sequence.generate_turntable_sequence`, `test_suite_multiview.render_turntable`, and `test_phase2_integration.render_turntable_silhouettes`, validate `num_views >= 1` before computing `angle_step = 360.0 / num_views`.
- 9.7 Validate multi-view core inputs: ✅ Done
  - In `integration/multi_view/visual_hull.py`, validate `resolution >= 1` and `bounds_min < bounds_max` on all axes.
  - In `CameraView.__init__`, require a 2D silhouette array and raise a clear error if the input has 3 channels.
- 9.8 Guard empty voxel meshes in Phase 2 helpers: ✅ Done
  - In `test_phase2_step1_profiles.create_visual_hull_from_images` and `test_phase2_step2_sliceanalyzer.create_visual_hull_voxel_mesh`, handle the case where `occupied_indices` is empty (return an empty mesh and skip join).
- 9.9 Validate view-count inputs at the loader boundary: ✅ Done
  - In `integration/image_processing/image_loader.load_multi_view_auto`, if `num_views` is provided and `<= 0`, raise a clear error before attempting auto-detection.

## Phase 10 - Performance Benchmarks + Hotspot Optimization (Required)
Tasks and subtasks:
- 10.0 Add a lightweight benchmark tool for hotspots: ✅ Done
  - New script: `blender_blocking/benchmarks/benchmark_perf.py`.
  - Benchmarks: visual hull reconstruction, canonicalize mask, compare silhouettes, extract binary silhouette.
  - Outputs per-iter timing and optional JSON summary.
- 10.1 Capture baseline performance (manual run): ✅ Done
  - Run: `python blender_blocking/benchmarks/benchmark_perf.py --all --json benchmarks/results/pre_opt.json`.
  - Record machine info (CPU, RAM, Python version) in a short text note in `benchmarks/results/`.
  - Keep the JSON for before/after comparisons.
- 10.2 Vectorize Visual Hull intersection (major speedup): ✅ Done
  - Replace triple nested voxel loops in `integration/multi_view/visual_hull.py` with numpy vectorized projection.
  - Build voxel center grids once (or per chunk) and project per view using broadcasting.
  - Confirm boolean results match the baseline for small resolutions (32/48) via a deterministic test.
- 10.3 Add chunked evaluation to cap memory (scales to 256^3+): ✅ Done
  - Process the voxel grid in Z slabs (or 3D blocks) with a fixed chunk size.
  - Ensure chunked results are identical to non-chunked for the same inputs.
  - Add a config or argument to toggle chunking and set chunk size.
- 10.4 Early-exit and short-circuit rules: ✅ Done
  - If any view silhouette is empty, return an empty voxel grid with a warning.
  - If no views are provided, keep current error behavior.
  - Avoid per-view work when a view mask is all False.
- 10.5 Canonicalization caching for repeated comparisons: ✅ Done
  - Add a small LRU cache keyed by mask hash + params (output_size, padding, anchor) in `validation/silhouette_iou.py`.
  - Use cached canonical masks in `integration/shape_matching/shape_matcher.compare_silhouettes`.
  - Add tests that repeated compares return identical results and hit the cache.
- 10.6 Capture post-optimization performance (manual run): ✅ Done
  - Run: `python blender_blocking/benchmarks/benchmark_perf.py --all --json benchmarks/results/post_opt.json`.
  - Compare deltas and note speedups/regressions.
  - Results saved: `blender_blocking/benchmarks/results/post_opt.json` and `blender_blocking/benchmarks/results/post_opt_env.txt`.
  - Deltas (per-iter, 32^3, 8 views): visual hull ~895.29 ms -> ~6.88 ms; canonicalize ~0.1309 ms -> ~0.1142 ms; compare ~0.9901 ms -> ~0.8548 ms; extract silhouette ~0.0698 ms -> ~0.0740 ms.
- 10.7 Vectorize vertical width profile sampling in `geometry/dual_profile.extract_vertical_width_profile_px` (left/right per-row) and `interp_nearest` filling: ✅ Done
  - Precompute per-row left/right indices with argmax and sample rows via numpy indexing.
  - Replace nearest-fill loop with a vectorized `np.searchsorted` approach.
  - Tests: add interp-nearest fill case in `test_profile_sampling.py`.
  - Bench: add `vertical_width_profile` bench to `benchmark_perf.py`; run: `python blender_blocking/benchmarks/benchmark_perf.py --bench vertical_width_profile --iterations 200`.
- 10.8 Batch ResFit optimization gradients across primitives: ✅ Done
  - In `placement/resfitting.optimize_primitives`, compute batched SDF + gradients for all primitives in one pass (fewer Python loops).
  - Keep the per-primitive non-vectorized path for fallback parity.
  - Tests: add a quick test comparing vectorized vs non-vectorized one-step updates in `placement/test_resfitting_quick.py`.
  - Bench: add `resfit_optimize` bench to `benchmark_perf.py`; run: `python blender_blocking/benchmarks/benchmark_perf.py --bench resfit_optimize --resfit-opt-steps 5 --resfit-opt-iterations 1`.
- 10.9 Vectorize `shape_matching/slice_shape_matcher._slice_at_plane` edge intersection math: ✅ Done
  - Batch edge endpoints into numpy arrays and compute plane intersections in one pass.
  - Tests: add a Blender-only cube slice sanity test in `test_slice_shape_matcher.py`.
- 10.10 Reduce ray-cast loop overhead in `integration/shape_matching/mesh_profile_extractor`: ✅ Done
  - Precompute angle arrays (cos/sin) and height samples; reuse center/max-distance per height.
  - Precompute fallback ray directions per angle to reduce trig per-sample.
  - Tests: add a Blender-only extraction sanity test (reuse existing cylinder test or new small test file).

## Phase 11 - Blender-Adjacent Render Hygiene + Reuse (Planned)
Tasks and subtasks:
- 11.0 Centralize silhouette rendering helpers: ✅ Done
  - Add `integration/blender_ops/silhouette_render.py` with utilities for camera framing, emission-only material, world background, and target-only rendering.
  - Provide helpers for temporary camera/light creation and cleanup (context manager).
- 11.1 Refactor duplicated render logic to the helper: ✅ Done
  - Migrate `generate_turntable_sequence.py`, `test_suite_multiview.py`, `test_phase2_integration.py`, and `integration/blender_ops/render_utils.py` to use the shared helper.
  - Ensure consistent output settings (PNG, BW/RGBA, background).
- 11.2 Render isolation + cleanup: ✅ Done
  - Hide non-target meshes during silhouette renders; restore state after render.
  - Reuse a single camera/light across view loops to avoid accumulation.
- 11.3 Add Blender-only validation tests: ✅ Done
  - Add a small render regression test that asserts no extra objects appear in silhouettes.
  - Add a framing sanity test that verifies the silhouette fits fully in frame.

## Phase 12 - Content-Adaptive Patches (Option C) (Required)
Reference: `docs/content_adaptive_patches_spec.md` (full spec). Build this as a model-agnostic, inference-orchestration layer that can refine any per-pixel prediction.

### 12.0 Core config + data contracts
- Add `ContentAdaptivePatchesConfig` to `blender_blocking/config.py` and wire into `BlockingConfig` as `config.patches`.
  - Required fields (defaults from spec):
    - `enabled=False`
    - `patch_count=12`, `nms_radius_frac=0.15`
    - `patch_sizes=(0.25, 0.40, 0.60)` (fractions of `min(H,W)`)
    - `size_thresholds=(0.80, 0.60)` for small/medium/large choice
    - `margin_frac=0.15` (context margin), `min_overlap=0.20`
    - `feather_frac=0.12`
    - `w_edge=0.35`, `w_uncertainty=0.40`, `w_boundary=0.25`
    - `edge_upweight_alpha=1.0`
    - `uncertainty_mode=("tta_flip", "gradient")` (default: `tta_flip`)
    - `boundary_mode=("segmentation", "depth_grad", "edges")` (default: `depth_grad`)
    - `alignment_mode=("affine", "none")` (default: `affine` for depth)
    - `output_type=("depth", "normals", "segmentation", "logits")`
  - Add `validate()` checks for ranges and tuple sizes.
  - Update `scripts/generate_default_configs.py` to include patch config fields in all defaults.

### 12.1 Patch selection utilities (pure Python)
- New module: `blender_blocking/geometry/content_adaptive_patches.py`.
- Add dataclasses:
  - `PatchSpec`: `center`, `size_px`, `size_class`, `score`, `box_full`, `box_infer`, `margin_px`
  - `PatchTransform`: `box_full`, `box_infer`, `resize_scale`, `padding`, `model_input_size`
- Implement helpers:
  - `edge_strength_map(image)` (Sobel/Scharr magnitude, grayscale).
  - `boundary_map_from_seg(mask)` (morph gradient).
  - `boundary_map_from_depth(depth)` (gradient + optional NMS thinning).
  - `normalize_map_percentile(map, lo=5, hi=95)` → [0,1].
  - `patch_score_map(E,U,B, weights)` with robust normalization.
  - `select_patch_centers_nms(score_map, k, radius_px)`.
  - `assign_patch_sizes(score, size_thresholds, sizes_px)`.
  - `build_patch_boxes(center, size_px, margin_px, image_shape)` with bounds clamping.

### 12.2 Patch inference orchestration
- New module: `blender_blocking/integration/content_adaptive_patches.py`.
- Entry point:
  - `run_content_adaptive_patches(image, infer_fn, *, output_type, model_input_size, config, seg=None, global_pred=None) -> (P_final, U_final, patches, debug)`
  - `infer_fn` signature: `infer_fn(image_batch_or_single) -> np.ndarray` (model-agnostic).
- Pipeline steps (mirror spec):
  1. Global inference → `P_G_full` (resize → infer → resize back).
  2. Compute `E`, `U`, `B` maps at full-res.
  3. Build score map `S` and select patches via NMS.
  4. For each patch: crop + margin, resize to model input, run `infer_fn`, reproject to full-res using `PatchTransform`.
  5. Optional alignment (depth): affine `(a,b)` least-squares on overlap.
  6. Fusion: feather weights + edge upweight + uncertainty weight.
  7. Final `P_final` and `U_final` (variance across predictions).

### 12.3 Alignment + fusion details
- Add `align_affine_depth(patch, ref, mask)` (least squares; clamp `a` to sane range).
- Add `feather_weights(box_full, feather_frac)` (distance-to-edge smoothstep).
- Add `blend_patches(accum, wsum, patch, weights)` and finalize with `P_final = where(wsum>0, accum/wsum, P_G_full)`.
- Add `seam_residual_map` for debug output (optional).

### 12.4 Integration hooks
- Add a lightweight hook in `BlockingWorkflow.process_images` or `BlockingWorkflow.analyze_shapes`:
  - If `config.patches.enabled` and `self.patch_infer_fn` is provided, route through `run_content_adaptive_patches`.
  - If no `infer_fn`, log a warning and continue with legacy path.
- Add CLI flag plumbing in `test_e2e_validation.py` for `--patches-config` (or `--patches-enabled`) to exercise the pipeline with `infer_fn` stubs.

### 12.5 Debug artifacts
- Optional outputs (behind flag):
  - `score_map` heatmap
  - patch overlay image
  - `U_final` visualization
  - seam/residual map

### 12.6 Tests (pure Python)
- Add `test_content_adaptive_patches.py`:
  - `patch_score_map` normalization (0–1).
  - NMS center selection with min spacing.
  - Patch reprojection correctness (crop → resize → back).
  - Alignment recovers known `(a,b)` for depth.
  - Fusion weights stay within bounds and preserve `P_G_full` where `wsum=0`.

### 12.7 Benchmarks (optional but recommended)
- Add `bench_patch_selection` and `bench_patch_fusion` in `benchmark_perf.py`.
- Record a baseline for patch selection + fusion on a 512×512 image.

### 12.8 Blender usage (optional integration)
- Add a helper in `integration/blender_ops` to convert `U_final` into a vertex group or attribute.
- Use the uncertainty map to drive adaptive remesh/subdivision density in the patch-fusion output stage.
- Ensure boundary masks (`B`) are treated as hard constraints (avoid smoothing across gaps).

## Acceptance Criteria (Global)
- Legacy pipeline runs unchanged when `reconstruction_mode="legacy"`.
- Optional new pipeline works end-to-end when enabled and produces watertight loft mesh.
- IoU validation is resolution- and framing-invariant.
- Metadata and manifest exist for every run.
- Tests are deterministic and partitioned between pure-Python and Blender.

## Specs and Schemas to Follow (No Deviations)
- `docs/IMPLEMENTATION_SPEC.md` lines 1-407 (cite narrower line ranges when referencing specific sections).

## Appendix A - NumPy-Optimized Functions (Benchmarked)
- `integration/multi_view/visual_hull.MultiViewVisualHull.reconstruct` (bench: `visual_hull_reconstruct`)
- `integration/multi_view/visual_hull.MultiViewVisualHull.extract_surface_voxels` (bench: `surface_voxels`)
- `validation/silhouette_iou.canonicalize_mask` (bench: `canonicalize_mask`)
- `integration/shape_matching/profile_extractor.extract_vertical_profile` (bench: `vertical_profile`)
- `geometry/dual_profile.extract_vertical_width_profile_px` + `geometry/dual_profile._fill_missing` (bench: `vertical_width_profile`)
- `placement/primitive_placement.SliceAnalyzer._interpolate_profile` (bench: `profile_interpolation`)
- `integration/shape_matching/shape_matcher.compare_silhouettes` (bench: `compare_silhouettes`)
- `integration/shape_matching/mesh_profile_extractor.combine_profiles` (bench: `combine_profiles`)
- `shape_matching/slice_shape_matcher.SliceBasedShapeMatcher._normalize_features`, `_cosine_similarity`, `_correlation` (bench: `slice_metrics`)
- `placement/resfitting.ResidualFitter.compute_residual_error` (bench: `resfit_residual`)
- `placement/resfitting.ResidualFitter.run_full_optimization` (bench: `resfit_full`)
- `placement/resfitting.optimize_primitives` (bench: `resfit_optimize`)
- `geometry.silhouette.extract_binary_silhouette` (bench: `extract_silhouette`)

## Appendix B - Silhouette Intersection Debug Notes (In-Progress)
Purpose: track everything tried to fix the silhouette_intersection black/blocked renders.

Findings + attempts:
- Found all-black outputs for silhouette_intersection were fully transparent (alpha=0). Root cause: `silhouette_session` was being called with a boolean `ensure_light` keyword that shadowed the `ensure_light` function. Fix applied: `silhouette_session(..., ensure_light_obj=...)` and reject unexpected kwargs.
- Adjusted `render_orthogonal_views(force_material)` default to `False` to restore lit renders; added explicit `force_material=True` in `test_silhouette_rendering.py` to keep silhouette-only tests.
- Discovered silhouette_intersection front/top renders became solid rectangles (full bbox fill) even when input silhouettes are non-rectangular (car/dudeguy/bottle/vase). Verified via render alpha bbox area ratio = 1.0 for many outputs.
- Added `triangulate_object()` and centered extrusions to prevent empty/transparent booleans (car/star previously transparent). This produced non-transparent meshes but did not fix blocky silhouettes.
- Switched to mesh-level centering (`center_extrusion`) instead of object translation, and triangulate before extrusion to keep face topology stable prior to boolean.
- Updated config label parsing so `*-extreme-ultra` is preserved (no truncation to just `ultra`).
- Added silhouette_intersection config controls (extrude_distance, contour_mode, boolean_solver override, and silhouette_extract_override) plus render controls (force_material, colors, camera_distance_factor, party_mode) and wired them into config parsing.
- Switched silhouette_intersection contour extraction to use configurable contour modes + hierarchy, with optional hole subtraction using boolean DIFFERENCE, and consistent normalization via source_size for hole alignment.
- Added targeted debug logging for silhouette_intersection mesh creation (per-part verts/faces) and boolean results; warns when intersections yield empty meshes.
- Added mesh cleanup pass before booleans (remove doubles, dissolve degenerate, recalc normals) to address non-manifold inputs.
- Debugged intersection mesh bounds: example `dudeguy` intersection mesh bounds show negative Z and large extents, indicating transforms are still suspicious (`bounds [-4.58, -1.135, -7.405] to [~0, 1.135, 6.405]`).
- Verified input contours are not rectangular: contour area / bbox area ratios are < 1.0 (car ~0.75, dudeguy ~0.54, bottle ~0.67, vase ~0.57).
- Rendered “front-only” and “side-only” extrusions: front-only front view matches silhouette; side-only front view is near-rect. Intersection front view matches side-only front view → front silhouette not constraining intersection.

Still broken:
- FIXME: silhouette_intersection front/top renders are boxy/rectangular for many inputs.

Fixed since last note:
- Config label parsing now preserves `*-extreme-ultra` suffixes.

Next investigation steps (must do):
- TODO: Replace object-level `location.z -= extrude/2` with mesh-level centering to avoid translation + rotation coupling (add `center_extrusion()` and call before transforms).
- TODO: Apply transforms in a consistent order and log bounds of front/side meshes before boolean.
- TODO: Capture and compare boolean result stats (volume/face count) for front-only, side-only, and intersect to confirm intersection is not ignoring front mesh.
- TODO: Consider using contour hierarchy (holes) and avoid auto “largest_component_only” for silhouette_intersection; confirm `extract_binary_silhouette` options for this path.
- TODO: Verify `create_mesh_from_contours` is not inadvertently generating a rectangle (inspect vertex list + face count).
