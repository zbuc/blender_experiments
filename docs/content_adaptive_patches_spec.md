# Content-Adaptive Patches (Option C) — Markdown Specification
Version: 0.1  
Status: Draft (implementation-oriented)  
Owner: BLENDTEC pipeline  

## 1. Overview

This spec defines a **content-adaptive patch selection + multi-pass inference + fusion** subsystem that improves fine detail recovery (gaps, thin structures, occlusion boundaries) by:
- Running **one global (full-frame)** inference pass for scene coherence.
- Selecting **local patches** only where detail is likely (high edges / uncertainty / silhouettes).
- Running the **same model/process** on those patches.
- **Aligning** and **merging** patch predictions back into a refined global result.

The approach is consistent with the “global + patches + merge” pattern used in high-resolution monocular depth “boosting” pipelines.

---

## 2. Goals / Non-Goals

### Goals
- Improve recovery of **high-frequency geometry cues**: gaps, thin separations, crisp boundaries.
- Preserve **global consistency**: avoid local patches distorting overall structure.
- Provide **uncertainty signals** usable downstream (e.g., adaptive remesh/subdivision in Blender).
- Remain model-agnostic: works with depth, normals, segmentation, edge maps, or other per-pixel predictions.

### Non-Goals
- Training new models (this is inference-time orchestration).
- Guaranteeing perfect metric depth scale (monocular depth may remain scale-ambiguous).
- Solving all seam artifacts without overlap/blending (overlap is mandatory here).

---

## 3. Terms & Definitions

- **Global pass (G):** Inference on the full image (possibly resized to model-native resolution).
- **Patch (Pi):** A cropped region of the original image, typically resized to model input size.
- **Overlap:** Fractional shared area between adjacent patches and/or patch-to-global context.
- **Patch score map (S):** Per-pixel scalar map indicating “detail likelihood.”
- **Uncertainty (U):** Per-pixel measure of model instability or disagreement.
- **Silhouette / occlusion boundary (B):** Boundary cues indicating depth discontinuities or object edges likely to contain gaps.

---

## 4. Inputs / Outputs

### Inputs
- `I`: Input image (H×W×3).
- `F(·)`: Existing inference function (“the same process”) returning prediction `P`.
- `model_input_size`: e.g., (512×512) or whatever `F` expects.
- Optional:
  - `Seg`: segmentation logits or mask (if already produced by the pipeline).
  - `G_depth`: global depth (if the pipeline already computes depth first).
  - `camera intrinsics`: if available for downstream mesh projection (not required for patch selection itself).

### Outputs
- `P_final`: Refined prediction in full-resolution coordinates.
- `U_final`: Final uncertainty map in full-resolution coordinates.
- `patch_set`: Metadata (boxes, scores, transforms, alignment parameters).
- Debug artifacts (optional): `S`, overlays, residual maps.

---

## 5. High-Level Pipeline

1. **Global inference**
   - Compute `P_G = F(resize(I, model_input_size))`
   - Reproject/upsample `P_G` into full-res coordinates as `P_G_full`.

2. **Compute patch selection signals (full-res)**
   - Edge strength `E(x,y)` from `I` (Sobel / Scharr / Canny).
   - Global uncertainty `U(x,y)` (see §6).
   - Boundary likelihood `B(x,y)` (see §7).

3. **Compute patch score map**
   - `S = wE * norm(E) + wU * norm(U) + wB * norm(B)`
   - Optional gating: mask out sky/background/low-interest regions.

4. **Select patches**
   - Extract top-K patch centers using **non-maximum suppression** (NMS) over `S`.
   - Assign patch sizes (multi-scale) based on score scale and/or boundary thickness.

5. **Patch inference**
   - For each patch `Pi`, run `P_i = F(resize(crop(I, Pi), model_input_size))`
   - Reproject `P_i` back into full-res coordinates.

6. **Alignment (if needed)**
   - Align each `P_i` to `P_G_full` (or to a running fused map) in overlap regions.

7. **Fusion**
   - Blend aligned patches into a refined full-res prediction using feathered, edge-aware weights.
   - Produce `P_final` and `U_final`.

---

## 6. Uncertainty Map (U): Recommended Implementations

> Pick one based on what your pipeline already provides.

### 6.1. Test-Time Augmentation (TTA) disagreement (simple, effective)
Run the global inference twice (or more) with small perturbations and measure disagreement:
- Horizontal flip: `P1 = F(I)`, `P2 = unflip(F(flip(I)))`
- Scale jitter: run at two input scales
- Crop jitter: small shifts

Define:
- `U = |P1 - P2|` (for scalar outputs like depth)
- `U = 1 - dot(N1, N2)` (for normals)
- `U = entropy(softmax(logits))` or `1 - max_prob` (for segmentation)

### 6.2. Local gradient proxy (fast fallback)
If additional passes are too expensive:
- `U = ||∇P_G_full||` (high gradients often correlate with uncertain discontinuities)

### 6.3. Ensemble disagreement (if you have multiple models)
- `U = variance({P_k})` across models/checkpoints.

**Normalization:** Use robust scaling (median/MAD or percentile clipping) before combining into `S`.

---

## 7. Boundary Map (B): Silhouette / Occlusion Cues

### 7.1. From segmentation (preferred if available)
- If `Seg` is logits/masks, compute boundaries via morphological gradient or label edges:
  - `B = boundary(Seg_mask)` (1px–5px thickness)

### 7.2. From depth discontinuities (if depth exists)
- `B = clamp(||∇P_G_full||, 0, t)` for depth-like outputs
- Optionally apply directional thinning (non-maximum suppression) to emphasize true edges.

### 7.3. From image edges (fallback)
- Use strong edges that are **stable across scales**:
  - `B = E_strong ∩ E_scale_stable`

---

## 8. Patch Selection

### 8.1. Candidate scoring
Compute:
- `S = wE * E_norm + wU * U_norm + wB * B_norm`

Suggested defaults:
- `wE = 0.35`, `wU = 0.40`, `wB = 0.25`
- Clamp each term to `[0,1]` via percentile normalization (e.g., 5th–95th).

### 8.2. Non-Maximum Suppression (NMS) to pick patch centers
- Find local maxima in `S` with radius `r` (in pixels).
- Sort candidates by score; pick top-K while enforcing minimum separation `r`.

Suggested defaults:
- `K = 12` (typical), range 6–24
- `r = 0.15 * min(H, W)` for large images; or proportional to patch size

### 8.3. Patch sizing (multi-scale)
Define patch sizes as fractions of the image:
- Small: `s1 = 0.25 * min(H,W)`
- Medium: `s2 = 0.40 * min(H,W)`
- Large: `s3 = 0.60 * min(H,W)`

Assignment rule (example):
- If `S(center) > 0.8`: use small
- Else if `> 0.6`: medium
- Else: large

**Mandatory:** add **context margin** around each patch (for model receptive field):
- `margin = 0.10–0.20 * patch_size`  
Final crop = patch box expanded by margin and clamped to image bounds.

### 8.4. Overlap requirement
Ensure patches overlap sufficiently to allow alignment and seam blending:
- Recommended overlap: `20%` (minimum), `30%` (high-detail scenes)

---

## 9. Patch-to-Global Reprojection

Store a transform `T_i` that maps:
- full-res crop coordinates → model input coordinates → predicted output coordinates → full-res coordinates

Minimum metadata per patch:
- `box_full = (x0,y0,x1,y1)` in full-res pixels
- `box_infer = expanded crop used for inference`
- `resize_scale_x`, `resize_scale_y`
- `padding` (if letterboxed)
- `score`, `size_class`

---

## 10. Alignment (Output-Dependent)

Alignment reduces patch inconsistency versus global context.

### 10.1. Depth (scalar): affine scale + shift
Fit `(a_i, b_i)` such that in overlap region Ω:
- `a_i * D_i + b_i ≈ D_ref`

Implementation:
- Use least squares on valid pixels in Ω.
- Optionally robust (Huber / RANSAC) if outliers are common.

Where `D_ref` is either:
- `P_G_full` (global anchor), or
- current fused map (iterative refinement)

### 10.2. Normals (unit vectors)
- Ensure vectors are normalized after reprojection.
- Optionally, blend without alignment; alignment is usually unnecessary unless the model has strong local bias.

### 10.3. Segmentation (logits/probabilities)
- No alignment needed; fuse logits/probabilities directly with weights.
- Consider temperature scaling only if calibration issues arise.

---

## 11. Fusion / Blending

### 11.1. Weighting per patch
Define per-pixel weight within a patch:
- `w_spatial`: feather weight increasing toward patch center  
  Example: `w_spatial = smoothstep(dist_to_edge / feather_width)`
- `w_edge`: upweight areas near boundaries/edges  
  Example: `w_edge = 1 + α * B(x,y)` (α ~ 0.5–1.5)
- `w_conf`: optional inverse uncertainty of the local prediction  
  Example: `w_conf = 1 / (ε + U_local)`

Combine:
- `w_i = w_spatial * w_edge * w_conf`

### 11.2. Accumulation
Initialize:
- `accum = 0`, `wsum = 0`

For each aligned patch prediction `P_i_full`:
- `accum += w_i * P_i_full`
- `wsum += w_i`

Finalize:
- `P_patch = accum / max(wsum, ε)`
- `P_final = blend(P_G_full, P_patch, mask=(wsum > 0))`  
  Common: `P_final = where(wsum>0, P_patch, P_G_full)` or soft blend using `wsum`.

### 11.3. Seam diagnostics
Compute a seam/residual map:
- `R = |P_i_full - P_ref|` in overlap
Large persistent seams indicate:
- insufficient overlap
- poor alignment (depth)
- patch sizing too small (context loss)

---

## 12. Final Uncertainty (U_final)

Recommended:
- `U_final = combine(U_global, U_patch_disagreement)`

Where:
- `U_patch_disagreement = variance({P_k(x,y)})` over all contributing predictions at pixel (including global)
- If only one patch contributes: fall back to `U_global`

Use cases:
- Blender: drive adaptive subdivision/remesh density and/or displacement strength.

---

## 13. Reference Pseudocode (Implementation Skeleton)

```pseudo
P_G_full = infer_global(I)

E = edge_strength(I)
U = global_uncertainty(I, P_G_full)     # TTA or proxy
B = boundary_likelihood(I, P_G_full, Seg)

S = normalize(wE*E + wU*U + wB*B)

patches = select_patches_by_NMS(S, K, sizes=[s1,s2,s3], overlap=0.2)

accum, wsum = zeros_like(P_G_full), zeros_like(P_G_full)

for Pi in patches:
    I_crop, Ti = crop_with_margin(I, Pi)
    P_i = infer(resize(I_crop))
    P_i_full = reproject(P_i, Ti)

    if output_type == DEPTH:
        (a,b) = align_affine(P_i_full, P_G_full, overlap_region(Pi))
        P_i_full = a*P_i_full + b

    w_i = feather_weight(Pi) * (1 + alpha*B) * inv_uncertainty_weight(P_i_full)
    accum += w_i * P_i_full
    wsum += w_i

P_patch = accum / max(wsum, eps)
P_final = where(wsum > 0, P_patch, P_G_full)

U_final = compute_final_uncertainty(P_G_full, patches, P_final)
return P_final, U_final, patches
```

---

## 14. Blender Integration Notes (Blocking Tool)

### 14.1. Using U_final for adaptive geometry
- Convert `U_final` to a vertex group / attribute:
  - High uncertainty → higher subdivision, more aggressive remesh resolution
  - Low uncertainty → preserve coarse topology for stability/performance

### 14.2. Preserving “gaps”
- Treat high `B` regions as **hard constraints**:
  - Avoid smoothing/closing operations across boundary masks
  - Consider boolean cuts / alpha shapes guided by boundary map where appropriate

### 14.3. Debug overlays
Export images to inspect:
- `S` heatmap
- Patch boxes overlaid on `I`
- `U_final`
- Seam/residual maps

---

## 15. Parameter Defaults (Recommended Starting Point)

- `K = 12` patches
- Patch sizes: `{0.25, 0.40, 0.60} * min(H,W)`
- Margin: `0.15 * patch_size`
- Overlap: `0.20` minimum
- Feather width: `0.12 * patch_size`
- Score weights: `wE=0.35, wU=0.40, wB=0.25`
- Edge upweight: `α = 1.0`
- Alignment: affine `(a,b)` for depth only

---

## 16. Validation Checklist

- **Detail gain:** small gaps and thin features appear more distinctly than baseline.
- **Global coherence:** no obvious “scale pop” across patches (depth).
- **Seams:** overlap + feathering removes visible patch borders.
- **Runtime:** patch count and TTA kept within budget.
- **Stability:** repeat runs produce similar results (uncertainty decreases where model is stable).

---

## 17. References
- High-resolution depth boosting / patch merge pattern (CVPR 2021):  
  Yaksoy et al., “Boosting Monocular Depth Estimation Models to High-Resolution via Content-Adaptive Multi-Resolution Merging.”  
  <https://yaksoy.github.io/papers/CVPR21-HighResDepth.pdf>
