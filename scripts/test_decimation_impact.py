"""
Simple test to measure IoU impact of mesh decimation on loft_profile meshes.

Usage:
blender --background --python scripts/test_decimation_impact.py
"""

import sys
from pathlib import Path

# Add paths
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir / "blender_blocking"))
sys.path.insert(0, str(script_dir))

import os

user_packages = os.path.expanduser("~/blender_python_packages")
if os.path.exists(user_packages):
    sys.path.insert(0, user_packages)

import bpy
import numpy as np
import json
from PIL import Image
from blender_blocking.config import BlockingConfig
from blender_blocking.main_integration import BlockingWorkflow
from blender_blocking.geometry.silhouette import extract_binary_silhouette
from blender_blocking.validation.silhouette_iou import compute_mask_iou
from blender_blocking.integration.blender_ops.render_utils import render_orthogonal_views
import tempfile
import shutil
from typing import Any, Dict


def load_config(config_path: str) -> BlockingConfig:
    """Load BlockingConfig from JSON file."""
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config = BlockingConfig()

    # Apply reconstruction overrides
    if "reconstruction" in config_dict:
        recon = config_dict["reconstruction"]
        if "reconstruction_mode" in recon:
            config.reconstruction.reconstruction_mode = recon["reconstruction_mode"]
        if "num_slices" in recon:
            config.reconstruction.num_slices = int(recon["num_slices"])
        if "unit_scale" in recon:
            config.reconstruction.unit_scale = float(recon["unit_scale"])

    # Apply profile_sampling overrides
    if "profile_sampling" in config_dict:
        ps = config_dict["profile_sampling"]
        for key, value in ps.items():
            setattr(config.profile_sampling, key, value)

    # Apply mesh_from_profile overrides
    if "mesh_from_profile" in config_dict:
        mfp = config_dict["mesh_from_profile"]
        for key, value in mfp.items():
            setattr(config.mesh_from_profile, key, value)

    # Apply render_silhouette overrides
    if "render_silhouette" in config_dict:
        rs = config_dict["render_silhouette"]
        for key, value in rs.items():
            setattr(config.render_silhouette, key, value)

    # Apply silhouette_extract_ref overrides
    if "silhouette_extract_ref" in config_dict:
        ser = config_dict["silhouette_extract_ref"]
        for key, value in ser.items():
            setattr(config.silhouette_extract_ref, key, value)

    return config


def run_loft_workflow_with_decimation(
    config_path, front_img, side_img, decimate_ratio=None
):
    """Run loft workflow and optionally decimate the result."""
    print(f"\n{'='*70}")
    if decimate_ratio:
        print(f"TESTING WITH DECIMATION (ratio={decimate_ratio})")
    else:
        print("BASELINE (no decimation)")
    print(f"{'='*70}\n")

    # Reset scene
    bpy.ops.wm.read_homefile(use_empty=True)

    # Load config
    config = load_config(config_path)

    # Create workflow
    workflow = BlockingWorkflow(front_path=front_img, side_path=side_img, config=config)
    workflow.load_images()

    # Run loft pipeline
    final_mesh = workflow.create_3d_blockout_loft()

    if final_mesh is None:
        print("ERROR: Mesh creation failed")
        return None, {}

    poly_count_before = len(final_mesh.data.polygons)
    vert_count_before = len(final_mesh.data.vertices)

    print(f"  Initial mesh: {poly_count_before} polygons, {vert_count_before} vertices")

    # Apply decimation if requested
    if decimate_ratio is not None:
        print(f"  Applying decimate modifier (ratio={decimate_ratio})...")
        mod = final_mesh.modifiers.new(name="Decimate", type="DECIMATE")
        mod.decimate_type = "COLLAPSE"
        mod.ratio = decimate_ratio

        bpy.context.view_layer.objects.active = final_mesh
        bpy.ops.object.modifier_apply(modifier="Decimate")

        poly_count_after = len(final_mesh.data.polygons)
        vert_count_after = len(final_mesh.data.vertices)
        reduction_pct = 100 * (poly_count_before - poly_count_after) / poly_count_before

        print(
            f"  Decimated mesh: {poly_count_after} polygons, {vert_count_after} vertices"
        )
        print(f"  Polygon reduction: {reduction_pct:.1f}%")
    else:
        poly_count_after = poly_count_before
        vert_count_after = vert_count_before
        reduction_pct = 0.0

    metadata = {
        "poly_count": poly_count_after,
        "vert_count": vert_count_after,
        "poly_reduction_pct": reduction_pct,
        "decimated": decimate_ratio is not None,
        "decimate_ratio": decimate_ratio,
    }

    return final_mesh, metadata


def compute_iou_for_mesh(mesh, front_ref_img, side_ref_img, config):
    """Render mesh and compute IoU vs reference silhouettes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Render orthogonal views
        render_paths = render_orthogonal_views(
            output_dir=tmpdir,
            views=["front", "side"],
            target_objects=[mesh],
            resolution=config.render_silhouette.resolution,
            transparent_bg=config.render_silhouette.transparent_bg,
            color_mode=config.render_silhouette.color_mode,
            force_material=config.render_silhouette.force_material,
            background_color=config.render_silhouette.background_color,
            silhouette_color=config.render_silhouette.silhouette_color,
            camera_distance_factor=config.render_silhouette.camera_distance_factor,
            margin_frac=config.render_silhouette.margin_frac,
        )

        # Load rendered images
        front_render = np.array(Image.open(render_paths["front"]).convert("RGBA"))
        side_render = np.array(Image.open(render_paths["side"]).convert("RGBA"))

    # Resize reference images to match render resolution for fair comparison
    target_size = tuple(config.render_silhouette.resolution)
    front_ref_resized = np.array(
        Image.fromarray(front_ref_img).resize(
            (target_size[0], target_size[1]), Image.LANCZOS
        )
    )
    side_ref_resized = np.array(
        Image.fromarray(side_ref_img).resize(
            (target_size[0], target_size[1]), Image.LANCZOS
        )
    )

    # Extract silhouettes from both reference and renders
    front_ref_mask = extract_binary_silhouette(front_ref_resized)
    side_ref_mask = extract_binary_silhouette(side_ref_resized)
    front_render_mask = extract_binary_silhouette(front_render)
    side_render_mask = extract_binary_silhouette(side_render)

    # Compute IoU
    iou_result_front = compute_mask_iou(front_ref_mask, front_render_mask)
    iou_result_side = compute_mask_iou(side_ref_mask, side_render_mask)
    iou_front = iou_result_front.iou
    iou_side = iou_result_side.iou
    iou_avg = (iou_front + iou_side) / 2.0

    return {
        "iou_front": iou_front,
        "iou_side": iou_side,
        "iou_avg": iou_avg,
    }


def main():
    print(f"\n{'='*70}")
    print("MESH DECIMATION IoU IMPACT TEST")
    print(f"{'='*70}\n")

    # Paths
    config_path = str(
        Path(__file__).parent.parent / "configs" / "loft_profile-ultra.json"
    )
    front_img = str(
        Path(__file__).parent.parent / "blender_blocking" / "test_images" / "vase_front.png"
    )
    side_img = str(
        Path(__file__).parent.parent / "blender_blocking" / "test_images" / "vase_side.png"
    )

    # Load reference images
    front_ref = np.array(Image.open(front_img).convert("RGBA"))
    side_ref = np.array(Image.open(side_img).convert("RGBA"))

    config = load_config(config_path)

    results = {}

    # Test baseline (no decimation)
    mesh, meta = run_loft_workflow_with_decimation(
        config_path, front_img, side_img, decimate_ratio=None
    )
    if mesh:
        iou_results = compute_iou_for_mesh(mesh, front_ref, side_ref, config)
        results["baseline"] = {**meta, **iou_results}
        print(f"  Baseline IoU: front={iou_results['iou_front']:.4f}, side={iou_results['iou_side']:.4f}, avg={iou_results['iou_avg']:.4f}")

    # Test with different decimation ratios
    for ratio in [0.9, 0.75, 0.5, 0.25, 0.1]:
        mesh, meta = run_loft_workflow_with_decimation(
            config_path, front_img, side_img, decimate_ratio=ratio
        )
        if mesh:
            iou_results = compute_iou_for_mesh(
                mesh, front_ref, side_ref, config
            )
            results[f"decimate_{ratio}"] = {**meta, **iou_results}
            print(f"  Decimated ({ratio}) IoU: front={iou_results['iou_front']:.4f}, side={iou_results['iou_side']:.4f}, avg={iou_results['iou_avg']:.4f}")

    # Print summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    print(f"{'Config':<20} {'Avg IoU':<10} {'Δ IoU':<10} {'Polygons':<10} {'Reduction':<10}")
    print(f"{'-'*70}")

    baseline = results.get("baseline")
    if baseline:
        print(
            f"{'Baseline':<20} {baseline['iou_avg']:>9.4f} {'':<10} "
            f"{baseline['poly_count']:>9} {'':<10}"
        )

        for key in sorted([k for k in results.keys() if k != "baseline"]):
            result = results[key]
            ratio = key.replace("decimate_", "")
            delta = result["iou_avg"] - baseline["iou_avg"]
            print(
                f"{f'Decimate {ratio}':<20} {result['iou_avg']:>9.4f} {delta:>+9.4f} "
                f"{result['poly_count']:>9} {result['poly_reduction_pct']:>8.1f}%"
            )

    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}\n")

    if baseline:
        # Find best tradeoff (>10% reduction with <0.01 IoU loss)
        candidates = []
        for key in results.keys():
            if key == "baseline":
                continue
            result = results[key]
            delta = result["iou_avg"] - baseline["iou_avg"]
            if result["poly_reduction_pct"] > 10 and delta > -0.01:
                candidates.append((key, result))

        if candidates:
            best = max(candidates, key=lambda x: x[1]["poly_reduction_pct"])
            ratio = best[0].replace("decimate_", "")
            result = best[1]
            delta = result["iou_avg"] - baseline["iou_avg"]
            print(f"✓ RECOMMENDED: Decimate ratio={ratio}")
            print(f"  - Polygon reduction: {result['poly_reduction_pct']:.1f}%")
            print(f"  - IoU delta: {delta:+.4f} (minimal impact)")
            print(f"  - Final IoU: {result['iou_avg']:.4f}")
        else:
            print("✗ NO GOOD DECIMATION RATIO FOUND")
            print("  All tested ratios either:")
            print("  - Reduce polygons by <10%, OR")
            print("  - Lose >0.01 IoU")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
