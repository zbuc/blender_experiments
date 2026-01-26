from __future__ import annotations

import argparse
import json
from pathlib import Path


SILHOUETTE_DEFAULTS = {
    "prefer_alpha": True,
    "alpha_threshold": 127,
    "gray_threshold": None,
    "invert_policy": "auto",
    "morph_close_px": 0,
    "morph_open_px": 0,
    "fill_holes": True,
    "largest_component_only": True,
}

QUALITY_PRESETS = {
    "default": {
        "num_slices": 10,
        "profile_samples": 100,
        "smoothing_window": 3,
        "radial_segments": 24,
        "min_radius_u": 0.0,
        "merge_threshold_u": 0.0,
        "render_resolution": 512,
        "render_samples": 1,
        "margin_frac": 0.08,
    },
    "higher": {
        "num_slices": 60,
        "profile_samples": 160,
        "smoothing_window": 3,
        "radial_segments": 48,
        "min_radius_u": 0.001,
        "merge_threshold_u": 0.001,
        "render_resolution": 1024,
        "render_samples": 4,
        "margin_frac": 0.06,
    },
    "ultra": {
        "num_slices": 120,
        "profile_samples": 240,
        "smoothing_window": 3,
        "radial_segments": 64,
        "min_radius_u": 0.0005,
        "merge_threshold_u": 0.0005,
        "render_resolution": 1536,
        "render_samples": 8,
        "margin_frac": 0.05,
    },
    "extreme-ultra": {
        "num_slices": 200,
        "profile_samples": 320,
        "smoothing_window": 3,
        "radial_segments": 96,
        "min_radius_u": 0.0005,
        "merge_threshold_u": 0.0005,
        "render_resolution": 2048,
        "render_samples": 16,
        "margin_frac": 0.04,
    },
}

MODES = ("legacy", "loft_profile", "silhouette_intersection")


def build_config(mode: str, preset: dict) -> dict:
    return {
        "reconstruction": {
            "reconstruction_mode": mode,
            "unit_scale": 0.01,
            "num_slices": preset["num_slices"],
        },
        "mesh_join": {
            "mode": "boolean",
            "boolean_solver": "auto",
        },
        "silhouette_extract_ref": dict(SILHOUETTE_DEFAULTS),
        "silhouette_extract_render": dict(SILHOUETTE_DEFAULTS),
        "silhouette_intersection": {
            "extrude_distance": 1.0,
            "contour_mode": "external",
            "largest_component_only": None,
            "silhouette_extract_override": None,
            "boolean_solver": "auto",
        },
        "profile_sampling": {
            "num_samples": preset["profile_samples"],
            "sample_policy": "endpoints",
            "fill_strategy": "interp_linear",
            "smoothing_window": preset["smoothing_window"],
        },
        "mesh_from_profile": {
            "radial_segments": preset["radial_segments"],
            "cap_mode": "fan",
            "min_radius_u": preset["min_radius_u"],
            "merge_threshold_u": preset["merge_threshold_u"],
            "recalc_normals": True,
            "shade_smooth": True,
            "weld_degenerate_rings": True,
        },
        "render_silhouette": {
            "resolution": [
                preset["render_resolution"],
                preset["render_resolution"],
            ],
            "engine": "BLENDER_EEVEE",
            "transparent_bg": True,
            "samples": preset["render_samples"],
            "margin_frac": preset["margin_frac"],
            "color_mode": "RGBA",
            "force_material": False,
            "background_color": [1.0, 1.0, 1.0, 1.0],
            "silhouette_color": [0.0, 0.0, 0.0, 1.0],
            "camera_distance_factor": 2.0,
            "party_mode": False,
        },
        "canonicalize": {
            "output_size": 256,
            "padding_frac": 0.1,
            "anchor": "bottom_center",
            "interp": "nearest",
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate default config presets under configs/."
    )
    parser.add_argument(
        "--out-dir",
        default="configs",
        help="Output directory for preset JSON files",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip any config files that already exist",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for mode in MODES:
        for quality, preset in QUALITY_PRESETS.items():
            config = build_config(mode, preset)
            path = output_dir / f"{mode}-{quality}.json"
            if path.exists() and args.skip_existing:
                skipped += 1
                continue
            path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            written += 1

    print(
        f"Wrote {written} config files to {output_dir}"
        + (f" (skipped {skipped})" if skipped else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
