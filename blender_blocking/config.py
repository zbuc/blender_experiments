"""Configuration models for the Blender automated blocking tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

try:
    # Try relative import first (for package imports)
    from .utils.blender_version import get_eevee_engine_name
except ImportError:
    # Fall back to direct import (for direct script execution)
    from utils.blender_version import get_eevee_engine_name


_VALID_RECON_MODES = {"legacy", "loft_profile", "silhouette_intersection"}
_VALID_JOIN_MODES = {"auto", "boolean", "voxel", "simple"}
_VALID_CAP_MODES = {"fan", "none", "ngon"}
_VALID_SAMPLE_POLICIES = {"endpoints", "cell_centers"}
_VALID_FILL_STRATEGIES = {"interp_linear", "interp_nearest", "constant"}
_VALID_CANON_ANCHORS = {"center", "bottom_center"}
_VALID_CANON_INTERP = {"nearest"}
_VALID_RENDER_ENGINES = {"BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "WORKBENCH"}
_VALID_COLOR_MODES = {"BW", "RGBA"}
_VALID_CONTOUR_MODES = {"external", "ccomp", "tree", "hierarchy"}
_VALID_BOOLEAN_SOLVERS = {"auto", "EXACT", "MANIFOLD", "FLOAT", "FAST"}


@dataclass
class SilhouetteExtractConfig:
    """Configuration for silhouette extraction from images."""

    prefer_alpha: bool = True
    alpha_threshold: int = 127
    gray_threshold: Optional[int] = None
    invert_policy: str = "auto"
    morph_close_px: int = 0
    morph_open_px: int = 0
    fill_holes: bool = True
    largest_component_only: bool = True

    def validate(self) -> None:
        """Validate configuration values."""
        if not (0 <= self.alpha_threshold <= 255):
            raise ValueError("alpha_threshold must be in [0, 255]")
        if self.gray_threshold is not None and not (0 <= self.gray_threshold <= 255):
            raise ValueError("gray_threshold must be in [0, 255] when provided")
        if self.morph_close_px < 0 or self.morph_open_px < 0:
            raise ValueError("morph_close_px and morph_open_px must be >= 0")

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict."""
        return {
            "prefer_alpha": self.prefer_alpha,
            "alpha_threshold": self.alpha_threshold,
            "gray_threshold": self.gray_threshold,
            "invert_policy": self.invert_policy,
            "morph_close_px": self.morph_close_px,
            "morph_open_px": self.morph_open_px,
            "fill_holes": self.fill_holes,
            "largest_component_only": self.largest_component_only,
        }


@dataclass
class ProfileSamplingConfig:
    """Configuration for sampling silhouettes into profiles."""

    num_samples: int = 100
    sample_policy: str = "endpoints"
    fill_strategy: str = "interp_linear"
    smoothing_window: int = 3

    def validate(self) -> None:
        """Validate configuration values."""
        if self.num_samples < 2:
            raise ValueError("num_samples must be >= 2")
        if self.sample_policy not in _VALID_SAMPLE_POLICIES:
            raise ValueError(f"sample_policy must be one of {_VALID_SAMPLE_POLICIES}")
        if self.fill_strategy not in _VALID_FILL_STRATEGIES:
            raise ValueError(f"fill_strategy must be one of {_VALID_FILL_STRATEGIES}")
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be >= 1")

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict."""
        return {
            "num_samples": self.num_samples,
            "sample_policy": self.sample_policy,
            "fill_strategy": self.fill_strategy,
            "smoothing_window": self.smoothing_window,
        }


@dataclass
class LoftMeshOptions:
    """Configuration for loft mesh generation."""

    radial_segments: int = 24
    cap_mode: str = "fan"
    min_radius_u: float = 0.0
    merge_threshold_u: float = 0.0
    recalc_normals: bool = True
    shade_smooth: bool = True
    weld_degenerate_rings: bool = True
    apply_decimation: bool = True
    decimate_ratio: float = 0.1
    decimate_method: str = "COLLAPSE"

    def validate(self) -> None:
        """Validate configuration values."""
        if self.radial_segments < 3:
            raise ValueError("radial_segments must be >= 3")
        if self.cap_mode not in _VALID_CAP_MODES:
            raise ValueError(f"cap_mode must be one of {_VALID_CAP_MODES}")
        if self.min_radius_u < 0 or self.merge_threshold_u < 0:
            raise ValueError("min_radius_u and merge_threshold_u must be >= 0")
        if self.decimate_ratio < 0 or self.decimate_ratio > 1:
            raise ValueError("decimate_ratio must be between 0 and 1")
        if self.decimate_method not in ("COLLAPSE", "UNSUBDIV", "DISSOLVE"):
            raise ValueError("decimate_method must be COLLAPSE, UNSUBDIV, or DISSOLVE")

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict."""
        return {
            "radial_segments": self.radial_segments,
            "cap_mode": self.cap_mode,
            "min_radius_u": self.min_radius_u,
            "merge_threshold_u": self.merge_threshold_u,
            "recalc_normals": self.recalc_normals,
            "shade_smooth": self.shade_smooth,
            "weld_degenerate_rings": self.weld_degenerate_rings,
            "apply_decimation": self.apply_decimation,
            "decimate_ratio": self.decimate_ratio,
            "decimate_method": self.decimate_method,
        }


@dataclass
class RenderConfig:
    """Configuration for rendering orthographic silhouettes."""

    resolution: Tuple[int, int] = (512, 512)
    engine: str = field(default_factory=get_eevee_engine_name)
    transparent_bg: bool = True
    samples: int = 1
    margin_frac: float = 0.08
    color_mode: str = "RGBA"
    force_material: bool = False
    background_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    silhouette_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    camera_distance_factor: float = 2.0
    party_mode: bool = False

    def validate(self) -> None:
        """Validate configuration values."""
        if len(self.resolution) != 2:
            raise ValueError("resolution must be a (width, height) tuple")
        if any(val < 32 for val in self.resolution):
            raise ValueError("resolution values must be >= 32")
        if self.engine not in _VALID_RENDER_ENGINES:
            raise ValueError(f"engine must be one of {_VALID_RENDER_ENGINES}")
        if self.samples < 1:
            raise ValueError("samples must be >= 1")
        if not (0.0 <= self.margin_frac <= 1.0):
            raise ValueError("margin_frac must be in [0, 1]")
        if self.color_mode not in _VALID_COLOR_MODES:
            raise ValueError(f"color_mode must be one of {_VALID_COLOR_MODES}")
        if len(self.background_color) != 4:
            raise ValueError("background_color must be RGBA with 4 values")
        if len(self.silhouette_color) != 4:
            raise ValueError("silhouette_color must be RGBA with 4 values")
        if any(not (0.0 <= val <= 1.0) for val in self.background_color):
            raise ValueError("background_color values must be in [0, 1]")
        if any(not (0.0 <= val <= 1.0) for val in self.silhouette_color):
            raise ValueError("silhouette_color values must be in [0, 1]")
        if self.camera_distance_factor <= 0:
            raise ValueError("camera_distance_factor must be > 0")

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict."""
        return {
            "resolution": list(self.resolution),
            "engine": self.engine,
            "transparent_bg": self.transparent_bg,
            "samples": self.samples,
            "margin_frac": self.margin_frac,
            "color_mode": self.color_mode,
            "force_material": self.force_material,
            "background_color": list(self.background_color),
            "silhouette_color": list(self.silhouette_color),
            "camera_distance_factor": self.camera_distance_factor,
            "party_mode": self.party_mode,
        }


@dataclass
class CanonicalizeConfig:
    """Configuration for canonicalizing silhouette masks."""

    output_size: int = 256
    padding_frac: float = 0.1
    anchor: str = "bottom_center"
    interp: str = "nearest"

    def validate(self) -> None:
        """Validate configuration values."""
        if self.output_size < 32:
            raise ValueError("output_size must be >= 32")
        if not (0.0 <= self.padding_frac <= 1.0):
            raise ValueError("padding_frac must be in [0, 1]")
        if self.anchor not in _VALID_CANON_ANCHORS:
            raise ValueError(f"anchor must be one of {_VALID_CANON_ANCHORS}")
        if self.interp not in _VALID_CANON_INTERP:
            raise ValueError(f"interp must be one of {_VALID_CANON_INTERP}")

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict."""
        return {
            "output_size": self.output_size,
            "padding_frac": self.padding_frac,
            "anchor": self.anchor,
            "interp": self.interp,
        }


@dataclass
class MeshJoinConfig:
    """Configuration for mesh join behavior."""

    mode: str = "boolean"
    boolean_solver: str = "auto"

    def validate(self) -> None:
        """Validate configuration values."""
        if self.mode not in _VALID_JOIN_MODES:
            raise ValueError(f"mode must be one of {_VALID_JOIN_MODES}")
        if self.boolean_solver not in _VALID_BOOLEAN_SOLVERS:
            raise ValueError(f"boolean_solver must be one of {_VALID_BOOLEAN_SOLVERS}")

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict."""
        return {
            "mode": self.mode,
            "boolean_solver": self.boolean_solver,
        }


@dataclass
class SilhouetteIntersectionConfig:
    """Configuration for silhouette intersection reconstruction."""

    extrude_distance: float = 1.0
    contour_mode: str = "external"
    largest_component_only: Optional[bool] = None
    silhouette_extract_override: Optional[Dict[str, object]] = None
    boolean_solver: str = "auto"

    def validate(self) -> None:
        """Validate configuration values."""
        if self.extrude_distance <= 0:
            raise ValueError("extrude_distance must be > 0")
        if self.contour_mode not in _VALID_CONTOUR_MODES:
            raise ValueError(f"contour_mode must be one of {_VALID_CONTOUR_MODES}")
        if self.boolean_solver not in _VALID_BOOLEAN_SOLVERS:
            raise ValueError(f"boolean_solver must be one of {_VALID_BOOLEAN_SOLVERS}")
        if self.silhouette_extract_override is not None:
            if not isinstance(self.silhouette_extract_override, dict):
                raise ValueError("silhouette_extract_override must be a dict or None")
            valid_keys = set(SilhouetteExtractConfig().to_dict().keys())
            invalid = set(self.silhouette_extract_override.keys()) - valid_keys
            if invalid:
                raise ValueError(
                    f"silhouette_extract_override has invalid keys: {sorted(invalid)}"
                )

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict."""
        return {
            "extrude_distance": self.extrude_distance,
            "contour_mode": self.contour_mode,
            "largest_component_only": self.largest_component_only,
            "silhouette_extract_override": self.silhouette_extract_override,
            "boolean_solver": self.boolean_solver,
        }


@dataclass
class ReconstructionConfig:
    """Top-level reconstruction settings."""

    reconstruction_mode: str = "legacy"
    unit_scale: float = 0.01
    num_slices: int = 10

    def validate(self) -> None:
        """Validate configuration values."""
        if self.reconstruction_mode not in _VALID_RECON_MODES:
            raise ValueError(f"reconstruction_mode must be one of {_VALID_RECON_MODES}")
        if self.unit_scale < 0:
            raise ValueError("unit_scale must be >= 0")
        if self.num_slices < 1:
            raise ValueError("num_slices must be >= 1")

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict."""
        return {
            "reconstruction_mode": self.reconstruction_mode,
            "unit_scale": self.unit_scale,
            "num_slices": self.num_slices,
        }


@dataclass
class BlockingConfig:
    """Root configuration for the blocking workflow."""

    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    mesh_join: MeshJoinConfig = field(default_factory=MeshJoinConfig)
    silhouette_extract_ref: SilhouetteExtractConfig = field(
        default_factory=SilhouetteExtractConfig
    )
    silhouette_extract_render: SilhouetteExtractConfig = field(
        default_factory=SilhouetteExtractConfig
    )
    silhouette_intersection: SilhouetteIntersectionConfig = field(
        default_factory=SilhouetteIntersectionConfig
    )
    profile_sampling: ProfileSamplingConfig = field(
        default_factory=ProfileSamplingConfig
    )
    mesh_from_profile: LoftMeshOptions = field(default_factory=LoftMeshOptions)
    render_silhouette: RenderConfig = field(default_factory=RenderConfig)
    canonicalize: CanonicalizeConfig = field(default_factory=CanonicalizeConfig)

    def validate(self) -> None:
        """Validate configuration values across groups."""
        self.reconstruction.validate()
        self.mesh_join.validate()
        self.silhouette_extract_ref.validate()
        self.silhouette_extract_render.validate()
        self.silhouette_intersection.validate()
        self.profile_sampling.validate()
        self.mesh_from_profile.validate()
        self.render_silhouette.validate()
        self.canonicalize.validate()

        # Placeholder for mutually exclusive scale policies if added later.

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict matching the canonical schema."""
        return {
            "reconstruction": self.reconstruction.to_dict(),
            "mesh_join": self.mesh_join.to_dict(),
            "silhouette_extract_ref": self.silhouette_extract_ref.to_dict(),
            "silhouette_extract_render": self.silhouette_extract_render.to_dict(),
            "silhouette_intersection": self.silhouette_intersection.to_dict(),
            "profile_sampling": self.profile_sampling.to_dict(),
            "mesh_from_profile": self.mesh_from_profile.to_dict(),
            "canonicalize": self.canonicalize.to_dict(),
            "render_silhouette": self.render_silhouette.to_dict(),
        }
