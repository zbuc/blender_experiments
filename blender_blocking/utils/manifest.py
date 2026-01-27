"""Manifest helpers for tagging objects and recording run metadata."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from utils.generation_context import GenerationContext, SCHEMA_VERSION


def _safe_json(value: Any) -> Any:
    """Ensure value is JSON-serializable; fallback to string."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def apply_object_tags(
    obj: Any,
    role: str,
    context: GenerationContext,
    index: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None,
) -> None:
    """Attach blocktool metadata tags to a Blender object."""
    if obj is None:
        return

    obj["blocktool_schema"] = SCHEMA_VERSION
    obj["blocktool_run_id"] = context.run_id
    if context.seed is not None:
        obj["blocktool_seed"] = int(context.seed)
    obj["blocktool_role"] = role
    if index is not None:
        obj["blocktool_index"] = int(index)
    if params is not None:
        obj["blocktool_params"] = _safe_json(params)


def build_manifest(
    context: GenerationContext,
    outputs: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    errors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the run manifest payload."""
    manifest = {
        "manifest_version": context.schema_version,
        "run_id": context.run_id,
        "created_utc": context.created_utc or datetime.now(timezone.utc).isoformat(),
        "context": context.to_dict(),
        "stages": [stage.to_dict() for stage in context.stages],
        "outputs": outputs or {},
        "warnings": warnings or [],
        "errors": errors or [],
    }
    return manifest


def write_manifest(scene: Any, manifest: Dict[str, Any]) -> None:
    """Write the manifest into the Blender scene custom properties."""
    if scene is None:
        raise ValueError("scene is required to write manifest")
    scene["blocktool_manifest"] = manifest
