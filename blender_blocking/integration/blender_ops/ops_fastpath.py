"""Optional fast-path helpers for bulk Blender ops.

WARNING: This uses a private Blender API and can break across Blender versions.
Only use this for known-safe bulk creation loops (e.g. adding primitives).
Do NOT use this around modifiers, boolean ops, rendering, or depsgraph-sensitive
operations.
"""

from __future__ import annotations

import contextlib
import warnings

try:
    import bpy
    from bpy.ops import _BPyOpsSubModOp

    BLENDER_AVAILABLE = True
except Exception:
    bpy = None
    _BPyOpsSubModOp = None
    BLENDER_AVAILABLE = False


@contextlib.contextmanager
def suppress_view_layer_updates(enabled: bool = False):
    """Temporarily suppress view layer updates for bpy.ops calls.

    Args:
        enabled: If False, does nothing.

    Yields:
        bool indicating whether suppression is active.

    WARNING: This relies on a private Blender API and can corrupt state if
    misused. Only wrap known-safe bulk creation loops, and keep it off by
    default.
    """
    if not enabled or not BLENDER_AVAILABLE or _BPyOpsSubModOp is None:
        yield False
        return

    warnings.warn(
        "View layer update suppression is enabled. This uses a private Blender "
        "API and is not safe for modifiers, boolean ops, or rendering.",
        RuntimeWarning,
    )

    view_layer_update = _BPyOpsSubModOp._view_layer_update

    def dummy_view_layer_update(context):
        return None

    try:
        _BPyOpsSubModOp._view_layer_update = dummy_view_layer_update
        yield True
    finally:
        _BPyOpsSubModOp._view_layer_update = view_layer_update
        if bpy is not None:
            try:
                bpy.context.view_layer.update()
            except Exception:
                pass
