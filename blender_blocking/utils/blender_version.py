"""
Blender version detection and compatibility utilities.

Provides version-aware API compatibility for different Blender versions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    import bpy

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False


def get_blender_version() -> Optional[Tuple[int, int, int]]:
    """
    Get Blender version as tuple.

    Returns:
        Tuple (major, minor, patch) or None if Blender not available
    """
    if not BLENDER_AVAILABLE:
        return None
    return bpy.app.version


def get_blender_version_string() -> Optional[str]:
    """
    Get Blender version as string.

    Returns:
        Version string (e.g., "5.0.1") or None if Blender not available
    """
    if not BLENDER_AVAILABLE:
        return None
    return bpy.app.version_string


def get_eevee_engine_name() -> str:
    """
    Get the correct EEVEE engine name for the current Blender version.

    Returns:
        "BLENDER_EEVEE_NEXT" for Blender 4.2+, "BLENDER_EEVEE" for earlier versions
    """
    if not BLENDER_AVAILABLE:
        return "BLENDER_EEVEE"  # Default for non-Blender context

    version = bpy.app.version
    # Blender 4.2+ uses EEVEE_NEXT, earlier versions use EEVEE
    if version[0] >= 4 and version[1] >= 2:
        return "BLENDER_EEVEE_NEXT"
    return "BLENDER_EEVEE"


def is_blender_version_at_least(major: int, minor: int = 0, patch: int = 0) -> bool:
    """
    Check if Blender version is at least the specified version.

    Args:
        major: Major version number
        minor: Minor version number
        patch: Patch version number

    Returns:
        True if Blender version >= specified version, False otherwise
    """
    version = get_blender_version()
    if version is None:
        return False

    current_major, current_minor, current_patch = version
    required = (major, minor, patch)
    current = (current_major, current_minor, current_patch)

    return current >= required


def get_boolean_solver() -> str:
    """
    Get appropriate boolean solver enum for current Blender version.

    The boolean solver enum changed between Blender versions:
    - Blender 4.x: 'FAST' exists
    - Blender 5.0+: 'FAST' removed, use 'EXACT' or 'MANIFOLD'

    Returns:
        String: 'EXACT' for Blender 5.0+, 'FAST' for older versions
    """
    if not BLENDER_AVAILABLE:
        # Default to newer version enum when not in Blender
        return "EXACT"

    # Blender 5.0+ removed 'FAST' solver
    if is_blender_version_at_least(5, 0, 0):
        return "EXACT"
    else:
        return "FAST"


def resolve_boolean_solver(override: Optional[str]) -> str:
    """
    Resolve a boolean solver from an override, falling back to version-aware default.

    Args:
        override: Solver enum ("EXACT", "FLOAT", "MANIFOLD", "FAST") or "auto"/None

    Returns:
        Solver string that is valid for the current Blender version.
    """
    if override is None or override == "auto":
        return get_boolean_solver()
    if check_boolean_solver_compatibility(override):
        return override
    print(
        f"Warning: Boolean solver '{override}' not available; using recommended solver."
    )
    return get_boolean_solver()


def get_available_boolean_solvers() -> List[str]:
    """
    Get list of available boolean solver enums for current Blender version.

    Returns:
        List of available solver enum strings
    """
    if not BLENDER_AVAILABLE:
        # Return Blender 5.0+ enums as default
        return ["EXACT", "FLOAT", "MANIFOLD"]

    try:
        # Get enum items from Blender's BooleanModifier RNA
        solver_prop = bpy.types.BooleanModifier.bl_rna.properties["solver"]
        return [item.identifier for item in solver_prop.enum_items]
    except (AttributeError, KeyError):
        # Fallback to known enums
        if is_blender_version_at_least(5, 0, 0):
            return ["EXACT", "FLOAT", "MANIFOLD"]
        else:
            return ["FAST", "EXACT"]


def check_boolean_solver_compatibility(solver: str) -> bool:
    """
    Check if a boolean solver enum is available in current Blender version.

    Args:
        solver: Solver enum string to check

    Returns:
        True if solver is available, False otherwise
    """
    available = get_available_boolean_solvers()
    return solver in available


def get_version_info() -> Optional[Dict[str, Any]]:
    """
    Get comprehensive Blender version information.

    Returns:
        Dictionary with version details, or None if Blender not available
    """
    if not BLENDER_AVAILABLE:
        return None

    version = get_blender_version()
    return {
        "version": version,
        "version_string": get_blender_version_string(),
        "major": version[0],
        "minor": version[1],
        "patch": version[2],
        "is_5_or_later": is_blender_version_at_least(5, 0, 0),
        "recommended_boolean_solver": get_boolean_solver(),
        "available_boolean_solvers": get_available_boolean_solvers(),
    }


def print_version_info() -> None:
    """Print Blender version information to console."""
    info = get_version_info()

    if info is None:
        print("Blender not available")
        return

    print("Blender Version Information:")
    print(f"  Version: {info['version_string']} ({info['version']})")
    print(f"  Blender 5.0+: {info['is_5_or_later']}")
    print(f"  Recommended boolean solver: {info['recommended_boolean_solver']}")
    print(
        f"  Available boolean solvers: {', '.join(info['available_boolean_solvers'])}"
    )


if __name__ == "__main__":
    # When run directly in Blender, print version info
    print_version_info()
