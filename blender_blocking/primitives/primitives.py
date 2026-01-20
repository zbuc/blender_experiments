"""
Blender primitive shape library module.

This module provides functions to programmatically spawn basic primitives
(cube, sphere, cylinder, cone, torus) with configurable size, position, and rotation.
"""

import bpy
from typing import Tuple, Optional


def spawn_cube(
    size: float = 2.0,
    location: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    name: Optional[str] = None
) -> bpy.types.Object:
    """
    Spawn a cube primitive.

    Args:
        size: Size of the cube (default: 2.0)
        location: (x, y, z) position (default: origin)
        rotation: (x, y, z) rotation in radians (default: no rotation)
        name: Optional name for the object

    Returns:
        The created cube object
    """
    bpy.ops.mesh.primitive_cube_add(size=size, location=location, rotation=rotation)
    obj = bpy.context.active_object
    if name:
        obj.name = name
    return obj


def spawn_sphere(
    radius: float = 1.0,
    location: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    segments: int = 32,
    ring_count: int = 16,
    name: Optional[str] = None
) -> bpy.types.Object:
    """
    Spawn a UV sphere primitive.

    Args:
        radius: Radius of the sphere (default: 1.0)
        location: (x, y, z) position (default: origin)
        rotation: (x, y, z) rotation in radians (default: no rotation)
        segments: Number of segments (default: 32)
        ring_count: Number of rings (default: 16)
        name: Optional name for the object

    Returns:
        The created sphere object
    """
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        location=location,
        rotation=rotation,
        segments=segments,
        ring_count=ring_count
    )
    obj = bpy.context.active_object
    if name:
        obj.name = name
    return obj


def spawn_cylinder(
    radius: float = 1.0,
    depth: float = 2.0,
    location: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 32,
    name: Optional[str] = None
) -> bpy.types.Object:
    """
    Spawn a cylinder primitive.

    Args:
        radius: Radius of the cylinder (default: 1.0)
        depth: Height/depth of the cylinder (default: 2.0)
        location: (x, y, z) position (default: origin)
        rotation: (x, y, z) rotation in radians (default: no rotation)
        vertices: Number of vertices in the base (default: 32)
        name: Optional name for the object

    Returns:
        The created cylinder object
    """
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius,
        depth=depth,
        location=location,
        rotation=rotation,
        vertices=vertices
    )
    obj = bpy.context.active_object
    if name:
        obj.name = name
    return obj


def spawn_cone(
    radius1: float = 1.0,
    radius2: float = 0.0,
    depth: float = 2.0,
    location: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 32,
    name: Optional[str] = None
) -> bpy.types.Object:
    """
    Spawn a cone primitive.

    Args:
        radius1: Radius of the base (default: 1.0)
        radius2: Radius of the top (default: 0.0 for pointed cone)
        depth: Height/depth of the cone (default: 2.0)
        location: (x, y, z) position (default: origin)
        rotation: (x, y, z) rotation in radians (default: no rotation)
        vertices: Number of vertices in the base (default: 32)
        name: Optional name for the object

    Returns:
        The created cone object
    """
    bpy.ops.mesh.primitive_cone_add(
        radius1=radius1,
        radius2=radius2,
        depth=depth,
        location=location,
        rotation=rotation,
        vertices=vertices
    )
    obj = bpy.context.active_object
    if name:
        obj.name = name
    return obj


def spawn_torus(
    major_radius: float = 1.0,
    minor_radius: float = 0.25,
    location: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    major_segments: int = 48,
    minor_segments: int = 12,
    name: Optional[str] = None
) -> bpy.types.Object:
    """
    Spawn a torus primitive.

    Args:
        major_radius: Major radius of the torus (default: 1.0)
        minor_radius: Minor radius of the torus (default: 0.25)
        location: (x, y, z) position (default: origin)
        rotation: (x, y, z) rotation in radians (default: no rotation)
        major_segments: Number of segments for major radius (default: 48)
        minor_segments: Number of segments for minor radius (default: 12)
        name: Optional name for the object

    Returns:
        The created torus object
    """
    bpy.ops.mesh.primitive_torus_add(
        major_radius=major_radius,
        minor_radius=minor_radius,
        location=location,
        rotation=rotation,
        major_segments=major_segments,
        minor_segments=minor_segments
    )
    obj = bpy.context.active_object
    if name:
        obj.name = name
    return obj


def spawn_ellipsoid(
    rx: float = 1.0,
    ry: float = 1.0,
    rz: float = 1.0,
    location: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    segments: int = 32,
    ring_count: int = 16,
    name: Optional[str] = None
) -> bpy.types.Object:
    """
    Spawn an ellipsoid primitive with independent X, Y, Z radii.

    Creates a UV sphere and scales it non-uniformly to create an ellipsoid.
    This is ideal for organic shapes like vases, bottles, and other forms
    that need different dimensions along each axis.

    Args:
        rx: Radius along X axis (width) (default: 1.0)
        ry: Radius along Y axis (depth) (default: 1.0)
        rz: Radius along Z axis (height) (default: 1.0)
        location: (x, y, z) position (default: origin)
        rotation: (x, y, z) rotation in radians (default: no rotation)
        segments: Number of longitudinal segments (default: 32)
        ring_count: Number of latitudinal rings (default: 16)
        name: Optional name for the object

    Returns:
        The created ellipsoid object

    Example:
        # Create a tall, narrow ellipsoid (bottle-like shape)
        spawn_ellipsoid(rx=0.5, ry=0.5, rz=2.0, location=(0, 0, 1))

        # Create a wide, flat ellipsoid (disc-like shape)
        spawn_ellipsoid(rx=2.0, ry=2.0, rz=0.5, location=(0, 0, 0))
    """
    # Create a unit sphere at the location with the specified rotation
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=1.0,
        location=location,
        rotation=rotation,
        segments=segments,
        ring_count=ring_count
    )
    obj = bpy.context.active_object

    # Scale non-uniformly to create ellipsoid
    obj.scale = (rx, ry, rz)

    if name:
        obj.name = name

    return obj
