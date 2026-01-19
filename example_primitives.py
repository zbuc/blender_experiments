"""
Example usage of the primitives module.

This script demonstrates how to use the primitives module to spawn
various shapes in Blender with different configurations.

Usage:
    Run this script inside Blender's scripting environment or via:
    blender --background --python example_primitives.py
"""

import math
from primitives import spawn_cube, spawn_sphere, spawn_cylinder, spawn_cone, spawn_torus


def create_primitive_showcase():
    """Create a showcase of all primitive shapes with different configurations."""

    # Spawn a cube at the origin
    spawn_cube(
        size=2.0,
        location=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
        name="MyCube"
    )

    # Spawn a sphere offset to the right
    spawn_sphere(
        radius=1.5,
        location=(4.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
        segments=32,
        ring_count=16,
        name="MySphere"
    )

    # Spawn a cylinder offset to the left
    spawn_cylinder(
        radius=1.0,
        depth=3.0,
        location=(-4.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
        vertices=32,
        name="MyCylinder"
    )

    # Spawn a cone in front, rotated
    spawn_cone(
        radius1=1.5,
        radius2=0.0,
        depth=2.5,
        location=(0.0, 4.0, 0.0),
        rotation=(0.0, 0.0, math.radians(45)),
        vertices=32,
        name="MyCone"
    )

    # Spawn a torus behind
    spawn_torus(
        major_radius=1.5,
        minor_radius=0.5,
        location=(0.0, -4.0, 0.0),
        rotation=(math.radians(90), 0.0, 0.0),
        major_segments=48,
        minor_segments=12,
        name="MyTorus"
    )

    print("Created 5 primitive shapes: cube, sphere, cylinder, cone, and torus")


if __name__ == "__main__":
    create_primitive_showcase()
