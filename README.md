# blender_experiments
Experiments with Blender Python

## Primitives Module

The `primitives.py` module provides a simple API for programmatically spawning basic primitive shapes in Blender.

### Supported Primitives

- **Cube**: spawn_cube()
- **Sphere**: spawn_sphere()
- **Cylinder**: spawn_cylinder()
- **Cone**: spawn_cone()
- **Torus**: spawn_torus()

### Features

All primitive functions support:
- Configurable size/radius/depth
- Position control via location parameter (x, y, z)
- Rotation control via rotation parameter (x, y, z in radians)
- Optional naming
- Shape-specific parameters (segments, vertices, etc.)

### Usage

```python
from primitives import spawn_cube, spawn_sphere, spawn_cylinder, spawn_cone, spawn_torus
import math

# Create a cube
spawn_cube(size=2.0, location=(0, 0, 0), rotation=(0, 0, 0), name="MyCube")

# Create a sphere
spawn_sphere(radius=1.5, location=(4, 0, 0), segments=32, name="MySphere")

# Create a cylinder
spawn_cylinder(radius=1.0, depth=3.0, location=(-4, 0, 0), name="MyCylinder")

# Create a cone
spawn_cone(radius1=1.5, depth=2.5, location=(0, 4, 0), name="MyCone")

# Create a torus with rotation
spawn_torus(
    major_radius=1.5,
    minor_radius=0.5,
    location=(0, -4, 0),
    rotation=(math.radians(90), 0, 0),
    name="MyTorus"
)
```

### Running the Example

Run the example script in Blender:
```bash
blender --background --python example_primitives.py
```

Or open Blender and run `example_primitives.py` from the scripting tab.
