# blender_experiments
Experiments with Blender Python

## 3D Primitive Placement and Mesh Joining

This project demonstrates advanced Blender Python techniques for:
- Analyzing 3D volumes using slice-based analysis
- Positioning and scaling primitives based on slice data
- Joining multiple meshes using boolean union operations
- Creating base meshes suitable for sculpting

### Features

- **SliceAnalyzer**: Analyzes 2D slices through a 3D volume to determine primitive placement
- **PrimitivePlacer**: Creates and positions primitives (cubes, spheres, cylinders, cones)
- **MeshJoiner**: Joins meshes using boolean union operations

### Usage

#### In Blender GUI

1. Open Blender
2. Go to Scripting workspace
3. Open `primitive_placement.py` or `example_simple.py`
4. Click "Run Script" or press Alt+P

#### Command Line

```bash
blender --python primitive_placement.py
blender --python example_simple.py
```

#### Python API

```python
from primitive_placement import SliceAnalyzer, PrimitivePlacer, MeshJoiner
from mathutils import Vector

# Define volume bounds
bounds_min = Vector((-2, -2, 0))
bounds_max = Vector((2, 2, 6))

# Analyze slices
analyzer = SliceAnalyzer(bounds_min, bounds_max, num_slices=12)
slice_data = analyzer.get_all_slice_data()

# Place primitives
placer = PrimitivePlacer()
objects = placer.place_primitives_from_slices(slice_data, primitive_type='CYLINDER')

# Join with boolean union
joiner = MeshJoiner()
final_mesh = joiner.join_with_boolean_union(objects, target_name="Sculpt_Base")
```

### Files

- `primitive_placement.py` - Main module with core functionality
- `example_simple.py` - Simple example creating an organic form

### Requirements

- Blender 3.0+ (works with Blender's built-in Python)
