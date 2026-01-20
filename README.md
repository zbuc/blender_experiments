# blender_experiments
Experiments with Blender Python

## Slice-based Shape Matching Algorithm

A sophisticated algorithm for comparing 3D meshes by analyzing their cross-sectional profiles.

### Features

- Slices 3D meshes at regular intervals along any axis (X, Y, or Z)
- Extracts and analyzes 2D cross-sectional profiles
- Computes multiple similarity metrics:
  - Area correlation
  - Shape descriptor matching
  - Feature vector cosine similarity
- Returns similarity scores from 0 (completely different) to 1 (identical)

### Usage

#### Quick Test with Selected Objects

```python
from slice_shape_matcher import test_slice_matcher

# Select 2 mesh objects in Blender, then run:
test_slice_matcher()
```

#### Programmatic Comparison

```python
from slice_shape_matcher import SliceBasedShapeMatcher
import bpy

# Get two mesh objects
obj1 = bpy.data.objects['Object1']
obj2 = bpy.data.objects['Object2']

# Create matcher with 30 slices
matcher = SliceBasedShapeMatcher(num_slices=30)

# Compare shapes
result = matcher.match_shapes(obj1, obj2, axis='Z')

print(f"Similarity: {result['similarity']:.3f}")
print(f"Quality: {result['match_quality']}")
```

#### Running the Demo

```python
# In Blender's Python console:
exec(open('/path/to/demo_slice_matcher.py').read())
```

The demo creates test shapes (cylinder, cone, cube, sphere) and compares them all.

### Algorithm Details

1. **Slicing**: Divides each mesh into parallel cross-sections
2. **Profile Extraction**: Captures intersection points at each slice plane
3. **Feature Calculation**: Computes area, perimeter, compactness, and complexity
4. **Comparison**: Uses weighted combination of multiple similarity metrics
5. **Scoring**: Returns normalized similarity score (0-1)

### Applications

- 3D model comparison and classification
- Shape recognition and retrieval
- Quality control in manufacturing
- Geometric analysis
- Automated mesh categorization
