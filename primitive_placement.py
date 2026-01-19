"""
Blender script for 3D primitive placement and mesh joining.
Positions and scales primitives based on slice analysis and joins them using boolean unions.
"""

import bpy
import bmesh
from mathutils import Vector
import math


class SliceAnalyzer:
    """Analyzes 2D slices to determine primitive placement parameters."""

    def __init__(self, bounds_min, bounds_max, num_slices=10):
        """
        Initialize slice analyzer.

        Args:
            bounds_min: Vector representing minimum bounds (x, y, z)
            bounds_max: Vector representing maximum bounds (x, y, z)
            num_slices: Number of slices to analyze along Z axis
        """
        self.bounds_min = Vector(bounds_min)
        self.bounds_max = Vector(bounds_max)
        self.num_slices = num_slices

    def analyze_slice(self, z_position):
        """
        Analyze a single horizontal slice at given z position.

        Args:
            z_position: Z coordinate of the slice

        Returns:
            dict with 'center', 'radius', and 'scale' for primitive placement
        """
        # Calculate relative position in the volume (0 to 1)
        z_range = self.bounds_max.z - self.bounds_min.z
        z_normalized = (z_position - self.bounds_min.z) / z_range if z_range > 0 else 0.5

        # Example: create a tapered profile (wider in middle, narrower at ends)
        # Using a sine wave to create interesting variation
        profile_factor = math.sin(z_normalized * math.pi)

        # Calculate center position for this slice
        center = Vector((
            (self.bounds_min.x + self.bounds_max.x) / 2,
            (self.bounds_min.y + self.bounds_max.y) / 2,
            z_position
        ))

        # Calculate radius based on profile
        max_radius = min(
            (self.bounds_max.x - self.bounds_min.x) / 2,
            (self.bounds_max.y - self.bounds_min.y) / 2
        )
        radius = max_radius * profile_factor * 0.8

        # Scale factor for primitive
        scale = Vector((
            radius,
            radius,
            z_range / self.num_slices
        ))

        return {
            'center': center,
            'radius': radius,
            'scale': scale,
            'profile_factor': profile_factor
        }

    def get_all_slice_data(self):
        """
        Analyze all slices and return placement data.

        Returns:
            List of slice analysis results
        """
        slice_data = []
        z_step = (self.bounds_max.z - self.bounds_min.z) / (self.num_slices - 1)

        for i in range(self.num_slices):
            z_pos = self.bounds_min.z + i * z_step
            slice_data.append(self.analyze_slice(z_pos))

        return slice_data


class PrimitivePlacer:
    """Places and manages 3D primitives based on slice analysis."""

    def __init__(self):
        self.placed_objects = []

    def create_primitive(self, primitive_type='CUBE', location=(0, 0, 0), scale=(1, 1, 1)):
        """
        Create a primitive object.

        Args:
            primitive_type: Type of primitive ('CUBE', 'SPHERE', 'CYLINDER', 'CONE')
            location: World location for the primitive
            scale: Scale vector for the primitive

        Returns:
            Created Blender object
        """
        if primitive_type == 'CUBE':
            bpy.ops.mesh.primitive_cube_add(location=location)
        elif primitive_type == 'SPHERE':
            bpy.ops.mesh.primitive_uv_sphere_add(location=location)
        elif primitive_type == 'CYLINDER':
            bpy.ops.mesh.primitive_cylinder_add(location=location)
        elif primitive_type == 'CONE':
            bpy.ops.mesh.primitive_cone_add(location=location)
        else:
            raise ValueError(f"Unknown primitive type: {primitive_type}")

        obj = bpy.context.active_object
        obj.scale = scale
        self.placed_objects.append(obj)
        return obj

    def place_primitives_from_slices(self, slice_data, primitive_type='CYLINDER'):
        """
        Place primitives based on slice analysis data.

        Args:
            slice_data: List of slice analysis results
            primitive_type: Type of primitive to place

        Returns:
            List of created objects
        """
        created_objects = []

        for i, data in enumerate(slice_data):
            if data['radius'] > 0.01:  # Only place if radius is significant
                obj = self.create_primitive(
                    primitive_type=primitive_type,
                    location=data['center'],
                    scale=data['scale']
                )
                obj.name = f"Primitive_{i:02d}"
                created_objects.append(obj)

        return created_objects


class MeshJoiner:
    """Joins multiple mesh objects using boolean operations."""

    @staticmethod
    def join_with_boolean_union(objects, target_name="Joined_Mesh"):
        """
        Join multiple objects using boolean union operations.

        Args:
            objects: List of mesh objects to join
            target_name: Name for the resulting joined mesh

        Returns:
            The final joined object
        """
        if not objects:
            raise ValueError("No objects provided for joining")

        if len(objects) == 1:
            objects[0].name = target_name
            return objects[0]

        # Start with the first object as base
        base_obj = objects[0]
        base_obj.name = target_name

        # Apply boolean union with each subsequent object
        for i, obj in enumerate(objects[1:], 1):
            # Create boolean modifier
            modifier = base_obj.modifiers.new(name=f"Union_{i}", type='BOOLEAN')
            modifier.operation = 'UNION'
            modifier.object = obj
            modifier.solver = 'FAST'

            # Apply the modifier
            bpy.context.view_layer.objects.active = base_obj
            bpy.ops.object.modifier_apply(modifier=modifier.name)

            # Delete the source object
            bpy.data.objects.remove(obj, do_unlink=True)

        return base_obj

    @staticmethod
    def join_simple(objects, target_name="Joined_Mesh"):
        """
        Join objects using simple mesh joining (faster but less precise).

        Args:
            objects: List of mesh objects to join
            target_name: Name for the resulting joined mesh

        Returns:
            The joined object
        """
        if not objects:
            raise ValueError("No objects provided for joining")

        # Deselect all
        bpy.ops.object.select_all(action='DESELECT')

        # Select all objects to join
        for obj in objects:
            obj.select_set(True)

        # Set active object
        bpy.context.view_layer.objects.active = objects[0]

        # Join
        bpy.ops.object.join()

        # Rename
        joined_obj = bpy.context.active_object
        joined_obj.name = target_name

        return joined_obj


def clear_scene():
    """Remove all mesh objects from the scene."""
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()


def example_usage():
    """Example demonstrating primitive placement and mesh joining."""

    # Clear existing meshes
    clear_scene()

    # Define volume bounds
    bounds_min = Vector((-2, -2, 0))
    bounds_max = Vector((2, 2, 6))

    # Analyze slices
    print("Analyzing slices...")
    analyzer = SliceAnalyzer(bounds_min, bounds_max, num_slices=12)
    slice_data = analyzer.get_all_slice_data()

    # Place primitives based on analysis
    print("Placing primitives...")
    placer = PrimitivePlacer()
    objects = placer.place_primitives_from_slices(slice_data, primitive_type='CYLINDER')

    print(f"Placed {len(objects)} primitives")

    # Join all primitives using boolean union
    print("Joining meshes with boolean union...")
    joiner = MeshJoiner()
    final_mesh = joiner.join_with_boolean_union(objects, target_name="Sculpt_Base")

    print(f"Created final mesh: {final_mesh.name}")
    print("Ready for sculpting!")

    return final_mesh


# Run example if executed in Blender
if __name__ == "__main__":
    result = example_usage()
