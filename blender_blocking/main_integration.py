"""
Main integration script for Blender automated blocking tool.

This script ties together image processing, shape matching, and 3D operations
to create rough 3D blockouts from orthogonal reference images.
"""

import sys
from pathlib import Path
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def check_setup():
    """
    Verify that dependencies are properly installed and compatible.

    Raises helpful error messages if setup is incomplete or incorrect.
    """
    try:
        # Try importing PIL and accessing C extension
        from PIL import Image
        try:
            from PIL import _imaging
        except ImportError as e:
            print("\n" + "="*70)
            print("❌ SETUP ERROR: Pillow C extensions not compatible")
            print("="*70)
            print("\nYour Pillow installation is not compatible with this Python version.")
            print("This typically happens when:")
            print("  - You installed Pillow in a venv with Python 3.13")
            print("  - But Blender is using a different Python version (e.g., 3.11)")
            print("\nPillow includes compiled C extensions that must match your Python")
            print("version exactly. Virtual environment packages won't work.")
            print("\n🔧 REQUIRED FIX:")
            print("Install dependencies directly into Blender's Python:")
            print("\n  # Find Blender's Python path:")
            print("  # In Blender console: import sys; print(sys.executable)")
            print("\n  # Then install:")
            print("  /path/to/blender/python -m pip install numpy opencv-python Pillow scipy")
            print("\n📖 See BLENDER_SETUP.md for detailed instructions")
            print("="*70 + "\n")
            raise SystemExit(1)

        # Check other critical imports
        import cv2
        import scipy

    except ImportError as e:
        if "PIL" not in str(e):
            print("\n" + "="*70)
            print("❌ SETUP ERROR: Missing dependencies")
            print("="*70)
            print(f"\nCould not import required package: {e}")
            print("\n🔧 REQUIRED FIX:")
            print("Install dependencies into Blender's Python:")
            print("\n  # Find Blender's Python:")
            print("  # In Blender console: import sys; print(sys.executable)")
            print("\n  # Then install:")
            print("  /path/to/blender/python -m pip install numpy opencv-python Pillow scipy")
            print("\n📖 See BLENDER_SETUP.md for complete setup guide")
            print("="*70 + "\n")
            raise SystemExit(1)


# Run setup check when module is imported
check_setup()

# Import integration modules
from integration.image_processing.image_loader import load_orthogonal_views
from integration.image_processing.image_processor import process_image
from integration.shape_matching.contour_analyzer import find_contours, analyze_shape

# Import Blender modules (only available when running in Blender)
try:
    import bpy
    from primitives.primitives import spawn_cube, spawn_sphere, spawn_cylinder
    from placement.primitive_placement import SliceAnalyzer, PrimitivePlacer, MeshJoiner
    from integration.blender_ops.scene_setup import setup_scene, add_camera, add_lighting
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    print("Warning: Blender API not available. Running in analysis-only mode.")


class BlockingWorkflow:
    """Main workflow class for automated blocking from reference images."""

    def __init__(self, front_path=None, side_path=None, top_path=None):
        """
        Initialize the blocking workflow.

        Args:
            front_path: Path to front view reference image
            side_path: Path to side view reference image
            top_path: Path to top view reference image
        """
        self.front_path = front_path
        self.side_path = side_path
        self.top_path = top_path
        self.views = {}
        self.processed_views = {}
        self.contours = {}
        self.shape_analysis = {}
        self.placement_data = None
        self.created_objects = []

    def load_images(self):
        """Load orthogonal reference images."""
        print("Loading images...")
        self.views = load_orthogonal_views(
            front_path=self.front_path,
            side_path=self.side_path,
            top_path=self.top_path
        )

        if not self.views:
            raise ValueError("No images loaded. Please provide at least one reference image.")

        print(f"Loaded {len(self.views)} views: {', '.join(self.views.keys())}")
        return self.views

    def process_images(self):
        """Process images to extract edges and prepare for shape analysis."""
        print("Processing images...")

        for view_name, image in self.views.items():
            # Process image: normalize and extract edges
            processed = process_image(image, extract_edges_flag=True, normalize_flag=True)
            self.processed_views[view_name] = processed
            print(f"  Processed {view_name} view ({image.shape})")

        return self.processed_views

    def analyze_shapes(self):
        """Analyze shapes from processed images."""
        print("Analyzing shapes...")

        for view_name, edge_image in self.processed_views.items():
            # Find contours
            contours = find_contours(edge_image)
            self.contours[view_name] = contours

            # Analyze each contour
            shapes = []
            for i, contour in enumerate(contours):
                if len(contour) >= 5:  # Need at least 5 points for meaningful analysis
                    analysis = analyze_shape(contour)
                    if analysis['area'] > 100:  # Filter out tiny contours
                        shapes.append(analysis)

            self.shape_analysis[view_name] = shapes
            print(f"  {view_name}: Found {len(contours)} contours, {len(shapes)} significant shapes")

        return self.shape_analysis

    def determine_primitive_type(self, shape_info):
        """
        Determine which primitive type best matches a shape.

        Args:
            shape_info: Dictionary with shape properties

        Returns:
            String indicating primitive type ('CUBE', 'SPHERE', 'CYLINDER')
        """
        circularity = shape_info.get('circularity', 0)
        aspect_ratio = shape_info.get('aspect_ratio', 1.0)

        # High circularity -> sphere or cylinder
        if circularity > 0.8:
            return 'SPHERE'
        elif circularity > 0.5:
            return 'CYLINDER'
        else:
            # Low circularity, check aspect ratio
            if 0.8 < aspect_ratio < 1.2:
                return 'CUBE'
            else:
                return 'CYLINDER'

    def calculate_bounds_from_shapes(self):
        """
        Calculate 3D bounds based on analyzed shapes from multiple views.

        Returns:
            Tuple of (bounds_min, bounds_max) as vectors
        """
        # Use front and side views to estimate 3D dimensions
        front_shapes = self.shape_analysis.get('front', [])
        side_shapes = self.shape_analysis.get('side', [])

        if not front_shapes and not side_shapes:
            # Default bounds if no shapes found
            return ((-2, -2, 0), (2, 2, 4))

        # Get largest shape from each view
        def get_largest_bbox(shapes):
            if not shapes:
                return None
            largest = max(shapes, key=lambda s: s['area'])
            return largest['bounding_box']

        front_bbox = get_largest_bbox(front_shapes)
        side_bbox = get_largest_bbox(side_shapes)

        # Scale factor from image coordinates to world coordinates
        scale = 0.01  # 1 pixel = 0.01 units

        if front_bbox and side_bbox:
            fx, fy, fw, fh = front_bbox
            sx, sy, sw, sh = side_bbox

            # front view: x, z; side view: y, z
            width = fw * scale
            depth = sw * scale
            height = max(fh, sh) * scale
        elif front_bbox:
            fx, fy, fw, fh = front_bbox
            width = fw * scale
            depth = width  # Assume square
            height = fh * scale
        else:  # side_bbox
            sx, sy, sw, sh = side_bbox
            depth = sw * scale
            width = depth  # Assume square
            height = sh * scale

        # Center the object at origin
        bounds_min = (-width/2, -depth/2, 0)
        bounds_max = (width/2, depth/2, height)

        return bounds_min, bounds_max

    def create_3d_blockout(self, num_slices=10, primitive_type='CYLINDER'):
        """
        Create 3D blockout in Blender based on analyzed shapes.

        Args:
            num_slices: Number of vertical slices for reconstruction
            primitive_type: Default primitive type to use

        Returns:
            Final joined mesh object
        """
        if not BLENDER_AVAILABLE:
            print("Error: Blender API not available. Cannot create 3D blockout.")
            return None

        print("Creating 3D blockout in Blender...")

        # Setup clean Blender scene
        setup_scene(clear_existing=True)

        # Calculate bounds from shape analysis
        bounds_min, bounds_max = self.calculate_bounds_from_shapes()
        print(f"  Bounds: {bounds_min} to {bounds_max}")

        # Analyze slices
        print(f"  Analyzing {num_slices} slices...")
        analyzer = SliceAnalyzer(bounds_min, bounds_max, num_slices=num_slices)
        slice_data = analyzer.get_all_slice_data()

        # Place primitives
        print("  Placing primitives...")
        placer = PrimitivePlacer()

        # Determine best primitive type from shape analysis
        if self.shape_analysis:
            all_shapes = []
            for shapes in self.shape_analysis.values():
                all_shapes.extend(shapes)

            if all_shapes:
                # Use the primitive type from the largest shape
                largest_shape = max(all_shapes, key=lambda s: s['area'])
                primitive_type = self.determine_primitive_type(largest_shape)
                print(f"  Auto-selected primitive type: {primitive_type}")

        objects = placer.place_primitives_from_slices(slice_data, primitive_type=primitive_type)
        self.created_objects = objects
        print(f"  Placed {len(objects)} primitives")

        # Join primitives
        if objects:
            print("  Joining meshes...")
            joiner = MeshJoiner()
            final_mesh = joiner.join_with_boolean_union(objects, target_name="Blockout_Mesh")

            # Setup camera and lighting
            print("  Setting up camera and lighting...")
            add_camera()
            add_lighting()

            print(f"✓ Created blockout mesh: {final_mesh.name}")
            return final_mesh
        else:
            print("  Warning: No primitives were created")
            return None

    def run_full_workflow(self, num_slices=10):
        """
        Run the complete workflow from images to 3D blockout.

        Args:
            num_slices: Number of slices for 3D reconstruction

        Returns:
            Final mesh object (if Blender available)
        """
        print("="*60)
        print("BLENDER AUTOMATED BLOCKING WORKFLOW")
        print("="*60)

        # Step 1: Load images
        self.load_images()

        # Step 2: Process images
        self.process_images()

        # Step 3: Analyze shapes
        self.analyze_shapes()

        # Step 4: Create 3D blockout (only if Blender available)
        result = None
        if BLENDER_AVAILABLE:
            result = self.create_3d_blockout(num_slices=num_slices)
        else:
            print("\nShape analysis complete. Run in Blender to create 3D blockout.")
            print(f"Analyzed views: {', '.join(self.shape_analysis.keys())}")
            for view, shapes in self.shape_analysis.items():
                print(f"  {view}: {len(shapes)} shapes")

        print("="*60)
        print("WORKFLOW COMPLETE")
        print("="*60)

        return result


def example_workflow_with_images(front_path=None, side_path=None, top_path=None):
    """
    Run example workflow with provided image paths.

    Args:
        front_path: Path to front view image
        side_path: Path to side view image
        top_path: Path to top view image

    Returns:
        BlockingWorkflow instance
    """
    workflow = BlockingWorkflow(
        front_path=front_path,
        side_path=side_path,
        top_path=top_path
    )

    workflow.run_full_workflow(num_slices=12)

    return workflow


def example_workflow_no_images():
    """
    Run example workflow without images (generates procedural blockout).
    Useful for testing the 3D generation pipeline.
    """
    if not BLENDER_AVAILABLE:
        print("Error: Blender API required for procedural generation")
        return None

    print("="*60)
    print("PROCEDURAL BLOCKOUT (No Reference Images)")
    print("="*60)

    # Setup scene
    setup_scene(clear_existing=True)

    # Define bounds
    bounds_min = (-2, -2, 0)
    bounds_max = (2, 2, 6)

    # Analyze and place
    analyzer = SliceAnalyzer(bounds_min, bounds_max, num_slices=12)
    slice_data = analyzer.get_all_slice_data()

    placer = PrimitivePlacer()
    objects = placer.place_primitives_from_slices(slice_data, primitive_type='CYLINDER')

    # Join
    joiner = MeshJoiner()
    final_mesh = joiner.join_with_boolean_union(objects, target_name="Procedural_Blockout")

    # Setup scene
    add_camera()
    add_lighting()

    print(f"✓ Created procedural blockout: {final_mesh.name}")
    print("="*60)

    return final_mesh


if __name__ == "__main__":
    # When run in Blender, execute procedural example
    if BLENDER_AVAILABLE:
        print("Running in Blender - creating procedural blockout...")
        result = example_workflow_no_images()
    else:
        print("Not running in Blender - image analysis only mode")
        print("\nTo use this script:")
        print("1. With reference images in Blender:")
        print("   >>> from blender_blocking.main_integration import example_workflow_with_images")
        print("   >>> example_workflow_with_images('front.png', 'side.png', 'top.png')")
        print("\n2. Procedural generation in Blender:")
        print("   >>> from blender_blocking.main_integration import example_workflow_no_images")
        print("   >>> example_workflow_no_images()")
