"""
Phase 2 Step 1: Demonstrate Multi-Profile Extraction

Shows that multi-angle profile extraction works on Visual Hull mesh.

Usage:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python test_phase2_step1_profiles.py
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path.home() / 'blender_python_packages'))

import bpy
import numpy as np
from mathutils import Vector

# Phase 1: Visual Hull
from integration.multi_view.visual_hull import MultiViewVisualHull

# Phase 2: Multi-profile extraction (import directly to avoid cv2 dependency)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "mesh_profile_extractor",
    Path(__file__).parent / "integration/shape_matching/mesh_profile_extractor.py"
)
mesh_profile_extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mesh_profile_extractor)

extract_multi_angle_profiles = mesh_profile_extractor.extract_multi_angle_profiles
combine_profiles = mesh_profile_extractor.combine_profiles


def clear_scene():
    """Remove all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def create_visual_hull_from_images(turntable_dir, resolution=128):
    """Create Visual Hull mesh from existing turntable images."""
    # Import directly to avoid cv2 dependency
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "image_loader",
        Path(__file__).parent / "integration/image_processing/image_loader.py"
    )
    image_loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(image_loader)
    load_multi_view_auto = image_loader.load_multi_view_auto

    print(f"\nLoading images from {turntable_dir}...")
    views_dict = load_multi_view_auto(str(turntable_dir), num_views=12, include_top=True)
    print(f"✓ Loaded {len(views_dict)} views")

    # Create Visual Hull
    print(f"\nReconstructing Visual Hull at {resolution}³ resolution...")
    hull = MultiViewVisualHull(
        resolution=resolution,
        bounds_min=np.array([-2.0, -2.0, -2.0]),
        bounds_max=np.array([2.0, 2.0, 2.0])
    )

    # Add views
    for view_name, (img, angle, view_type) in views_dict.items():
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img_gray = np.mean(img, axis=2)
        else:
            img_gray = img

        # Create silhouette (black object on white background)
        silhouette = img_gray < 128
        hull.add_view_from_silhouette(silhouette, angle=angle, view_type=view_type)

    # Reconstruct
    voxel_grid = hull.reconstruct(verbose=True)
    occupied = voxel_grid.sum()
    print(f"✓ Visual Hull: {occupied:,} occupied voxels ({occupied/(resolution**3)*100:.2f}%)")

    # Create voxel mesh directly from voxel grid
    # This creates actual cube faces for each voxel, enabling raycasting
    print("\nCreating voxel mesh...")

    voxel_size_world = (hull.bounds_max - hull.bounds_min) / hull.resolution
    voxel_scale = voxel_size_world / 2.0  # Cube is 2x2x2, we want voxel_size

    # Find occupied voxels
    occupied_indices = np.argwhere(voxel_grid)
    print(f"✓ Found {len(occupied_indices):,} occupied voxels")

    # Sample only a subset for speed (every 2nd voxel in each dimension)
    sampled = occupied_indices[::4]  # Sample every 4th voxel
    print(f"✓ Sampling {len(sampled):,} voxels for mesh")

    # Create a cube for each voxel
    voxels_to_add = []
    for idx in sampled:
        i, j, k = idx
        pos = hull.bounds_min + (idx + 0.5) * voxel_size_world
        voxels_to_add.append((pos, voxel_scale))

    # Create initial cube
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, -100))  # Start off-screen
    cube_template = bpy.context.active_object

    # Create copies for voxels
    all_objects = []
    for i, (pos, scale) in enumerate(voxels_to_add[:500]):  # Limit to 500 voxels for speed
        if i == 0:
            obj = cube_template
        else:
            obj = cube_template.copy()
            obj.data = cube_template.data.copy()
            bpy.context.collection.objects.link(obj)

        obj.location = pos
        obj.scale = scale
        all_objects.append(obj)

    # Join all cubes
    print(f"Joining {len(all_objects)} cubes...")
    bpy.context.view_layer.objects.active = all_objects[0]
    for obj in all_objects:
        obj.select_set(True)

    bpy.ops.object.join()
    final_obj = bpy.context.active_object
    final_obj.name = "VisualHull"

    print(f"✓ Created voxel mesh: {len(final_obj.data.vertices):,} vertices, {len(final_obj.data.polygons):,} faces")

    return final_obj


def test_profile_extraction():
    """Step 1: Demonstrate multi-profile extraction works."""
    print("="*70)
    print("PHASE 2 STEP 1: MULTI-PROFILE EXTRACTION")
    print("="*70)
    print("\nGoal: Show that extracting and combining profiles works")
    print("Input: Visual Hull mesh from Phase 1")
    print("Output: Combined profile data for SliceAnalyzer")

    # Check if we have test images
    turntable_dir = Path("test_images/turntable_vase")
    if not turntable_dir.exists():
        print(f"\n✗ Error: {turntable_dir} not found")
        print("Run generate_turntable_sequence.py first to create test images")
        return

    # Step 1a: Create Visual Hull
    print("\n" + "="*70)
    print("STEP 1A: Create Visual Hull Mesh (Phase 1)")
    print("="*70)

    clear_scene()
    visual_hull = create_visual_hull_from_images(turntable_dir, resolution=128)

    if len(visual_hull.data.vertices) == 0:
        print("\n✗ Error: Visual Hull has 0 vertices")
        print("This indicates silhouette images may be incorrect")
        return

    # Step 1b: Extract multi-angle profiles
    print("\n" + "="*70)
    print("STEP 1B: Extract Multi-Angle Profiles (Phase 2)")
    print("="*70)

    # Define bounds for profile extraction
    bounds_min = Vector((-1.0, -1.0, -1.0))
    bounds_max = Vector((1.0, 1.0, 1.0))

    print(f"\nExtracting profiles from mesh...")
    print(f"  Bounds: {bounds_min} to {bounds_max}")
    print(f"  Number of angles: 12")
    print(f"  Samples per profile: 20")

    profiles = extract_multi_angle_profiles(
        visual_hull,
        num_angles=12,
        num_heights=20,
        bounds_min=bounds_min,
        bounds_max=bounds_max
    )

    print(f"\n✓ Extracted {len(profiles)} profiles")

    # Show sample profiles
    print("\nSample Profile Data:")
    for i, profile in enumerate(profiles[:3]):  # Show first 3
        print(f"\n  Profile {i} ({len(profile)} points):")
        radii = [r for h, r in profile]
        if radii:
            print(f"    Height range: {profile[0][0]:.3f} to {profile[-1][0]:.3f}")
            print(f"    Radius range: {min(radii):.3f} to {max(radii):.3f}")
            print(f"    Average radius: {np.mean(radii):.3f}")
        else:
            print(f"    Empty profile")

    # Step 1c: Combine profiles
    print("\n" + "="*70)
    print("STEP 1C: Combine Profiles (Phase 2)")
    print("="*70)

    print("\nTesting combination methods...")

    for method in ['mean', 'median', 'min', 'max']:
        combined = combine_profiles(profiles, method=method)
        radii = [r for h, r in combined]

        print(f"\n  {method.upper()} combination:")
        if radii:
            print(f"    Points: {len(combined)}")
            print(f"    Radius range: {min(radii):.3f} to {max(radii):.3f}")
            print(f"    Average radius: {np.mean(radii):.3f}")
        else:
            print(f"    Empty profile")

    # Use median as default
    combined_profile = combine_profiles(profiles, method='median')

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    radii = [r for h, r in combined_profile]

    print(f"\nVisual Hull Mesh:")
    print(f"  Vertices: {len(visual_hull.data.vertices):,}")

    print(f"\nProfile Extraction:")
    print(f"  Individual profiles: {len(profiles)}")
    print(f"  Combined profile (median): {len(combined_profile)} height samples")

    if radii:
        print(f"\nCombined Profile Statistics:")
        print(f"  Radius range: {min(radii):.3f} to {max(radii):.3f}")
        print(f"  Average radius: {np.mean(radii):.3f}")
        print(f"  Std dev: {np.std(radii):.3f}")

        print(f"\nProfile Data (first 5 samples):")
        for i, (height, radius) in enumerate(combined_profile[:5]):
            print(f"  Height {height:.3f}: radius {radius:.3f}")

        print("\n✓ SUCCESS: Multi-profile extraction working")
        print("✓ Ready for Step 2: SliceAnalyzer integration")
    else:
        print("\n✗ FAILURE: Combined profile is empty")
        print("Need to debug profile extraction")

    print("\n" + "="*70)
    print("STEP 1 COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_profile_extraction()
