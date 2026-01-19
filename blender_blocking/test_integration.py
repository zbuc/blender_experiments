"""
Test script for the Blender blocking integration.

This script creates sample reference images and tests the end-to-end workflow.
"""

import sys
from pathlib import Path
import numpy as np

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Add ~/blender_python_packages for user-installed dependencies (numpy, opencv-python, Pillow, scipy)
sys.path.insert(0, str(Path.home() / 'blender_python_packages'))

# Check if we can use image creation libraries
try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available, cannot create test images")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available, cannot create test images")


def create_simple_bottle_silhouette(width=512, height=512, view='front'):
    """
    Create a simple bottle silhouette for testing.

    Args:
        width: Image width
        height: Image height
        view: Which view to create ('front', 'side', 'top')

    Returns:
        PIL Image or numpy array
    """
    if not PIL_AVAILABLE:
        print("Error: PIL required to create test images")
        return None

    # Create white background
    img = Image.new('L', (width, height), color=255)
    draw = ImageDraw.Draw(img)

    if view in ['front', 'side']:
        # Bottle shape (front/side view)
        # Neck
        neck_width = 60
        neck_height = 100
        neck_x = width // 2 - neck_width // 2
        neck_y = height // 4

        # Body
        body_width = 150
        body_height = 250
        body_x = width // 2 - body_width // 2
        body_y = neck_y + neck_height

        # Draw neck (rectangle)
        draw.rectangle(
            [neck_x, neck_y, neck_x + neck_width, neck_y + neck_height],
            fill=0
        )

        # Draw body (rounded shape)
        draw.ellipse(
            [body_x, body_y, body_x + body_width, body_y + body_height],
            fill=0
        )

    else:  # top view
        # Circular cross-section
        center_x = width // 2
        center_y = height // 2
        radius = 75

        draw.ellipse(
            [center_x - radius, center_y - radius,
             center_x + radius, center_y + radius],
            fill=0
        )

    return img


def create_simple_vase_silhouette(width=512, height=512, view='front'):
    """
    Create a simple vase silhouette for testing.

    Args:
        width: Image width
        height: Image height
        view: Which view to create ('front', 'side', 'top')

    Returns:
        PIL Image
    """
    if not PIL_AVAILABLE:
        print("Error: PIL required to create test images")
        return None

    # Create white background
    img = Image.new('L', (width, height), color=255)
    draw = ImageDraw.Draw(img)

    if view in ['front', 'side']:
        # Vase profile (tapered from bottom to top, then flaring at top)
        center_x = width // 2

        # Bottom
        bottom_width = 120
        bottom_y = height - 50

        # Middle (narrowest part)
        middle_width = 80
        middle_y = height // 2

        # Top (flared)
        top_width = 140
        top_y = 50

        # Draw as series of ellipses
        num_segments = 20
        for i in range(num_segments):
            t = i / (num_segments - 1)
            y = int(top_y + t * (bottom_y - top_y))

            # Interpolate width with a curve
            if t < 0.3:
                # Top flare
                w = top_width - (top_width - middle_width) * (t / 0.3)
            elif t < 0.7:
                # Middle section
                w = middle_width
            else:
                # Bottom expansion
                w = middle_width + (bottom_width - middle_width) * ((t - 0.7) / 0.3)

            w = int(w)
            h = int((bottom_y - top_y) / num_segments + 5)

            draw.ellipse(
                [center_x - w//2, y - h//2, center_x + w//2, y + h//2],
                fill=0
            )

    else:  # top view
        # Circular cross-section
        center_x = width // 2
        center_y = height // 2
        radius = 70

        draw.ellipse(
            [center_x - radius, center_y - radius,
             center_x + radius, center_y + radius],
            fill=0
        )

    return img


def create_test_images(output_dir='test_images', shape='bottle'):
    """
    Create a set of test reference images.

    Args:
        output_dir: Directory to save images
        shape: Shape to create ('bottle' or 'vase')

    Returns:
        Dictionary mapping view names to file paths
    """
    if not PIL_AVAILABLE:
        print("Error: PIL required to create test images")
        return None

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"Creating test images ({shape}) in {output_dir}/")

    # Create function based on shape
    if shape == 'bottle':
        create_func = create_simple_bottle_silhouette
    elif shape == 'vase':
        create_func = create_simple_vase_silhouette
    else:
        print(f"Unknown shape: {shape}")
        return None

    # Create images
    views = {}
    for view in ['front', 'side', 'top']:
        img = create_func(view=view)
        filename = output_path / f"{shape}_{view}.png"
        img.save(filename)
        views[view] = str(filename)
        print(f"  Created {filename}")

    print("✓ Test images created")
    return views


def test_image_processing():
    """Test image loading and processing without Blender."""
    print("\n" + "="*60)
    print("TEST: Image Processing Pipeline")
    print("="*60)

    # Create test images
    views = create_test_images(shape='vase')
    if not views:
        print("Failed to create test images")
        return False

    # Import and test workflow
    from main_integration import BlockingWorkflow

    workflow = BlockingWorkflow(
        front_path=views['front'],
        side_path=views.get('side'),
        top_path=views.get('top')
    )

    try:
        # Test loading
        workflow.load_images()
        print("✓ Image loading successful")

        # Test processing
        workflow.process_images()
        print("✓ Image processing successful")

        # Test shape analysis
        workflow.analyze_shapes()
        print("✓ Shape analysis successful")

        # Print results
        print("\nAnalysis Results:")
        for view, shapes in workflow.shape_analysis.items():
            print(f"  {view}: {len(shapes)} significant shapes")
            if shapes:
                largest = max(shapes, key=lambda s: s['area'])
                print(f"    Largest: area={largest['area']:.0f}, "
                      f"circularity={largest['circularity']:.2f}, "
                      f"aspect_ratio={largest['aspect_ratio']:.2f}")

        print("\n✓ All image processing tests passed")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_blender_workflow():
    """Test complete workflow in Blender."""
    print("\n" + "="*60)
    print("TEST: Full Blender Workflow")
    print("="*60)

    try:
        import bpy
        BLENDER_AVAILABLE = True
    except ImportError:
        print("Blender not available, skipping Blender tests")
        return None

    # Create test images
    views = create_test_images(shape='vase')
    if not views:
        print("Failed to create test images")
        return False

    # Run workflow
    from main_integration import example_workflow_with_images

    try:
        workflow = example_workflow_with_images(
            front_path=views['front'],
            side_path=views.get('side'),
            top_path=views.get('top')
        )

        print("\n✓ Full Blender workflow completed successfully")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_procedural_generation():
    """Test procedural generation in Blender (no images)."""
    print("\n" + "="*60)
    print("TEST: Procedural Generation")
    print("="*60)

    try:
        import bpy
        BLENDER_AVAILABLE = True
    except ImportError:
        print("Blender not available, skipping procedural test")
        return None

    from main_integration import example_workflow_no_images

    try:
        result = example_workflow_no_images()
        print("\n✓ Procedural generation test passed")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all available tests."""
    print("\n" + "="*60)
    print("BLENDER BLOCKING INTEGRATION TEST SUITE")
    print("="*60)

    results = {}

    # Test 1: Image processing (always available)
    results['image_processing'] = test_image_processing()

    # Test 2: Full workflow (Blender only)
    results['full_workflow'] = test_blender_workflow()

    # Test 3: Procedural generation (Blender only)
    results['procedural'] = test_procedural_generation()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "- SKIP"
        print(f"  {test_name}: {status}")

    print("="*60)

    return results


if __name__ == "__main__":
    run_all_tests()
