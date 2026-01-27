"""
Simple script to create test reference images.
This script has minimal dependencies and can run without numpy/opencv.
"""

from __future__ import annotations

from pathlib import Path

try:
    from PIL import Image, ImageDraw

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Error: PIL/Pillow is required. Install with: pip install Pillow")
    exit(1)


def create_bottle_silhouette(
    width: int = 512, height: int = 512, view: str = "front"
) -> object:
    """Create a simple bottle silhouette."""
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)

    if view in ["front", "side"]:
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

        # Draw neck
        draw.rectangle(
            [neck_x, neck_y, neck_x + neck_width, neck_y + neck_height], fill=0
        )

        # Draw body
        draw.ellipse(
            [body_x, body_y, body_x + body_width, body_y + body_height], fill=0
        )
    else:  # top view
        center_x = width // 2
        center_y = height // 2
        radius = 75
        draw.ellipse(
            [
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ],
            fill=0,
        )

    return img


def create_vase_silhouette(
    width: int = 512, height: int = 512, view: str = "front"
) -> object:
    """Create a simple vase silhouette."""
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)

    if view in ["front", "side"]:
        center_x = width // 2
        bottom_width = 120
        bottom_y = height - 50
        middle_width = 80
        middle_y = height // 2
        top_width = 140
        top_y = 50

        # Draw as series of ellipses
        num_segments = 20
        for i in range(num_segments):
            t = i / (num_segments - 1)
            y = int(top_y + t * (bottom_y - top_y))

            if t < 0.3:
                w = top_width - (top_width - middle_width) * (t / 0.3)
            elif t < 0.7:
                w = middle_width
            else:
                w = middle_width + (bottom_width - middle_width) * ((t - 0.7) / 0.3)

            w = int(w)
            h = int((bottom_y - top_y) / num_segments + 5)

            draw.ellipse(
                [center_x - w // 2, y - h // 2, center_x + w // 2, y + h // 2], fill=0
            )
    else:  # top view
        center_x = width // 2
        center_y = height // 2
        radius = 70
        draw.ellipse(
            [
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ],
            fill=0,
        )

    return img


def create_cube_silhouette(
    width: int = 512, height: int = 512, view: str = "front"
) -> object:
    """Create a simple cube silhouette."""
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)

    # Simple square for all views
    size = 200
    x = (width - size) // 2
    y = (height - size) // 2

    draw.rectangle([x, y, x + size, y + size], fill=0)

    return img


def main() -> None:
    """Create test images for different shapes."""
    shapes = {
        "bottle": create_bottle_silhouette,
        "vase": create_vase_silhouette,
        "cube": create_cube_silhouette,
    }

    output_dir = Path("test_images")
    output_dir.mkdir(exist_ok=True)

    print("Creating test reference images...")
    print(f"Output directory: {output_dir}/")
    print()

    for shape_name, create_func in shapes.items():
        print(f"Creating {shape_name} images:")
        for view in ["front", "side", "top"]:
            img = create_func(view=view)
            filename = output_dir / f"{shape_name}_{view}.png"
            img.save(filename)
            print(f"  OK: {filename}")
        print()

    print("All test images created successfully!")
    print()
    print("To use these images in Blender:")
    print("  1. Open Blender")
    print("  2. Open the Scripting workspace")
    print("  3. Run the following:")
    print()
    print("    import sys")
    print(f"    sys.path.append('{Path.cwd()}')")
    print(
        "    from blender_blocking.main_integration import example_workflow_with_images"
    )
    print("    example_workflow_with_images(")
    print("        'test_images/vase_front.png',")
    print("        'test_images/vase_side.png',")
    print("        'test_images/vase_top.png'")
    print("    )")


if __name__ == "__main__":
    main()
