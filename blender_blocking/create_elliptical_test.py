"""Create test images with elliptical cross-section to test directional profiles."""

from pathlib import Path
from PIL import Image, ImageDraw


def create_elliptical_vase(width=512, height=512, view='front'):
    """
    Create a vase with elliptical cross-section.
    Front view (X-Z): narrower
    Side view (Y-Z): wider
    """
    img = Image.new('L', (width, height), color=255)
    draw = ImageDraw.Draw(img)

    center_x = width // 2
    bottom_y = height - 50
    top_y = 50

    if view == 'front':
        # Front view: X-Z plane - narrower (60% of side view)
        bottom_width = 80  # Narrower
        middle_width = 50  # Narrower
        top_width = 90     # Narrower
    elif view == 'side':
        # Side view: Y-Z plane - wider
        bottom_width = 133  # Wider (80 / 0.6)
        middle_width = 83   # Wider (50 / 0.6)
        top_width = 150     # Wider (90 / 0.6)
    else:  # top view
        # Top view should show ellipse (wider Y, narrower X)
        center_y = height // 2
        # Ellipse: X-radius = 45 (matching front narrower), Y-radius = 75 (matching side wider)
        x_radius = 45
        y_radius = 75
        draw.ellipse(
            [center_x - x_radius, center_y - y_radius,
             center_x + x_radius, center_y + y_radius],
            fill=0
        )
        return img

    # Draw vertical profile for front/side views
    num_segments = 20
    for i in range(num_segments):
        t = i / (num_segments - 1)
        y = int(top_y + t * (bottom_y - top_y))

        # Vase profile curve
        if t < 0.3:
            w = top_width - (top_width - middle_width) * (t / 0.3)
        elif t < 0.7:
            w = middle_width
        else:
            w = middle_width + (bottom_width - middle_width) * ((t - 0.7) / 0.3)

        w = int(w)
        h = int((bottom_y - top_y) / num_segments + 5)

        draw.ellipse(
            [center_x - w//2, y - h//2, center_x + w//2, y + h//2],
            fill=0
        )

    return img


def main():
    """Create elliptical test images."""
    output_dir = Path('test_images')
    output_dir.mkdir(exist_ok=True)

    print("Creating elliptical vase test images...")
    print(f"Output directory: {output_dir}/")
    print()
    print("This vase has an elliptical cross-section:")
    print("  - Front view (X-Z): narrower profile")
    print("  - Side view (Y-Z): wider profile")
    print("  - Top view: elliptical (narrow X, wide Y)")
    print()

    for view in ['front', 'side', 'top']:
        img = create_elliptical_vase(view=view)
        filename = output_dir / f"elliptical_vase_{view}.png"
        img.save(filename)
        print(f"  ✓ {filename}")

    print()
    print("Test images created successfully!")
    print()
    print("This will test if directional profiles work correctly:")
    print("  - Front profile should control X-axis width (narrower)")
    print("  - Side profile should control Y-axis width (wider)")
    print("  - Result should match the elliptical top view")


if __name__ == "__main__":
    main()
