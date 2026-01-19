# Image Preprocessing and Silhouette Extraction Module

Rust library for extracting clean silhouettes from orthogonal reference photos using OpenCV.

## Features

- **Edge Detection**: Canny edge detector with configurable thresholds
- **Thresholding**: Otsu's method for automatic binary thresholding
- **Morphological Operations**: Closing and opening to clean up silhouettes
- **Multi-View Support**: Process front, side, and top orthogonal views
- **Gaussian Blur**: Noise reduction preprocessing
- **Binary Masks**: Clean binary silhouette output

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
image_preprocessing = "0.1.0"
opencv = "0.95"
```

## Usage

### Basic Example

```rust
use image_preprocessing::{ImagePreprocessor, OrthogonalView, SilhouetteConfig};

fn main() -> anyhow::Result<()> {
    let preprocessor = ImagePreprocessor::with_default_config();

    let silhouette = preprocessor.extract_silhouette(
        "reference_photo.png",
        OrthogonalView::Front
    )?;

    println!("Silhouette extracted: {}x{}", silhouette.cols(), silhouette.rows());
    Ok(())
}
```

### Custom Configuration

```rust
use image_preprocessing::{ImagePreprocessor, SilhouetteConfig};

let config = SilhouetteConfig {
    canny_threshold1: 100.0,
    canny_threshold2: 200.0,
    blur_kernel_size: 7,
    morph_kernel_size: 5,
    use_adaptive_threshold: false,
};

let preprocessor = ImagePreprocessor::new(config);
```

### Multi-View Processing

```rust
let silhouettes = preprocessor.extract_multi_view_silhouettes(
    Some("front.png"),
    Some("side.png"),
    Some("top.png"),
)?;

if let Some(front) = silhouettes.get_view(OrthogonalView::Front) {
    // Process front view silhouette
}
```

## API Reference

### `ImagePreprocessor`

Main preprocessing interface.

**Methods:**
- `new(config: SilhouetteConfig)` - Create with custom config
- `with_default_config()` - Create with defaults
- `load_image(path: &str, grayscale: bool)` - Load image from path
- `apply_gaussian_blur(image: &Mat)` - Apply Gaussian blur
- `detect_edges(image: &Mat)` - Canny edge detection
- `apply_threshold(image: &Mat)` - Otsu thresholding
- `apply_morphology(image: &Mat, operation: i32)` - Morphological ops
- `extract_silhouette(path: &str, view: OrthogonalView)` - Extract single view
- `extract_multi_view_silhouettes(...)` - Extract multiple views

### `SilhouetteConfig`

Configuration parameters.

**Fields:**
- `canny_threshold1: f64` - Lower Canny threshold (default: 50.0)
- `canny_threshold2: f64` - Upper Canny threshold (default: 150.0)
- `blur_kernel_size: i32` - Gaussian blur kernel (default: 5)
- `morph_kernel_size: i32` - Morphology kernel (default: 5)
- `use_adaptive_threshold: bool` - Use thresholding instead of edges (default: false)

### `OrthogonalView`

Enum for view types: `Front`, `Side`, `Top`

## Error Handling

The module uses a custom `PreprocessingError` type:

```rust
pub enum PreprocessingError {
    OpenCVError(opencv::Error),
    InvalidPath(String),
    EmptyImage,
    InvalidParameters(String),
}
```

## Requirements

- OpenCV 4.x installed on system
- Rust 2024 edition

## Testing

Run tests with:

```bash
cargo test
```

Run example:

```bash
cargo run --example extract_silhouette
```

## Algorithm Pipeline

1. Load image in grayscale
2. Apply Gaussian blur for noise reduction
3. Edge detection with Canny OR thresholding with Otsu
4. Morphological closing to fill gaps
5. Morphological opening to remove noise
6. Return binary mask

## Use Cases

- 3D modeling reference preparation
- Blender orthographic reference photos
- Automated silhouette extraction for modeling
- Multi-view 3D reconstruction preprocessing
