use image_preprocessing::{ImagePreprocessor, OrthogonalView, SilhouetteConfig};

fn main() -> anyhow::Result<()> {
    let config = SilhouetteConfig {
        canny_threshold1: 50.0,
        canny_threshold2: 150.0,
        blur_kernel_size: 5,
        morph_kernel_size: 5,
        use_adaptive_threshold: false,
    };

    let preprocessor = ImagePreprocessor::new(config);

    println!("Extracting silhouettes from orthogonal views...");

    let silhouettes = preprocessor.extract_multi_view_silhouettes(
        Some("front_view.png"),
        Some("side_view.png"),
        Some("top_view.png"),
    )?;

    if let Some(front) = silhouettes.get_view(OrthogonalView::Front) {
        println!("Front view silhouette extracted: {}x{}", front.cols(), front.rows());
    }

    if let Some(side) = silhouettes.get_view(OrthogonalView::Side) {
        println!("Side view silhouette extracted: {}x{}", side.cols(), side.rows());
    }

    if let Some(top) = silhouettes.get_view(OrthogonalView::Top) {
        println!("Top view silhouette extracted: {}x{}", top.cols(), top.rows());
    }

    println!("Processing complete!");

    Ok(())
}
