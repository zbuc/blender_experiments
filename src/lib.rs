use opencv::{
    core::{Mat, Size, CV_8UC1},
    imgcodecs::{imread, IMREAD_COLOR, IMREAD_GRAYSCALE},
    imgproc::{
        canny, threshold, gaussian_blur, morphology_ex,
        THRESH_BINARY, THRESH_OTSU, MORPH_CLOSE, MORPH_OPEN,
    },
    prelude::*,
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PreprocessingError {
    #[error("OpenCV error: {0}")]
    OpenCVError(#[from] opencv::Error),

    #[error("Invalid image path: {0}")]
    InvalidPath(String),

    #[error("Empty image")]
    EmptyImage,

    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
}

pub type Result<T> = std::result::Result<T, PreprocessingError>;

#[derive(Debug, Clone, Copy)]
pub enum OrthogonalView {
    Front,
    Side,
    Top,
}

#[derive(Debug, Clone)]
pub struct SilhouetteConfig {
    pub canny_threshold1: f64,
    pub canny_threshold2: f64,
    pub blur_kernel_size: i32,
    pub morph_kernel_size: i32,
    pub use_adaptive_threshold: bool,
}

impl Default for SilhouetteConfig {
    fn default() -> Self {
        Self {
            canny_threshold1: 50.0,
            canny_threshold2: 150.0,
            blur_kernel_size: 5,
            morph_kernel_size: 5,
            use_adaptive_threshold: false,
        }
    }
}

pub struct ImagePreprocessor {
    config: SilhouetteConfig,
}

impl ImagePreprocessor {
    pub fn new(config: SilhouetteConfig) -> Self {
        Self { config }
    }

    pub fn with_default_config() -> Self {
        Self {
            config: SilhouetteConfig::default(),
        }
    }

    pub fn load_image(&self, path: &str, grayscale: bool) -> Result<Mat> {
        let flag = if grayscale { IMREAD_GRAYSCALE } else { IMREAD_COLOR };
        let img = imread(path, flag)?;

        if img.empty() {
            return Err(PreprocessingError::EmptyImage);
        }

        Ok(img)
    }

    pub fn apply_gaussian_blur(&self, image: &Mat) -> Result<Mat> {
        let mut blurred = Mat::default();
        let kernel_size = Size::new(self.config.blur_kernel_size, self.config.blur_kernel_size);

        gaussian_blur(
            image,
            &mut blurred,
            kernel_size,
            0.0,
            0.0,
            opencv::core::BORDER_DEFAULT,
        )?;

        Ok(blurred)
    }

    pub fn detect_edges(&self, image: &Mat) -> Result<Mat> {
        let mut edges = Mat::default();

        canny(
            image,
            &mut edges,
            self.config.canny_threshold1,
            self.config.canny_threshold2,
            3,
            false,
        )?;

        Ok(edges)
    }

    pub fn apply_threshold(&self, image: &Mat) -> Result<Mat> {
        let mut thresholded = Mat::default();

        threshold(
            image,
            &mut thresholded,
            0.0,
            255.0,
            THRESH_BINARY | THRESH_OTSU,
        )?;

        Ok(thresholded)
    }

    pub fn apply_morphology(&self, image: &Mat, operation: i32) -> Result<Mat> {
        let mut result = Mat::default();
        let kernel = Mat::ones(
            self.config.morph_kernel_size,
            self.config.morph_kernel_size,
            CV_8UC1,
        )?;

        morphology_ex(
            image,
            &mut result,
            operation,
            &kernel,
            opencv::core::Point::new(-1, -1),
            1,
            opencv::core::BORDER_CONSTANT,
            opencv::imgproc::morphology_default_border_value()?,
        )?;

        Ok(result)
    }

    pub fn extract_silhouette(&self, image_path: &str, view: OrthogonalView) -> Result<Mat> {
        let gray_image = self.load_image(image_path, true)?;
        let blurred = self.apply_gaussian_blur(&gray_image)?;

        let binary_mask = if self.config.use_adaptive_threshold {
            self.apply_threshold(&blurred)?
        } else {
            let edges = self.detect_edges(&blurred)?;
            let closed = self.apply_morphology(&edges, MORPH_CLOSE)?;
            self.apply_morphology(&closed, MORPH_OPEN)?
        };

        Ok(binary_mask)
    }

    pub fn extract_multi_view_silhouettes(
        &self,
        front_path: Option<&str>,
        side_path: Option<&str>,
        top_path: Option<&str>,
    ) -> Result<MultiViewSilhouettes> {
        let front = front_path
            .map(|path| self.extract_silhouette(path, OrthogonalView::Front))
            .transpose()?;

        let side = side_path
            .map(|path| self.extract_silhouette(path, OrthogonalView::Side))
            .transpose()?;

        let top = top_path
            .map(|path| self.extract_silhouette(path, OrthogonalView::Top))
            .transpose()?;

        Ok(MultiViewSilhouettes { front, side, top })
    }
}

pub struct MultiViewSilhouettes {
    pub front: Option<Mat>,
    pub side: Option<Mat>,
    pub top: Option<Mat>,
}

impl MultiViewSilhouettes {
    pub fn get_view(&self, view: OrthogonalView) -> Option<&Mat> {
        match view {
            OrthogonalView::Front => self.front.as_ref(),
            OrthogonalView::Side => self.side.as_ref(),
            OrthogonalView::Top => self.top.as_ref(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SilhouetteConfig::default();
        assert_eq!(config.canny_threshold1, 50.0);
        assert_eq!(config.canny_threshold2, 150.0);
        assert_eq!(config.blur_kernel_size, 5);
    }

    #[test]
    fn test_preprocessor_creation() {
        let preprocessor = ImagePreprocessor::with_default_config();
        assert_eq!(preprocessor.config.blur_kernel_size, 5);
    }

    #[test]
    fn test_custom_config() {
        let config = SilhouetteConfig {
            canny_threshold1: 100.0,
            canny_threshold2: 200.0,
            blur_kernel_size: 7,
            morph_kernel_size: 3,
            use_adaptive_threshold: true,
        };

        let preprocessor = ImagePreprocessor::new(config.clone());
        assert_eq!(preprocessor.config.canny_threshold1, 100.0);
        assert_eq!(preprocessor.config.use_adaptive_threshold, true);
    }

    #[test]
    fn test_multi_view_silhouettes_get_view() {
        let silhouettes = MultiViewSilhouettes {
            front: None,
            side: None,
            top: None,
        };

        assert!(silhouettes.get_view(OrthogonalView::Front).is_none());
        assert!(silhouettes.get_view(OrthogonalView::Side).is_none());
        assert!(silhouettes.get_view(OrthogonalView::Top).is_none());
    }
}
