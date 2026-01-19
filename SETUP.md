# Setup Instructions

## System Dependencies

This project requires OpenCV 4.x and LLVM to be installed on your system.

### macOS (Homebrew)

```bash
# Install OpenCV
brew install opencv

# Install LLVM (for libclang, required for Rust bindings)
brew install llvm

# Set environment variables for the build
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/Cellar/llvm/20.1.8/lib
export PKG_CONFIG_PATH=/opt/homebrew/opt/opencv/lib/pkgconfig
```

### Linux (Ubuntu/Debian)

```bash
# Install OpenCV development files
sudo apt-get install libopencv-dev clang libclang-dev

# Verify installation
pkg-config --modversion opencv4
```

## Building the Project

```bash
# Build the library
cargo build

# Run tests
cargo test

# Run example
cargo run --example extract_silhouette
```

## Troubleshooting

### libclang.dylib not found

If you see errors about `libclang.dylib` not being found:

```bash
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/Cellar/llvm/$(ls /opt/homebrew/Cellar/llvm | sort -V | tail -n1)/lib
```

### OpenCV not found

If the build can't find OpenCV:

```bash
# Check OpenCV installation
brew list opencv

# Set PKG_CONFIG_PATH
export PKG_CONFIG_PATH=$(brew --prefix opencv)/lib/pkgconfig
```

### CMake errors

Ensure you have CMake installed:

```bash
brew install cmake
```

## Development

The module is structured as follows:

- `src/lib.rs` - Main library code with image preprocessing functions
- `examples/` - Example usage programs
- `tests/` - Integration tests (require actual image files)

## CI/CD Note

For automated builds, you'll need to ensure OpenCV and LLVM are installed in your CI environment. Consider using pre-built Docker images with OpenCV installed for faster builds.
