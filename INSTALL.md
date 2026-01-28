# Installation Guide

## Prerequisites

Before installing the GeoTIFF Tile Service, ensure you have the following prerequisites:

### System Requirements

- Python 3.7 or higher
- GDAL 3.0 or higher
- (Optional) CUDA-capable GPU for GPU acceleration
- (Optional) CUDA Toolkit 11.x or higher for GPU support

### Operating System

The service has been tested on:
- Ubuntu 20.04+
- macOS 10.15+
- Windows 10+ (with WSL2 recommended)

## Installation Steps

### 1. Install System Dependencies

#### Ubuntu/Debian

```bash
# Update package list
sudo apt-get update

# Install GDAL
sudo apt-get install -y gdal-bin libgdal-dev python3-gdal

# Verify installation
gdalinfo --version
```

#### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install GDAL
brew install gdal

# Verify installation
gdalinfo --version
```

#### Windows (WSL2)

```bash
# Install WSL2 and Ubuntu from Microsoft Store first
# Then follow Ubuntu/Debian instructions above
```

### 2. Clone the Repository

```bash
git clone https://github.com/amir777R/geotiff-tile-service.git
cd geotiff-tile-service
```

### 3. Install Python Dependencies

```bash
# Basic installation (CPU only)
pip install -r requirements.txt
```

**Note:** If you encounter errors installing GDAL via pip, you may need to set environment variables:

```bash
export GDAL_CONFIG=/usr/bin/gdal-config
pip install GDAL==$(gdal-config --version)
pip install numpy Pillow
```

### 4. (Optional) Install GPU Support

If you have a CUDA-capable GPU and want to enable GPU acceleration:

#### Install CUDA Toolkit

Follow the official NVIDIA guide: https://developer.nvidia.com/cuda-downloads

#### Install CuPy

```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x

# Verify installation
python -c "import cupy; print('CuPy version:', cupy.__version__)"
```

**Note:** CuPy installation can take several minutes as it may need to compile CUDA kernels.

### 5. Verify Installation

Run the test suite to verify everything is working:

```bash
python -m unittest test_geotiff_tile_generator -v
```

All tests should pass.

## Quick Start

### 1. Create a Sample GeoTIFF

```bash
python create_sample.py
```

This creates a `sample.tif` file for testing.

### 2. Generate Tiles

```bash
python main.py sample.tif -z 8 9 10 -o tiles
```

### 3. View Tiles

Start a local web server:

```bash
python -m http.server 8000
```

Open your browser to: http://localhost:8000/viewer.html

## Troubleshooting

### GDAL Import Errors

If you see `ImportError: No module named 'osgeo'`:

1. Verify GDAL is installed:
   ```bash
   gdalinfo --version
   ```

2. Set environment variables:
   ```bash
   export GDAL_DATA=/usr/share/gdal
   export PROJ_LIB=/usr/share/proj
   ```

3. Reinstall with correct version:
   ```bash
   pip uninstall GDAL
   pip install GDAL==$(gdal-config --version)
   ```

### NumPy Version Conflicts

If you see NumPy-related errors:

```bash
pip install 'numpy>=1.19.0,<2.0.0'
```

### CuPy Installation Issues

If CuPy installation fails:

1. Ensure CUDA is installed:
   ```bash
   nvcc --version
   ```

2. Match CuPy version to your CUDA version:
   ```bash
   # Check CUDA version
   nvcc --version
   
   # Install matching CuPy
   pip install cupy-cuda11x  # for CUDA 11.x
   # OR
   pip install cupy-cuda12x  # for CUDA 12.x
   ```

3. If still failing, use CPU-only mode with `--no-gpu` flag

### Memory Issues

For large GeoTIFF files:

1. Process one zoom level at a time:
   ```bash
   python main.py input.tif -z 8
   python main.py input.tif -z 9
   python main.py input.tif -z 10
   ```

2. Reduce tile size:
   ```bash
   python main.py input.tif -z 8 --tile-size 128
   ```

## Next Steps

- Read the [README.md](README.md) for usage examples
- Check [examples.py](examples.py) for library usage
- See [viewer.html](viewer.html) for web map integration

## Getting Help

If you encounter issues:

1. Check existing GitHub issues
2. Create a new issue with:
   - Your operating system
   - Python version (`python --version`)
   - GDAL version (`gdalinfo --version`)
   - Error messages
   - Steps to reproduce
