# GeoTIFF Tile Service

Generate XYZ tiles from GeoTIFF files with GPU acceleration.

This service provides high-performance tile generation from GeoTIFF files using GPU acceleration via CuPy. It supports generating tiles in the XYZ (Slippy Map) tile format for use with web mapping libraries like Leaflet, OpenLayers, and Google Maps.

## Features

- **GPU Acceleration**: Uses CuPy for GPU-accelerated image processing and resampling
- **XYZ Tile Format**: Generates tiles in the standard XYZ/Slippy Map tile naming convention
- **Multi-zoom Support**: Generate tiles for multiple zoom levels in one run
- **Automatic Reprojection**: Handles coordinate transformation from any projection to WGS84
- **Efficient Processing**: Only generates tiles that intersect with the GeoTIFF bounds
- **Flexible Configuration**: Customizable tile size and output directory

## Requirements

- Python 3.7+
- GDAL 3.0+
- CUDA-capable GPU (for GPU acceleration)
- CuPy (for GPU acceleration)

## Installation

1. Install system dependencies:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev python3-gdal

# macOS (using Homebrew)
brew install gdal
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

**Note**: If you don't have a CUDA-capable GPU or CuPy installed, the service will automatically fall back to CPU processing.

## Usage

### Basic Usage

Generate tiles for zoom levels 10-12:

```bash
python main.py input.tif -z 10 11 12
```

### Advanced Usage

```bash
# Specify output directory
python main.py input.tif -z 10 11 12 -o my_tiles/

# Use custom tile size (512x512 instead of default 256x256)
python main.py input.tif -z 8 9 10 --tile-size 512

# Disable GPU acceleration
python main.py input.tif -z 5 --no-gpu

# Enable verbose logging
python main.py input.tif -z 10 -v
```

### Command-Line Options

```
positional arguments:
  input                 Path to input GeoTIFF file

optional arguments:
  -h, --help            Show help message and exit
  -z, --zoom ZOOM [ZOOM ...]
                        Zoom level(s) to generate (e.g., -z 10 11 12)
  -o, --output OUTPUT   Output directory for tiles (default: tiles)
  --tile-size TILE_SIZE
                        Tile size in pixels (default: 256)
  --no-gpu              Disable GPU acceleration
  -v, --verbose         Enable verbose logging
```

## Output Structure

Generated tiles follow the XYZ tile naming convention:

```
output_dir/
  ├── {z}/
  │   ├── {x}/
  │   │   ├── {y}.png
  │   │   ├── {y}.png
  │   │   └── ...
  │   └── ...
  └── ...
```

Where:
- `z` is the zoom level
- `x` is the tile column
- `y` is the tile row

## Using Tiles in Web Maps

### Leaflet Example

```javascript
var map = L.map('map').setView([51.505, -0.09], 13);

L.tileLayer('tiles/{z}/{x}/{y}.png', {
    attribution: 'My GeoTIFF Tiles',
    maxZoom: 18,
}).addTo(map);
```

### OpenLayers Example

```javascript
var map = new ol.Map({
  target: 'map',
  layers: [
    new ol.layer.Tile({
      source: new ol.source.XYZ({
        url: 'tiles/{z}/{x}/{y}.png'
      })
    })
  ],
  view: new ol.View({
    center: ol.proj.fromLonLat([-0.09, 51.505]),
    zoom: 13
  })
});
```

## How It Works

1. **Load GeoTIFF**: Opens the GeoTIFF file using GDAL
2. **Calculate Bounds**: Determines the geographic extent in WGS84
3. **Tile Calculation**: For each zoom level, calculates which tiles intersect with the data
4. **GPU Processing**: 
   - Reads data from the GeoTIFF for each tile
   - Transfers data to GPU (if available)
   - Performs normalization and resampling on GPU
   - Transfers result back to CPU
5. **Save Tiles**: Saves tiles as PNG files in XYZ directory structure

## GPU vs CPU Performance

GPU acceleration can provide significant speedup, especially for:
- Large GeoTIFF files
- High zoom levels (many tiles)
- Complex resampling operations

Typical speedups: 3-10x faster with GPU acceleration

## Troubleshooting

### CuPy Installation Issues

If you have trouble installing CuPy, make sure you have:
- CUDA toolkit installed
- Compatible CuPy version for your CUDA version

See [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) for details.

### GDAL Import Errors

If you encounter GDAL import errors:
```bash
# Set GDAL environment variables
export GDAL_DATA=/usr/share/gdal
export PROJ_LIB=/usr/share/proj
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
