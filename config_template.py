# Configuration Template for GeoTIFF Tile Service
#
# Copy this file to config.py and customize as needed

# Input GeoTIFF file path
INPUT_GEOTIFF = "path/to/your/input.tif"

# Output directory for generated tiles
OUTPUT_DIR = "tiles"

# Tile size in pixels (256, 512, etc.)
TILE_SIZE = 256

# Zoom levels to generate (list of integers)
ZOOM_LEVELS = [8, 9, 10, 11, 12]

# Enable GPU acceleration (True/False)
USE_GPU = True

# Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
LOG_LEVEL = 'INFO'
