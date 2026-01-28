#!/usr/bin/env python3
"""
Example script demonstrating how to use the GeoTIFF Tile Generator
as a Python library.
"""

from geotiff_tile_generator import GeoTIFFTileGenerator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Basic example: Generate tiles for a single zoom level."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Initialize generator
    generator = GeoTIFFTileGenerator(
        geotiff_path='path/to/your/input.tif',
        output_dir='output/tiles',
        use_gpu=True
    )
    
    # Generate tiles for zoom level 10
    stats = generator.generate_tiles(zoom_levels=10)
    
    print(f"\nGenerated {stats['generated_tiles']} tiles")
    
    # Clean up
    generator.close()


def example_multiple_zoom_levels():
    """Example: Generate tiles for multiple zoom levels."""
    print("\n" + "=" * 60)
    print("Example 2: Multiple Zoom Levels")
    print("=" * 60)
    
    # Initialize generator
    generator = GeoTIFFTileGenerator(
        geotiff_path='path/to/your/input.tif',
        output_dir='output/tiles',
        use_gpu=True
    )
    
    # Generate tiles for zoom levels 8, 9, and 10
    stats = generator.generate_tiles(zoom_levels=[8, 9, 10])
    
    print(f"\nTotal tiles processed: {stats['total_tiles']}")
    print(f"Tiles generated: {stats['generated_tiles']}")
    print(f"Empty tiles skipped: {stats['empty_tiles']}")
    
    # Clean up
    generator.close()


def example_custom_tile_size():
    """Example: Generate tiles with custom size."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Tile Size")
    print("=" * 60)
    
    # Initialize generator with 512x512 tiles
    generator = GeoTIFFTileGenerator(
        geotiff_path='path/to/your/input.tif',
        output_dir='output/tiles_512',
        tile_size=512,
        use_gpu=True
    )
    
    # Generate tiles
    stats = generator.generate_tiles(zoom_levels=8)
    
    print(f"\nGenerated {stats['generated_tiles']} tiles (512x512 pixels)")
    
    # Clean up
    generator.close()


def example_single_tile():
    """Example: Generate a single specific tile."""
    print("\n" + "=" * 60)
    print("Example 4: Generate Single Tile")
    print("=" * 60)
    
    # Initialize generator
    generator = GeoTIFFTileGenerator(
        geotiff_path='path/to/your/input.tif',
        output_dir='output/single_tile',
        use_gpu=True
    )
    
    # Generate a specific tile
    x, y, zoom = 512, 342, 10  # Example tile coordinates
    
    img = generator.generate_tile(x, y, zoom)
    
    if img:
        tile_path = generator.save_tile(x, y, zoom, img)
        print(f"\nTile saved to: {tile_path}")
    else:
        print(f"\nTile {x}/{y} at zoom {zoom} is outside bounds")
    
    # Clean up
    generator.close()


def example_cpu_only():
    """Example: Use CPU processing instead of GPU."""
    print("\n" + "=" * 60)
    print("Example 5: CPU-only Processing")
    print("=" * 60)
    
    # Initialize generator with GPU disabled
    generator = GeoTIFFTileGenerator(
        geotiff_path='path/to/your/input.tif',
        output_dir='output/tiles_cpu',
        use_gpu=False  # Disable GPU
    )
    
    # Generate tiles
    stats = generator.generate_tiles(zoom_levels=8)
    
    print(f"\nGenerated {stats['generated_tiles']} tiles using CPU")
    
    # Clean up
    generator.close()


if __name__ == '__main__':
    print("\nGeoTIFF Tile Generator - Usage Examples")
    print("=" * 60)
    print("\nNote: Update the 'geotiff_path' in each example to point")
    print("to your actual GeoTIFF file before running.")
    print()
    
    # Uncomment the examples you want to run:
    
    # example_basic_usage()
    # example_multiple_zoom_levels()
    # example_custom_tile_size()
    # example_single_tile()
    # example_cpu_only()
    
    print("\nTo run these examples, uncomment the function calls above")
    print("and update the GeoTIFF file paths.")
