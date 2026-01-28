#!/usr/bin/env python3
"""
Main entry point for the GeoTIFF Tile Service.

Command-line interface for generating XYZ tiles from GeoTIFF files
with GPU acceleration.
"""

import argparse
import sys
import logging
from geotiff_tile_generator import GeoTIFFTileGenerator


def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(
        description='Generate XYZ tiles from GeoTIFF files with GPU acceleration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate tiles for zoom levels 10-12
  python main.py input.tif -z 10 11 12 -o tiles/
  
  # Generate tiles for zoom level 5 without GPU
  python main.py input.tif -z 5 --no-gpu
  
  # Generate tiles with custom tile size
  python main.py input.tif -z 8 9 10 --tile-size 512
        """
    )
    
    parser.add_argument(
        'input',
        help='Path to input GeoTIFF file'
    )
    
    parser.add_argument(
        '-z', '--zoom',
        nargs='+',
        type=int,
        required=True,
        help='Zoom level(s) to generate (e.g., -z 10 11 12)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='tiles',
        help='Output directory for tiles (default: tiles)'
    )
    
    parser.add_argument(
        '--tile-size',
        type=int,
        default=256,
        help='Tile size in pixels (default: 256)'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging (force reconfiguration to override module-level config)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    # Clear existing handlers to allow reconfiguration (Python 3.7 compatible)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize tile generator
        logger.info(f"Initializing tile generator for: {args.input}")
        generator = GeoTIFFTileGenerator(
            geotiff_path=args.input,
            output_dir=args.output,
            tile_size=args.tile_size,
            use_gpu=not args.no_gpu
        )
        
        # Generate tiles
        logger.info(f"Generating tiles for zoom levels: {args.zoom}")
        stats = generator.generate_tiles(args.zoom)
        
        # Print statistics
        logger.info("=" * 50)
        logger.info("Tile generation complete!")
        logger.info(f"Total tiles processed: {stats['total_tiles']}")
        logger.info(f"Tiles generated: {stats['generated_tiles']}")
        logger.info(f"Empty tiles skipped: {stats['empty_tiles']}")
        logger.info(f"Output directory: {args.output}")
        logger.info("=" * 50)
        
        # Close generator
        generator.close()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        return 1


if __name__ == '__main__':
    sys.exit(main())
