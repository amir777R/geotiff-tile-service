#!/usr/bin/env python3
"""
GeoTIFF Tile Service - GPU-accelerated XYZ tile generation from GeoTIFF files.

This module provides functionality to generate XYZ tiles from GeoTIFF files
using GPU acceleration with CuPy for faster processing.
"""

import os
import math
import numpy as np
from osgeo import gdal, osr
from PIL import Image
import logging

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Configure logging only if not already configured
if not logging.root.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# Log GPU availability
if GPU_AVAILABLE:
    logger.info("GPU acceleration available via CuPy")
else:
    logger.info("CuPy not available, falling back to CPU processing")


class GeoTIFFTileGenerator:
    """
    Generate XYZ tiles from GeoTIFF files with GPU acceleration.
    
    XYZ tiles follow the Slippy Map tile naming convention:
    - z: zoom level
    - x: tile column
    - y: tile row
    """
    
    def __init__(self, geotiff_path, output_dir='tiles', tile_size=256, use_gpu=True):
        """
        Initialize the tile generator.
        
        Args:
            geotiff_path: Path to the input GeoTIFF file
            output_dir: Directory to save generated tiles
            tile_size: Size of each tile in pixels (default: 256)
            use_gpu: Whether to use GPU acceleration if available
        """
        self.geotiff_path = geotiff_path
        self.output_dir = output_dir
        self.tile_size = tile_size
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        # Load GeoTIFF
        self.dataset = gdal.Open(geotiff_path, gdal.GA_ReadOnly)
        if self.dataset is None:
            raise ValueError(f"Failed to open GeoTIFF file: {geotiff_path}")
        
        # Get geotransform and projection
        self.geotransform = self.dataset.GetGeoTransform()
        self.projection = self.dataset.GetProjection()
        
        # Create coordinate transformation to WGS84 (EPSG:4326)
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(self.projection)
        
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(4326)
        
        self.transform_to_wgs84 = osr.CoordinateTransformation(src_srs, dst_srs)
        
        # Get bounds in WGS84
        self._calculate_bounds()
        
        logger.info(f"Initialized GeoTIFF Tile Generator")
        logger.info(f"GPU acceleration: {'enabled' if self.use_gpu else 'disabled'}")
        logger.info(f"Image size: {self.dataset.RasterXSize} x {self.dataset.RasterYSize}")
        logger.info(f"Bounds (WGS84): {self.bounds}")
    
    def _calculate_bounds(self):
        """Calculate the bounds of the GeoTIFF in WGS84 coordinates."""
        width = self.dataset.RasterXSize
        height = self.dataset.RasterYSize
        
        # Get corners in pixel coordinates
        corners_pixel = [
            (0, 0),
            (width, 0),
            (width, height),
            (0, height)
        ]
        
        # Transform to geo coordinates
        corners_geo = []
        for x, y in corners_pixel:
            geo_x = self.geotransform[0] + x * self.geotransform[1] + y * self.geotransform[2]
            geo_y = self.geotransform[3] + x * self.geotransform[4] + y * self.geotransform[5]
            
            # Transform to WGS84
            lon, lat, _ = self.transform_to_wgs84.TransformPoint(geo_x, geo_y)
            corners_geo.append((lon, lat))
        
        # Calculate bounds
        lons = [c[0] for c in corners_geo]
        lats = [c[1] for c in corners_geo]
        
        self.bounds = {
            'min_lon': min(lons),
            'max_lon': max(lons),
            'min_lat': min(lats),
            'max_lat': max(lats)
        }
    
    def _latlon_to_tile(self, lat, lon, zoom):
        """
        Convert latitude/longitude to tile coordinates at given zoom level.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            zoom: Zoom level
            
        Returns:
            Tuple of (tile_x, tile_y)
        """
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        
        return (x, y)
    
    def _tile_to_latlon(self, x, y, zoom):
        """
        Convert tile coordinates to latitude/longitude bounds.
        
        Args:
            x: Tile X coordinate
            y: Tile Y coordinate
            zoom: Zoom level
            
        Returns:
            Dictionary with min/max lat/lon bounds
        """
        n = 2.0 ** zoom
        
        lon_min = x / n * 360.0 - 180.0
        lon_max = (x + 1) / n * 360.0 - 180.0
        
        lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
        lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
        
        return {
            'min_lon': lon_min,
            'max_lon': lon_max,
            'min_lat': lat_min,
            'max_lat': lat_max
        }
    
    def _geo_to_pixel(self, lon, lat):
        """
        Convert geographic coordinates to pixel coordinates in the GeoTIFF.
        
        Args:
            lon: Longitude in WGS84
            lat: Latitude in WGS84
            
        Returns:
            Tuple of (pixel_x, pixel_y)
        """
        # Transform from WGS84 to source projection
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(self.projection)
        
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(4326)
        
        transform_from_wgs84 = osr.CoordinateTransformation(dst_srs, src_srs)
        geo_x, geo_y, _ = transform_from_wgs84.TransformPoint(lon, lat)
        
        # Apply inverse geotransform
        det = self.geotransform[1] * self.geotransform[5] - self.geotransform[2] * self.geotransform[4]
        
        pixel_x = (self.geotransform[5] * (geo_x - self.geotransform[0]) - 
                   self.geotransform[2] * (geo_y - self.geotransform[3])) / det
        pixel_y = (-self.geotransform[4] * (geo_x - self.geotransform[0]) + 
                   self.geotransform[1] * (geo_y - self.geotransform[3])) / det
        
        return (pixel_x, pixel_y)
    
    def _read_tile_data(self, tile_bounds):
        """
        Read and resample data from GeoTIFF for the given tile bounds.
        
        Args:
            tile_bounds: Dictionary with min/max lat/lon bounds
            
        Returns:
            NumPy array with tile data, or None if no data
        """
        # Convert tile bounds to pixel coordinates
        min_px, max_py = self._geo_to_pixel(tile_bounds['min_lon'], tile_bounds['min_lat'])
        max_px, min_py = self._geo_to_pixel(tile_bounds['max_lon'], tile_bounds['max_lat'])
        
        # Ensure coordinates are within bounds
        min_px = max(0, min(self.dataset.RasterXSize - 1, min_px))
        max_px = max(0, min(self.dataset.RasterXSize - 1, max_px))
        min_py = max(0, min(self.dataset.RasterYSize - 1, min_py))
        max_py = max(0, min(self.dataset.RasterYSize - 1, max_py))
        
        # Check if tile intersects with data
        if min_px >= max_px or min_py >= max_py:
            return None
        
        # Calculate read window
        x_off = int(min_px)
        y_off = int(min_py)
        x_size = int(max_px - min_px) + 1
        y_size = int(max_py - min_py) + 1
        
        # Ensure we don't read outside bounds
        if x_off + x_size > self.dataset.RasterXSize:
            x_size = self.dataset.RasterXSize - x_off
        if y_off + y_size > self.dataset.RasterYSize:
            y_size = self.dataset.RasterYSize - y_off
        
        if x_size <= 0 or y_size <= 0:
            return None
        
        # Read data from all bands
        num_bands = self.dataset.RasterCount
        tile_data = np.zeros((self.tile_size, self.tile_size, min(num_bands, 4)), dtype=np.uint8)
        
        for band_idx in range(min(num_bands, 4)):
            band = self.dataset.GetRasterBand(band_idx + 1)
            
            # Read data with resampling
            data = band.ReadAsArray(
                x_off, y_off, x_size, y_size,
                buf_xsize=self.tile_size,
                buf_ysize=self.tile_size,
                resample_alg=gdal.GRIORA_Bilinear
            )
            
            if data is None:
                continue
            
            # GPU-accelerated processing if available
            if self.use_gpu:
                # Transfer to GPU
                data_gpu = cp.asarray(data)
                
                # Normalize to 0-255 range
                data_min = cp.min(data_gpu)
                data_max = cp.max(data_gpu)
                if data_max > data_min:
                    data_gpu = ((data_gpu - data_min) / (data_max - data_min) * 255).astype(cp.uint8)
                else:
                    data_gpu = cp.zeros_like(data_gpu, dtype=cp.uint8)
                
                # Transfer back to CPU
                tile_data[:, :, band_idx] = cp.asnumpy(data_gpu)
            else:
                # CPU processing
                data_min = np.min(data)
                data_max = np.max(data)
                if data_max > data_min:
                    data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
                else:
                    data = np.zeros_like(data, dtype=np.uint8)
                
                tile_data[:, :, band_idx] = data
        
        # If single band, convert to RGB
        if num_bands == 1:
            tile_data = np.stack([tile_data[:, :, 0]] * 3, axis=2)
        
        return tile_data
    
    def generate_tile(self, x, y, zoom):
        """
        Generate a single XYZ tile.
        
        Args:
            x: Tile X coordinate
            y: Tile Y coordinate
            zoom: Zoom level
            
        Returns:
            PIL Image object or None if tile is outside bounds
        """
        # Get tile bounds
        tile_bounds = self._tile_to_latlon(x, y, zoom)
        
        # Check if tile intersects with GeoTIFF bounds
        if (tile_bounds['max_lon'] < self.bounds['min_lon'] or
            tile_bounds['min_lon'] > self.bounds['max_lon'] or
            tile_bounds['max_lat'] < self.bounds['min_lat'] or
            tile_bounds['min_lat'] > self.bounds['max_lat']):
            return None
        
        # Read tile data
        tile_data = self._read_tile_data(tile_bounds)
        
        if tile_data is None:
            return None
        
        # Convert to PIL Image
        if tile_data.shape[2] == 3:
            img = Image.fromarray(tile_data, 'RGB')
        elif tile_data.shape[2] == 4:
            img = Image.fromarray(tile_data, 'RGBA')
        else:
            img = Image.fromarray(tile_data[:, :, 0], 'L')
        
        return img
    
    def save_tile(self, x, y, zoom, img=None):
        """
        Generate and save a single tile to disk.
        
        Args:
            x: Tile X coordinate
            y: Tile Y coordinate
            zoom: Zoom level
            img: Optional pre-generated image, if None will generate
            
        Returns:
            Path to saved tile or None if tile is empty
        """
        if img is None:
            img = self.generate_tile(x, y, zoom)
        
        if img is None:
            return None
        
        # Create directory structure
        tile_dir = os.path.join(self.output_dir, str(zoom), str(x))
        os.makedirs(tile_dir, exist_ok=True)
        
        # Save tile
        tile_path = os.path.join(tile_dir, f"{y}.png")
        img.save(tile_path, 'PNG')
        
        return tile_path
    
    def generate_tiles(self, zoom_levels):
        """
        Generate tiles for specified zoom levels.
        
        Args:
            zoom_levels: List of zoom levels or single zoom level
            
        Returns:
            Dictionary with statistics about generated tiles
        """
        if isinstance(zoom_levels, int):
            zoom_levels = [zoom_levels]
        
        stats = {
            'total_tiles': 0,
            'generated_tiles': 0,
            'empty_tiles': 0
        }
        
        for zoom in zoom_levels:
            logger.info(f"Generating tiles for zoom level {zoom}...")
            
            # Calculate tile range for this zoom level
            min_x, min_y = self._latlon_to_tile(
                self.bounds['max_lat'], self.bounds['min_lon'], zoom
            )
            max_x, max_y = self._latlon_to_tile(
                self.bounds['min_lat'], self.bounds['max_lon'], zoom
            )
            
            # Generate tiles
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    stats['total_tiles'] += 1
                    
                    tile_path = self.save_tile(x, y, zoom)
                    
                    if tile_path:
                        stats['generated_tiles'] += 1
                        if stats['generated_tiles'] % 100 == 0:
                            logger.info(f"Generated {stats['generated_tiles']} tiles...")
                    else:
                        stats['empty_tiles'] += 1
            
            logger.info(f"Completed zoom level {zoom}: {stats['generated_tiles']} tiles generated")
        
        return stats
    
    def close(self):
        """Close the GeoTIFF dataset."""
        if self.dataset:
            self.dataset = None
