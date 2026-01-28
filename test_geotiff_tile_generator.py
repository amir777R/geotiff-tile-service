#!/usr/bin/env python3
"""
Unit tests for the GeoTIFF Tile Generator.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
from osgeo import gdal, osr
from geotiff_tile_generator import GeoTIFFTileGenerator


class TestGeoTIFFTileGenerator(unittest.TestCase):
    """Test cases for GeoTIFFTileGenerator class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - create a sample GeoTIFF."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_tif = os.path.join(cls.temp_dir, 'test.tif')
        cls.output_dir = os.path.join(cls.temp_dir, 'tiles')
        
        # Create a simple test GeoTIFF
        width, height = 500, 500
        data = np.zeros((height, width), dtype=np.uint8)
        
        # Create a simple pattern
        for i in range(height):
            for j in range(width):
                data[i, j] = (i + j) % 256
        
        # Create GeoTIFF
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(cls.test_tif, width, height, 1, gdal.GDT_Byte)
        
        # Set geotransform (San Francisco area)
        min_lon, max_lon = -122.5, -121.5
        min_lat, max_lat = 37.3, 38.3
        pixel_width = (max_lon - min_lon) / width
        pixel_height = (max_lat - min_lat) / height
        geotransform = (min_lon, pixel_width, 0, max_lat, 0, -pixel_height)
        dataset.SetGeoTransform(geotransform)
        
        # Set WGS84 projection
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        dataset.SetProjection(srs.ExportToWkt())
        
        # Write data
        band = dataset.GetRasterBand(1)
        band.WriteArray(data)
        band.FlushCache()
        
        dataset = None
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Set up each test."""
        # Clean output directory
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
    
    def test_initialization(self):
        """Test GeoTIFFTileGenerator initialization."""
        generator = GeoTIFFTileGenerator(
            self.test_tif,
            self.output_dir,
            use_gpu=False
        )
        
        self.assertIsNotNone(generator.dataset)
        self.assertEqual(generator.tile_size, 256)
        self.assertFalse(generator.use_gpu)
        
        # Check bounds were calculated
        self.assertIn('min_lon', generator.bounds)
        self.assertIn('max_lon', generator.bounds)
        self.assertIn('min_lat', generator.bounds)
        self.assertIn('max_lat', generator.bounds)
        
        generator.close()
    
    def test_latlon_to_tile(self):
        """Test latitude/longitude to tile coordinate conversion."""
        generator = GeoTIFFTileGenerator(
            self.test_tif,
            self.output_dir,
            use_gpu=False
        )
        
        # Test known coordinates
        # San Francisco center approximately
        x, y = generator._latlon_to_tile(37.7749, -122.4194, 10)
        
        # Verify tile coordinates are reasonable
        self.assertIsInstance(x, int)
        self.assertIsInstance(y, int)
        self.assertGreater(x, 0)
        self.assertGreater(y, 0)
        self.assertLess(x, 2**10)
        self.assertLess(y, 2**10)
        
        generator.close()
    
    def test_tile_to_latlon(self):
        """Test tile coordinate to latitude/longitude conversion."""
        generator = GeoTIFFTileGenerator(
            self.test_tif,
            self.output_dir,
            use_gpu=False
        )
        
        bounds = generator._tile_to_latlon(163, 395, 10)
        
        # Verify bounds structure
        self.assertIn('min_lon', bounds)
        self.assertIn('max_lon', bounds)
        self.assertIn('min_lat', bounds)
        self.assertIn('max_lat', bounds)
        
        # Verify bounds are reasonable
        self.assertLess(bounds['min_lon'], bounds['max_lon'])
        self.assertLess(bounds['min_lat'], bounds['max_lat'])
        self.assertGreater(bounds['min_lon'], -180)
        self.assertLess(bounds['max_lon'], 180)
        self.assertGreater(bounds['min_lat'], -90)
        self.assertLess(bounds['max_lat'], 90)
        
        generator.close()
    
    def test_generate_single_tile(self):
        """Test generating a single tile."""
        generator = GeoTIFFTileGenerator(
            self.test_tif,
            self.output_dir,
            use_gpu=False
        )
        
        # Generate a tile that should intersect with our test data
        img = generator.generate_tile(40, 98, 8)
        
        # Verify image was generated
        self.assertIsNotNone(img)
        
        # Verify image dimensions
        self.assertEqual(img.size, (256, 256))
        
        generator.close()
    
    def test_save_tile(self):
        """Test saving a tile to disk."""
        generator = GeoTIFFTileGenerator(
            self.test_tif,
            self.output_dir,
            use_gpu=False
        )
        
        # Save a tile
        tile_path = generator.save_tile(40, 98, 8)
        
        # Verify tile was saved
        self.assertIsNotNone(tile_path)
        self.assertTrue(os.path.exists(tile_path))
        
        # Verify directory structure
        self.assertTrue(tile_path.endswith('8/40/98.png'))
        
        generator.close()
    
    def test_generate_tiles_single_zoom(self):
        """Test generating tiles for a single zoom level."""
        generator = GeoTIFFTileGenerator(
            self.test_tif,
            self.output_dir,
            use_gpu=False
        )
        
        stats = generator.generate_tiles(zoom_levels=8)
        
        # Verify statistics
        self.assertIn('total_tiles', stats)
        self.assertIn('generated_tiles', stats)
        self.assertIn('empty_tiles', stats)
        
        # Verify some tiles were generated
        self.assertGreater(stats['total_tiles'], 0)
        
        generator.close()
    
    def test_generate_tiles_multiple_zooms(self):
        """Test generating tiles for multiple zoom levels."""
        generator = GeoTIFFTileGenerator(
            self.test_tif,
            self.output_dir,
            use_gpu=False
        )
        
        stats = generator.generate_tiles(zoom_levels=[7, 8])
        
        # Verify tiles were generated for multiple zoom levels
        self.assertGreater(stats['total_tiles'], 0)
        
        # Check that both zoom level directories exist
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, '7')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, '8')))
        
        generator.close()
    
    def test_custom_tile_size(self):
        """Test generating tiles with custom size."""
        generator = GeoTIFFTileGenerator(
            self.test_tif,
            self.output_dir,
            tile_size=512,
            use_gpu=False
        )
        
        img = generator.generate_tile(40, 98, 8)
        
        if img:
            # Verify custom tile size
            self.assertEqual(img.size, (512, 512))
        
        generator.close()


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
