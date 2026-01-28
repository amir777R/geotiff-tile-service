#!/usr/bin/env python3
"""
Create a sample GeoTIFF file for testing the tile generator.
"""

import numpy as np
from osgeo import gdal, osr
import sys


def create_sample_geotiff(filename='sample.tif', width=1000, height=1000):
    """
    Create a sample GeoTIFF file for testing.
    
    Args:
        filename: Output filename
        width: Image width in pixels
        height: Image height in pixels
    """
    # Create a simple gradient pattern
    print(f"Creating sample GeoTIFF: {filename} ({width}x{height} pixels)")
    
    # Create data - a gradient from 0 to 255
    data = np.zeros((height, width), dtype=np.uint8)
    
    # Create a pattern - vertical gradient + horizontal gradient
    for i in range(height):
        for j in range(width):
            data[i, j] = (i * 255 // height + j * 255 // width) // 2
    
    # Add some variation - circles
    center_y, center_x = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    
    # Create three circles
    for radius, intensity in [(100, 255), (200, 150), (300, 50)]:
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        data[mask] = intensity
    
    # Create driver
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(filename, width, height, 3, gdal.GDT_Byte)
    
    if dataset is None:
        print(f"Error: Could not create {filename}")
        return False
    
    # Define geotransform
    # Coverage: approximately San Francisco Bay Area
    # [min_lon, max_lon, min_lat, max_lat] ≈ [-122.5, -121.5, 37.3, 38.3]
    min_lon, max_lon = -122.5, -121.5
    min_lat, max_lat = 37.3, 38.3
    
    pixel_width = (max_lon - min_lon) / width
    pixel_height = (max_lat - min_lat) / height
    
    # Geotransform: (top_left_x, pixel_width, 0, top_left_y, 0, -pixel_height)
    geotransform = (min_lon, pixel_width, 0, max_lat, 0, -pixel_height)
    dataset.SetGeoTransform(geotransform)
    
    # Set projection to WGS84
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    
    # Write data to all three bands (RGB)
    for band_idx in range(1, 4):
        band = dataset.GetRasterBand(band_idx)
        
        # Slightly different data for each band
        if band_idx == 1:  # Red
            band_data = data
        elif band_idx == 2:  # Green
            band_data = np.roll(data, width // 3, axis=1)
        else:  # Blue
            band_data = np.roll(data, -width // 3, axis=1)
        
        band.WriteArray(band_data)
        band.FlushCache()
    
    # Close dataset
    dataset = None
    
    print(f"✓ Sample GeoTIFF created successfully: {filename}")
    print(f"  - Coverage: San Francisco Bay Area")
    print(f"  - Bounds: {min_lon:.2f}, {min_lat:.2f} to {max_lon:.2f}, {max_lat:.2f}")
    print(f"  - Size: {width}x{height} pixels")
    print(f"  - Bands: 3 (RGB)")
    
    return True


if __name__ == '__main__':
    filename = 'sample.tif' if len(sys.argv) < 2 else sys.argv[1]
    
    success = create_sample_geotiff(filename)
    
    if success:
        print(f"\nYou can now test the tile generator with:")
        print(f"  python3 main.py {filename} -z 8 9 10")
        sys.exit(0)
    else:
        sys.exit(1)
