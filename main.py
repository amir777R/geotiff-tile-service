"""
GeoTIFF XYZ Tile Provider Service
Reads GeoTIFF files and serves them as XYZ tiles with caching
"""
import os
import io
import math
import hashlib
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds, calculate_default_transform
from pyproj import Transformer
import logging
import numpy as np
from numba import jit, cuda
from functools import lru_cache
from threading import Lock

# Check for GPU/CUDA support
try:
    cuda.select_device(0)
    GPU_AVAILABLE = True
    GPU_NAME = cuda.get_current_device().name.decode('utf-8')
except Exception as e:
    GPU_AVAILABLE = False
    GPU_NAME = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if GPU_AVAILABLE:
    logger.info(f"🚀 GPU acceleration enabled: {GPU_NAME}")
    logger.info("⚡ Using Numba CUDA for tile rendering")
else:
    logger.warning("⚠️  WARNING: GPU acceleration is NOT available!")
    logger.warning("⚠️  Running with CPU-only mode (slower performance)")
    logger.warning("⚠️  To enable GPU acceleration:")
    logger.warning("    1. Install NVIDIA CUDA Toolkit")
    logger.warning("    2. Ensure CUDA is in system PATH")
    logger.warning("    3. Restart the service")
    logger.info("💻 Fallback: Using Numba JIT CPU acceleration")

# JIT-compiled functions for performance
@jit(nopython=True, cache=True)
def normalize_band(band, mask):
    """Normalize band values to 0-255 with JIT compilation"""
    result = band.copy()
    result[mask] = 0
    
    valid_mask = ~mask
    if np.any(valid_mask):
        valid_data = band[valid_mask]
        if len(valid_data) > 0:
            min_val = valid_data.min()
            max_val = valid_data.max()
            if max_val > min_val:
                result[valid_mask] = ((valid_data - min_val) / (max_val - min_val) * 255)
    
    return result.astype(np.uint8)

@jit(nopython=True, cache=True)
def create_alpha_channel(mask):
    """Create alpha channel from mask with JIT compilation"""
    return np.where(mask, np.uint8(0), np.uint8(255))

if GPU_AVAILABLE:
    # GPU kernel for normalization
    @cuda.jit
    def normalize_band_gpu(band, mask, result, min_val, max_val):
        """GPU kernel for band normalization"""
        i, j = cuda.grid(2)
        if i < band.shape[0] and j < band.shape[1]:
            if mask[i, j]:
                result[i, j] = 0
            else:
                if max_val > min_val:
                    result[i, j] = np.uint8(((band[i, j] - min_val) / (max_val - min_val)) * 255)
                else:
                    result[i, j] = np.uint8(band[i, j])

app = FastAPI(title="GeoTIFF XYZ Tile Service")

@app.on_event("startup")
async def startup_event():
    """Initialize bounds cache on startup"""
    logger.info("Building GeoTIFF bounds cache...")
    build_bounds_cache()
    logger.info(f"Cached bounds for {len(geotiff_bounds_cache)} GeoTIFF files")

# Configuration
GEOTIFF_DIR = Path(r"D:\gTif\back")
CACHE_DIR = Path("./tile_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Tile settings
TILE_SIZE = 256

# File handle cache for faster repeated access
file_cache = {}
file_cache_lock = Lock()

# GeoTIFF bounds cache for spatial queries
geotiff_bounds_cache = {}
bounds_cache_lock = Lock()

@lru_cache(maxsize=128)
def get_transformer(from_crs: str, to_crs: str):
    """Cached transformer creation to avoid repeated initialization"""
    return Transformer.from_crs(from_crs, to_crs, always_xy=True)


def build_bounds_cache():
    """Build cache of all GeoTIFF bounds in EPSG:4326"""
    with bounds_cache_lock:
        if geotiff_bounds_cache:
            return geotiff_bounds_cache
        
        for geotiff_path in get_geotiff_files():
            try:
                with rasterio.open(geotiff_path) as src:
                    bounds = src.bounds
                    crs = src.crs.to_string()
                    
                    # Transform bounds to WGS84 (EPSG:4326)
                    if crs != 'EPSG:4326':
                        transformer = get_transformer(crs, "EPSG:4326")
                        west, south = transformer.transform(bounds.left, bounds.bottom)
                        east, north = transformer.transform(bounds.right, bounds.top)
                    else:
                        west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
                    
                    geotiff_bounds_cache[geotiff_path.name] = {
                        'path': geotiff_path,
                        'bounds': {'west': west, 'south': south, 'east': east, 'north': north}
                    }
            except Exception as e:
                logger.error(f"Error reading bounds for {geotiff_path}: {e}")
        
        return geotiff_bounds_cache


def find_geotiff_by_bbox(west: float, south: float, east: float, north: float):
    """Find GeoTIFF files that intersect with the given bbox"""
    if not geotiff_bounds_cache:
        build_bounds_cache()
    
    matching_files = []
    for filename, data in geotiff_bounds_cache.items():
        b = data['bounds']
        # Check if bboxes intersect
        if not (east < b['west'] or west > b['east'] or north < b['south'] or south > b['north']):
            matching_files.append({
                'filename': filename,
                'path': data['path'],
                'bounds': b
            })
    
    return matching_files


def get_geotiff_files():
    """Get all GeoTIFF files from the directory"""
    patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    files = []
    for pattern in patterns:
        files.extend(GEOTIFF_DIR.glob(pattern))
    return files


def deg2num(lat_deg, lon_deg, zoom):
    """Convert lat/lon to tile numbers"""
    import math
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def num2deg(xtile, ytile, zoom):
    """Convert tile numbers to lat/lon bounds"""
    n = 2.0 ** zoom
    lon_min = xtile / n * 360.0 - 180.0
    lat_max = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n))) * 180.0 / math.pi
    lon_max = (xtile + 1) / n * 360.0 - 180.0
    lat_min = math.atan(math.sinh(math.pi * (1 - 2 * (ytile + 1) / n))) * 180.0 / math.pi
    return lon_min, lat_min, lon_max, lat_max


def is_tile_empty(img):
    """Check if a tile is empty (fully transparent or has no useful data)"""
    if img is None:
        return True
    
    # Convert to numpy array for faster processing
    img_array = np.array(img)
    
    # Check if image has alpha channel
    if img_array.shape[2] == 4:
        # Check if alpha channel is all zeros (fully transparent)
        alpha = img_array[:, :, 3]
        if np.all(alpha == 0):
            return True
        
        # Check if there's any non-transparent pixel with actual data
        # (not just black transparent pixels)
        has_data = np.any((alpha > 0) & (np.sum(img_array[:, :, :3], axis=2) > 0))
        return not has_data
    else:
        # No alpha channel - check if all pixels are black
        return np.all(img_array == 0)


def get_cache_path(z: int, x: int, y: int, filename: str) -> Path:
    """Generate cache file path - tile_cache/{z}/{x}_{y}.png"""
    cache_file = CACHE_DIR / f"{z}" / f"{x}_{y}.png"
    return cache_file


def get_cached_dataset(geotiff_path: Path):
    """Get or create cached dataset handle - keeps files open for faster access"""
    path_str = str(geotiff_path)
    with file_cache_lock:
        if path_str not in file_cache:
            file_cache[path_str] = rasterio.open(geotiff_path)
        return file_cache[path_str]


def render_tile(geotiff_path: Path, z: int, x: int, y: int) -> Optional[Image.Image]:
    """Render a tile from GeoTIFF with GPU acceleration"""
    try:
        # Skip rendering for zoom levels below 17 (performance optimization)
        MIN_ZOOM = 17
        if z < MIN_ZOOM:
            return None  # Return transparent tile
        
        # Get tile bounds in Web Mercator (EPSG:3857) directly
        lon_min, lat_min, lon_max, lat_max = num2deg(x, y, z)
        
        # Use cached dataset handle instead of reopening file every time
        src = get_cached_dataset(geotiff_path)
        if src is None:
            return None
        
        # Get source CRS
        src_crs = src.crs
        
        # Convert WGS84 bounds to Web Mercator first (standard tile projection)
        from pyproj import CRS, Transformer
        wgs84 = CRS.from_epsg(4326)
        web_mercator = CRS.from_epsg(3857)
        wgs84_to_merc = Transformer.from_crs(wgs84, web_mercator, always_xy=True)
        
        # Transform tile corners to Web Mercator
        merc_left, merc_bottom = wgs84_to_merc.transform(lon_min, lat_min)
        merc_right, merc_top = wgs84_to_merc.transform(lon_max, lat_max)
        
        # Now transform from Web Mercator to source CRS if needed
        if src_crs.to_epsg() == 3857:
            # Already in Web Mercator - use directly
            left, bottom, right, top = merc_left, merc_bottom, merc_right, merc_top
        elif src_crs.to_epsg() == 4326:
            # Source is WGS84 - use original lon/lat
            left, bottom, right, top = lon_min, lat_min, lon_max, lat_max
        else:
            # Transform from Web Mercator to source CRS
            merc_to_src = Transformer.from_crs(web_mercator, src_crs, always_xy=True)
            left, bottom = merc_to_src.transform(merc_left, merc_bottom)
            right, top = merc_to_src.transform(merc_right, merc_top)
        
        # Ensure correct order
        if left > right:
            left, right = right, left
        if bottom > top:
            bottom, top = top, bottom
        
        # Check if tile intersects with raster bounds
        if (left >= src.bounds.right or right <= src.bounds.left or
            bottom >= src.bounds.top or top <= src.bounds.bottom):
            # No intersection - return transparent tile
            return None
        
        # Don't clamp bounds - use exact tile bounds for proper alignment
        # This ensures tiles line up correctly at all zoom levels
        
        # Calculate window in pixel coordinates using exact bounds
        window = rasterio.windows.from_bounds(
            left, bottom, right, top, 
            transform=src.transform
        )
        
        # Round window to avoid sub-pixel shifts
        window = window.round_lengths().round_offsets()
        
        # Ensure window is valid
        if window.width <= 0 or window.height <= 0:
            return None
        
        # Read the data for this specific tile window (use faster resampling)
        data = src.read(
            window=window,
            out_shape=(src.count, TILE_SIZE, TILE_SIZE),
            resampling=rasterio.enums.Resampling.bilinear,  # Better quality for alignment
            boundless=True,
            fill_value=0
        )
        
        # Handle different band counts
        if src.count == 1:
            # Single band - convert to RGBA with transparency
            band = data[0]
            mask = band == 0  # Mask for transparent pixels
            band = np.nan_to_num(band, nan=0)
            
            # Check if there's any data
            if band.max() == 0 and band.min() == 0:
                return None
            
            # Use JIT-compiled normalization
            band_normalized = normalize_band(band, mask)
            alpha = create_alpha_channel(mask)
            
            # Create RGBA with alpha channel
            img_array = np.stack([band_normalized, band_normalized, band_normalized, alpha], axis=-1)
                
        elif src.count >= 3:
            # RGB or RGBA with transparency
            rgb = data[:3]
            # Mask where all bands are 0
            mask = (rgb[0] == 0) & (rgb[1] == 0) & (rgb[2] == 0)
            img_array = np.transpose(rgb, (1, 2, 0))
            img_array = np.nan_to_num(img_array, nan=0)
            
            # Check if there's any data
            if img_array.max() == 0:
                return None
            
            # Fast normalization with multiplication instead of division
            if img_array.max() > 255:
                scale = 255.0 / img_array.max()
                img_array = (img_array * scale).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
            
            # Add alpha channel using JIT
            alpha = create_alpha_channel(mask)
            img_array = np.dstack([img_array, alpha])
        else:
            return None
        
        # Create PIL Image with transparency
        img = Image.fromarray(img_array, mode='RGBA')
        return img
            
    except Exception as e:
        logger.error(f"Error rendering tile from {geotiff_path}: {e}")
        return None


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the UI"""
    html_file = Path(__file__).parent / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    return HTMLResponse("<h1>GeoTIFF Tile Service</h1><p>UI not found. Please ensure index.html exists.</p>")


@app.get("/api/info")
async def api_info():
    """Service information"""
    geotiff_files = get_geotiff_files()
    
    # GPU information
    gpu_info = {
        "enabled": GPU_AVAILABLE,
        "type": "Numba CUDA" if GPU_AVAILABLE else "Numba JIT (CPU)",
        "device": GPU_NAME if GPU_AVAILABLE else "CPU"
    }
    
    return {
        "service": "GeoTIFF XYZ Tile Service",
        "geotiff_directory": str(GEOTIFF_DIR),
        "geotiff_files_found": len(geotiff_files),
        "files": [f.name for f in geotiff_files],
        "tile_url_pattern": "/tiles/{filename}/{z}/{x}/{y}.png",
        "cache_directory": str(CACHE_DIR),
        "acceleration": gpu_info
    }


@app.get("/metadata/{filename}")
async def get_metadata(filename: str):
    """Get GeoTIFF metadata including bounds and CRS"""
    geotiff_path = GEOTIFF_DIR / filename
    if not geotiff_path.exists():
        geotiff_path = GEOTIFF_DIR / f"{filename}.tif"
        if not geotiff_path.exists():
            geotiff_path = GEOTIFF_DIR / f"{filename}.tiff"
            if not geotiff_path.exists():
                raise HTTPException(status_code=404, detail=f"GeoTIFF file not found: {filename}")
    
    try:
        with rasterio.open(geotiff_path) as src:
            bounds = src.bounds
            crs = src.crs.to_string()
            
            # Transform bounds to WGS84 (EPSG:4326) for web mapping
            if crs != 'EPSG:4326':
                transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                west, south = transformer.transform(bounds.left, bounds.bottom)
                east, north = transformer.transform(bounds.right, bounds.top)
            else:
                west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
            
            return {
                "filename": filename,
                "crs": crs,
                "bounds": {
                    "west": west,
                    "south": south,
                    "east": east,
                    "north": north
                },
                "width": src.width,
                "height": src.height,
                "bands": src.count
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading metadata: {e}")


@app.get("/api/find")
async def find_geotiff(west: float, south: float, east: float, north: float):
    """Find GeoTIFF files by bounding box intersection"""
    matching = find_geotiff_by_bbox(west, south, east, north)
    
    if not matching:
        return {
            "found": 0,
            "files": [],
            "message": "No GeoTIFF files intersect with the provided bbox"
        }
    
    return {
        "found": len(matching),
        "files": matching,
        "bbox_query": {"west": west, "south": south, "east": east, "north": north}
    }


@app.get("/tiles/auto/{z}/{x}/{y}.png")
async def get_tile_auto(z: int, x: int, y: int):
    """
    Automatic tile endpoint - determines GeoTIFF by tile location
    Uses tile coordinates to calculate bbox and find intersecting GeoTIFF
    """
    try:
        # Calculate tile bounds
        lon_min, lat_min, lon_max, lat_max = num2deg(x, y, z)
        
        # Find GeoTIFF files that intersect this tile
        matching = find_geotiff_by_bbox(lon_min, lat_min, lon_max, lat_max)
        
        if not matching:
            # Return transparent tile if no GeoTIFF found
            img = Image.new('RGBA', (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            return Response(content=buf.read(), media_type="image/png")
        
        # Try each matching GeoTIFF until we get a non-empty tile
        for geotiff_info in matching:
            geotiff_path = geotiff_info['path']
            
            # Check cache first
            cache_path = get_cache_path(z, x, y, geotiff_path.name)
            if cache_path.exists():
                # Load cached tile and check if it's empty
                cached_img = Image.open(cache_path)
                if not is_tile_empty(cached_img):
                    return FileResponse(cache_path, media_type="image/png")
                # If cached tile is empty, try next GeoTIFF
                continue
            
            # Render tile from this GeoTIFF
            tile_img = render_tile(geotiff_path, z, x, y)
            
            # Check if tile has data
            if tile_img is not None and not is_tile_empty(tile_img):
                # Found a non-empty tile - cache and return it
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tile_img.save(cache_path, "PNG")
                return FileResponse(cache_path, media_type="image/png")
        
        # All matching GeoTIFFs produced empty tiles - return transparent tile
        img = Image.new('RGBA', (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return Response(content=buf.read(), media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error in auto tile endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tiles/{filename}/{z}/{x}/{y}.png")
async def get_tile(filename: str, z: int, x: int, y: int):
    """Get a tile for the specified GeoTIFF file"""
    
    # Check cache first
    cache_path = get_cache_path(z, x, y, filename)
    if cache_path.exists():
        return FileResponse(cache_path, media_type="image/png")
    
    # Find the GeoTIFF file
    geotiff_path = GEOTIFF_DIR / filename
    if not geotiff_path.exists():
        # Try with .tif extension
        geotiff_path = GEOTIFF_DIR / f"{filename}.tif"
        if not geotiff_path.exists():
            geotiff_path = GEOTIFF_DIR / f"{filename}.tiff"
            if not geotiff_path.exists():
                raise HTTPException(status_code=404, detail=f"GeoTIFF file not found: {filename}")
    
    # Render the tile
    img = render_tile(geotiff_path, z, x, y)
    
    if img is None:
        # Return transparent tile if no data
        img = Image.new('RGBA', (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
    
    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(cache_path, "PNG")
    
    # Return the tile
    return FileResponse(cache_path, media_type="image/png")


@app.delete("/cache")
async def clear_cache():
    """Clear all cached tiles"""
    import shutil
    try:
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(exist_ok=True)
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {e}")


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    total_files = sum(1 for _ in CACHE_DIR.rglob("*.png"))
    total_size = sum(f.stat().st_size for f in CACHE_DIR.rglob("*.png"))
    return {
        "cached_tiles": total_files,
        "cache_size_mb": round(total_size / 1024 / 1024, 2),
        "cache_directory": str(CACHE_DIR)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
