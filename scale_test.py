import numpy as np
import rasterio
from rasterio.windows import Window

def scale_chunk(chunk, lower_percent=1, upper_percent=99, nodata_value=None):
    """Scale the chunk to its 1st and 99th percentile"""
    if nodata_value is not None:
        valid_data = chunk[chunk != nodata_value]
    else:
        valid_data = chunk
    
    if valid_data.size == 0:
        return np.zeros_like(chunk, dtype=np.uint8)
    
    # min_val = np.percentile(valid_data, lower_percent)
    # max_val = np.percentile(valid_data, upper_percent)
    
    min_val = np.min(valid_data)
    max_val = np.max(valid_data)
    
    scaled = np.clip((chunk - min_val) / (max_val - min_val), 0, 1)
    scaled = (scaled * 255).astype(np.uint8)
    
    if nodata_value is not None:
        scaled[chunk == nodata_value] = 0  # Set nodata areas to 0
    
    return scaled

def local_percentile_scaling(input_raster, output_raster, chunk_size=256):
    with rasterio.open(input_raster) as src:
        profile = src.profile.copy()
        nodata_value = src.nodata
        
        profile.update(dtype=rasterio.uint8, count=1, nodata=0)
        
        with rasterio.open(output_raster, 'w', **profile) as dst:
            for j in range(0, src.height, chunk_size):
                for i in range(0, src.width, chunk_size):
                    window = Window(i, j, min(chunk_size, src.width - i), min(chunk_size, src.height - j))
                    chunk = src.read(1, window=window)
                    
                    scaled_chunk = scale_chunk(chunk, nodata_value=nodata_value)
                    dst.write(scaled_chunk, 1, window=window)
    
    print(f"Scaled raster saved to: {output_raster}")
# Usage
#input_raster = 'Preprocessing\\data\\UOG_2057\\orthos\\data-analysis\\reg.tif'
input_raster = 'Blue_scaled.tif'
output_raster = 'scaled_output.tif'
local_percentile_scaling(input_raster, output_raster, chunk_size=256)