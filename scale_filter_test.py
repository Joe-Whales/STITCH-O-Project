import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage

def enhance_contrast_and_smooth(chunk, base_center_value=127, alpha=0.5, contrast_factor=5, sigma=2, final_stretch_alpha=0.5, nodata_value=None):
    """Enhance contrast of the chunk, smooth it, and apply final stretch with wrapping"""
    if nodata_value is not None:
        valid_mask = chunk != nodata_value
    else:
        valid_mask = np.ones_like(chunk, dtype=bool)
    
    if np.sum(valid_mask) == 0:
        return np.zeros_like(chunk, dtype=np.uint8)
    
    # Convert to float for processing
    chunk_float = chunk.astype(float)
    
    # Calculate the mean of the chunk
    chunk_mean = np.mean(chunk_float[valid_mask])
    
    # Calculate the adaptive center value
    center_value = alpha * base_center_value + (1 - alpha) * chunk_mean
    
    # Enhance contrast around the center value
    diff = chunk_float - center_value
    contrasted = np.where(diff > 0, 
                          center_value + diff * contrast_factor,
                          center_value + diff / contrast_factor)
    
    # Apply Gaussian smoothing
    smoothed = ndimage.gaussian_filter(contrasted, sigma=sigma)
    
    # Apply final stretch
    min_val = np.min(smoothed[valid_mask])
    max_val = np.max(smoothed[valid_mask])
    if max_val > min_val:
        stretched = (smoothed - min_val) / (max_val - min_val) * 255
    else:
        stretched = smoothed
    
    # Blend the smoothed and stretched results
    result = final_stretch_alpha * stretched + (1 - final_stretch_alpha) * smoothed
    
    # Wrap very dark values to white
    result = np.where(result < 1, 1, result)
    
    # Final clipping to ensure 0-255 range
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # if nodata_value is not None:
    #     result[~valid_mask] = 0  # Set original nodata areas to 0
    
    return result

def process_raster(input_raster, output_raster, chunk_size=1024, base_center_value=127, alpha=0.5, contrast_factor=5, sigma=2, final_stretch_alpha=0.5):
    with rasterio.open(input_raster) as src:
        profile = src.profile.copy()
        nodata_value = src.nodata
        
        profile.update(dtype=rasterio.uint8, count=1, nodata=0)
        
        with rasterio.open(output_raster, 'w', **profile) as dst:
            for j in range(0, src.height, chunk_size):
                for i in range(0, src.width, chunk_size):
                    window = Window(i, j, min(chunk_size, src.width - i), min(chunk_size, src.height - j))
                    chunk = src.read(1, window=window)
                    
                    processed_chunk = enhance_contrast_and_smooth(chunk, base_center_value, alpha, contrast_factor, sigma, final_stretch_alpha, nodata_value)
                    dst.write(processed_chunk, 1, window=window)
    
    print(f"Processed raster saved to: {output_raster}")

# Usage
input_raster = 'Blue_scaled.tif'
output_raster = 'enhanced_smoothed_output.tif'
process_raster(input_raster, output_raster, 
               chunk_size=128, 
               base_center_value=180, 
               alpha=0.9, 
               contrast_factor=200, 
               sigma=1, 
               final_stretch_alpha=0.3)