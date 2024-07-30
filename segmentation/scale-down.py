import os
import rasterio
from rasterio.enums import Resampling
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import yaml
from rasterio.windows import Window

def read_and_resample_block(input_file, window_data, scale_factor, overlap):
    with rasterio.open(input_file) as src:
        col_off, row_off, width, height = window_data
        # Expand the window by the overlap amount
        expanded_window = Window(
            max(0, col_off - overlap),
            max(0, row_off - overlap),
            min(src.width - col_off + overlap, width + 2*overlap),
            min(src.height - row_off + overlap, height + 2*overlap)
        )
        
        data = src.read(window=expanded_window)
        resampled = np.zeros(
            (data.shape[0],
             int(expanded_window.height * scale_factor),
             int(expanded_window.width * scale_factor)),
            dtype=np.uint8)
        
        for i in range(data.shape[0]):
            resampled[i] = src.read(
                i + 1,
                out_shape=(
                    int(expanded_window.height * scale_factor),
                    int(expanded_window.width * scale_factor)
                ),
                window=expanded_window,
                resampling=Resampling.lanczos
            )
    
    # Calculate the overlap in the resampled space
    resampled_overlap = int(overlap * scale_factor)
    
    # Clip the resampled data to remove the overlap
    start_row = resampled_overlap if row_off > 0 else 0
    start_col = resampled_overlap if col_off > 0 else 0
    end_row = resampled.shape[1] - resampled_overlap if row_off + height < src.height else resampled.shape[1]
    end_col = resampled.shape[2] - resampled_overlap if col_off + width < src.width else resampled.shape[2]
    
    clipped = resampled[:, start_row:end_row, start_col:end_col]
    
    return clipped, (col_off, row_off, width, height)

def downscale_tif(input_file, output_file, config):
    target_size = tuple(config['downscaling']['target_size'])
    chunk_size = config['downscaling']['chunk_size']
    overlap = config['downscaling'].get('overlap', 128)  # Default overlap of 128 pixels

    with rasterio.open(input_file) as src:
        print(f"Input image size: {src.width}x{src.height}")
        scale_factor = min(target_size[0] / src.height, target_size[1] / src.width)
        print(f"Calculated scale factor: {scale_factor}")
        
        output_height = int(src.height * scale_factor)
        output_width = int(src.width * scale_factor)
        output_profile = src.profile.copy()
        output_profile.update({
            'height': output_height,
            'width': output_width,
            'transform': src.transform * src.transform.scale(
                (src.width / output_width),
                (src.height / output_height)
            )
        })
        print(f"Output image size: {output_width}x{output_height}")
        
        windows = []
        for row in range(0, src.height, chunk_size):
            for col in range(0, src.width, chunk_size):
                width = min(chunk_size, src.width - col)
                height = min(chunk_size, src.height - row)
                windows.append((col, row, width, height))
        
        print(f"Number of windows: {len(windows)}")
        
        num_workers = config['downscaling']['num_workers']
        if num_workers <= 0:
            num_workers = multiprocessing.cpu_count()
        print(f"Number of workers: {num_workers}")

    with rasterio.open(output_file, 'w', **output_profile) as dst:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(read_and_resample_block, input_file, window, scale_factor, overlap) for window in windows]
            
            for future in futures:
                resampled, window_data = future.result()
                dst_window = Window(
                    col_off=int(window_data[0] * scale_factor),
                    row_off=int(window_data[1] * scale_factor),
                    width=resampled.shape[2],
                    height=resampled.shape[1]
                )
                
                dst.write(resampled, window=dst_window)

    print(f"Downscaled image saved to {output_file}")

def main(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    input_file = config['input_file']
    output_file = config['output_file']
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    downscale_tif(input_file, output_file, config)

if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_file)