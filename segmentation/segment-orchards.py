import os
import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
import yaml
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def read_and_resample_block(input_file, window_data, scale_factor, overlap):
    with rasterio.open(input_file) as src:
        col_off, row_off, width, height = window_data
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
    
    resampled_overlap = int(overlap * scale_factor)
    start_row = resampled_overlap if row_off > 0 else 0
    start_col = resampled_overlap if col_off > 0 else 0
    end_row = resampled.shape[1] - resampled_overlap if row_off + height < src.height else resampled.shape[1]
    end_col = resampled.shape[2] - resampled_overlap if col_off + width < src.width else resampled.shape[2]
    
    clipped = resampled[:, start_row:end_row, start_col:end_col]
    
    return clipped, (col_off, row_off, width, height)

def downscale_tif(input_file, config):
    target_size = tuple(config['downscaling']['target_size'])
    chunk_size = config['downscaling']['chunk_size']
    overlap = config['downscaling'].get('overlap', 128)

    with rasterio.open(input_file) as src:
        scale_factor = min(target_size[0] / src.height, target_size[1] / src.width)
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
        
        windows = [
            (col, row, min(chunk_size, src.width - col), min(chunk_size, src.height - row))
            for row in range(0, src.height, chunk_size)
            for col in range(0, src.width, chunk_size)
        ]
        
        num_workers = config['downscaling']['num_workers']
        if num_workers <= 0:
            num_workers = multiprocessing.cpu_count()

    downscaled_image = np.zeros((output_profile['count'], output_height, output_width), dtype=np.uint8)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(read_and_resample_block, input_file, window, scale_factor, overlap) for window in windows]
        
        for future in futures:
            resampled, window_data = future.result()
            dst_window = Window(
                col_off=int(window_data[0] * scale_factor),
                row_off=int(window_data[1] * scale_factor),
                width=int(window_data[2] * scale_factor),
                height=int(window_data[3] * scale_factor)
            )
            
            # Ensure we don't write outside the bounds of the downscaled_image
            write_window = Window(
                col_off=dst_window.col_off,
                row_off=dst_window.row_off,
                width=min(dst_window.width, output_width - dst_window.col_off),
                height=min(dst_window.height, output_height - dst_window.row_off)
            )
            
            downscaled_image[:, write_window.row_off:write_window.row_off+write_window.height, 
                                write_window.col_off:write_window.col_off+write_window.width] = resampled[:, :write_window.height, :write_window.width]

    return downscaled_image, output_profile

def segment_orchard(image, config):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_green = np.array(config['segmentation']['lower_green'])
    upper_green = np.array(config['segmentation']['upper_green'])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    kernel_size = config['segmentation']['morph_kernel_size']
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

def remove_small_segments(mask, min_size_ratio=0.2):
    total_area = mask.shape[0] * mask.shape[1]
    min_size = int(total_area * min_size_ratio)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    large_segments_mask = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_size:
            large_segments_mask[labels == label] = 255
    
    return large_segments_mask

def fill_holes_in_segments(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filled_mask = np.zeros_like(mask)
    
    for label in range(1, num_labels):
        segment_mask = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(segment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(filled_mask, contours, -1, 255, -1)
    
    return filled_mask

def adaptive_closing(mask, max_kernel_size=50, step=2):
    closed_mask = mask.copy()
    kernel = np.ones((3, 3), np.uint8)
    
    for size in range(3, max_kernel_size + 1, step):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        closed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_CLOSE, kernel)
        
        # Check if the closing operation has connected all segments
        num_labels, _ = cv2.connectedComponents(closed_mask)
        if num_labels == 2:  # Background and one large segment
            break
    
    return closed_mask

def process_single_file(input_file, output_dir, config, debug=False):
    # Generate a unique identifier based on the full path of the input file
    rel_path = os.path.relpath(input_file, config['root_dir'])
    unique_id = rel_path.replace(os.path.sep, '_').replace('.', '_')
    
    downscaled_file = os.path.join(output_dir, "orthos", f"downscaled_{unique_id}.tif")
    mask_file = os.path.join(output_dir, "masks", f"seg_mask_{unique_id}.tif")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(downscaled_file), exist_ok=True)
    os.makedirs(os.path.dirname(mask_file), exist_ok=True)

    # Downscale
    downscaled_image, output_profile = downscale_tif(input_file, config)

    # Save downscaled image
    with rasterio.open(downscaled_file, 'w', **output_profile) as dst:
        dst.write(downscaled_image)
    print(f"Downscaled image saved to {downscaled_file}")

    # Segment
    image = downscaled_image.transpose(1, 2, 0)
    initial_mask = segment_orchard(image, config)

    if debug:
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"1_initial_mask_{unique_id}.png"), initial_mask)

    # Apply adaptive closing
    max_kernel_size = config['segmentation'].get('max_kernel_size', 50)
    step = config['segmentation'].get('kernel_step', 2)
    closed_mask = adaptive_closing(initial_mask, max_kernel_size, step)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"2_closed_mask_{unique_id}.png"), closed_mask)

    min_segment_ratio = config['segmentation'].get('min_segment_ratio', 0.2)
    large_segments_mask = remove_small_segments(closed_mask, min_size_ratio=min_segment_ratio)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"3_large_segments_{unique_id}.png"), large_segments_mask)

    filled_mask = fill_holes_in_segments(large_segments_mask)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"4_filled_mask_{unique_id}.png"), filled_mask)

    final_mask = cv2.bitwise_not(filled_mask)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"5_final_mask_{unique_id}.png"), final_mask)

    mask_profile = output_profile.copy()
    mask_profile.update(dtype=rasterio.uint8, count=1, nodata=0)
    with rasterio.open(mask_file, 'w', **mask_profile) as dst:
        dst.write(final_mask, 1)

    print(f"Segmentation mask saved to {mask_file}")

def main(config_file, root_dir, debug=False):
    config = load_config(config_file)
    config['root_dir'] = root_dir
    target_filename = config.get('target_filename', 'orthomosaic_visible.tif')

    matching_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if target_filename in filenames:
            matching_files.append(os.path.join(dirpath, target_filename))

    print(f"Found {len(matching_files)} matching files.")

    # Create output directories
    output_dir = os.path.join(os.path.dirname(root_dir), "segmentation")
    os.makedirs(os.path.join(output_dir, "orthos"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    for input_file in matching_files:
        print(f"Processing: {input_file}")
        try:
            process_single_file(input_file, output_dir, config, debug)
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchard Downscaling and Segmentation")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument("root_dir", type=str, help="Root directory to search for files")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    main(args.config, args.root_dir, args.debug)