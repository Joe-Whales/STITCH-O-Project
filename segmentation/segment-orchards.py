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
from scipy import ndimage
import matplotlib.pyplot as plt

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

def remove_small_segments(mask, config):
    total_area = mask.shape[0] * mask.shape[1]
    min_size_ratio = config['segmentation']['min_segment_ratio']
    min_ratio_step = config['segmentation']['min_ratio_step']
    min_ratio_limit = config['segmentation']['min_ratio_limit']

    # Create structuring element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    while min_size_ratio >= min_ratio_limit:
        min_size = int(total_area * min_size_ratio)
        
        # Fill holes in the mask
        filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled_mask, connectivity=8)
        
        # Create a new mask for large segments
        large_segments_mask = np.zeros_like(mask, dtype=np.uint8)
        
        # Count segments larger than min_size
        large_segment_count = 0
        
        for label in range(1, num_labels):  # Start from 1 to skip background
            if stats[label, cv2.CC_STAT_AREA] >= min_size:
                large_segments_mask[labels == label] = 255
                large_segment_count += 1
        
        if large_segment_count >= 5 or min_size_ratio <= min_ratio_limit:
            break
        
        min_size_ratio -= min_ratio_step

    print(f"Final min_segment_ratio used: {min_size_ratio}")
    return large_segments_mask, min_size_ratio

def fill_holes_in_segments(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filled_mask = np.zeros_like(mask)
    
    for label in range(1, num_labels):
        segment_mask = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(segment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(filled_mask, contours, -1, 255, -1)
    
    return filled_mask

def adaptive_closing(mask, config):
    max_kernel_size = config['segmentation']['max_kernel_size']
    kernel_step = config['segmentation']['kernel_step']
    large_segment_threshold = config['segmentation']['large_segment_threshold']
    
    closed_mask = mask.copy()
    prev_num_labels = 0
    
    for size in range(3, max_kernel_size + 1, kernel_step):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        closed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_CLOSE, kernel)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask)
        
        large_segments = np.sum(stats[1:, cv2.CC_STAT_AREA] > large_segment_threshold * mask.size)
        if large_segments > 2 or num_labels == prev_num_labels:
            break
        
        prev_num_labels = num_labels
    
    return closed_mask

def add_buffer(mask, buffer_size=10):
    kernel = np.ones((buffer_size * 2 + 1, buffer_size * 2 + 1), np.uint8)
    buffered_mask = cv2.dilate(mask, kernel, iterations=1)
    return buffered_mask

def has_large_segments(mask, min_size_ratio=0.1):
    total_area = mask.shape[0] * mask.shape[1]
    min_size = int(total_area * min_size_ratio)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    for label in range(1, num_labels):  # Start from 1 to skip background
        if stats[label, cv2.CC_STAT_AREA] >= min_size:
            return True
    
    return False

def trim_edges(mask):
    # Find the bounding box of all non-zero pixels
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop the mask to the bounding box
    return mask[rmin:rmax+1, cmin:cmax+1], (rmin, rmax, cmin, cmax)

def remove_border_segments(mask, border_width=10):
    # Trim the edges
    trimmed_mask, (rmin, rmax, cmin, cmax) = trim_edges(mask)
    
    height, width = trimmed_mask.shape

    # Create a border mask
    border_mask = np.zeros_like(trimmed_mask)
    border_mask[:border_width, :] = 1
    border_mask[-border_width:, :] = 1
    border_mask[:, :border_width] = 1
    border_mask[:, -border_width:] = 1

    # Label the segments
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(trimmed_mask, connectivity=8)

    # Create a new mask without border-touching segments
    new_mask = np.zeros_like(trimmed_mask)
    for label in range(1, num_labels):  # Start from 1 to skip background
        segment = (labels == label)
        if not np.any(segment & border_mask):
            new_mask |= segment

    # Create a full-sized mask with the same dimensions as the input
    full_mask = np.zeros_like(mask)
    full_mask[rmin:rmax+1, cmin:cmax+1] = new_mask

    return full_mask

def add_border(mask, border_size=50):
    return ndimage.binary_dilation(mask, iterations=border_size)

def remove_nodata_segments(mask, image, nodata_value=(0, 0, 0, 0), nodata_threshold=100, border_size=50, debug=False, debug_dir=None, unique_id=None):
    # Ensure the image is in (height, width, channels) format
    if image.shape[0] == 4:
        image = image.transpose(1, 2, 0)

    # Create a binary mask of no-data values
    nodata_mask = np.all(image == nodata_value, axis=-1)

    print(f"Shape of input mask: {mask.shape}")
    print(f"Shape of image: {image.shape}")
    print(f"Shape of nodata_mask: {nodata_mask.shape}")
    print(f"Number of no-data pixels: {np.sum(nodata_mask)}")
    print(f"Percentage of no-data pixels: {np.sum(nodata_mask) / nodata_mask.size * 100:.2f}%")


    # Label the segments
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    print(f"Number of segments in input mask: {num_labels - 1}")  # Subtract 1 to exclude background

    # Create a new mask without segments containing too many no-data values
    new_mask = np.zeros_like(mask, dtype=bool)

    for label in range(1, num_labels):
        segment = labels == label
        # Add border to the segment for checking
        bordered_segment = add_border(segment, border_size)
        filled_bordered_segment = ndimage.binary_fill_holes(bordered_segment)
        
        # Count no-data pixels in the filled bordered segment
        nodata_count = np.sum(filled_bordered_segment & nodata_mask)
        
        if nodata_count <= nodata_threshold:
            # Keep the original segment shape if it passes the check
            new_mask |= segment
            print(f"Keeping segment {label} with {nodata_count} no-data pixels")
        else:
            print(f"Removing segment {label} with {nodata_count} no-data pixels")

    print(f"Number of segments in output mask: {np.max(cv2.connectedComponents(new_mask.astype(np.uint8))[1]) - 1}")


    return new_mask.astype(np.uint8) * 255

def process_single_file(input_file, output_dir, config, debug=False):
    # Generate a unique identifier based on the full path of the input file
    rel_path = os.path.relpath(input_file, config['root_dir'])
    unique_id = rel_path.replace(os.path.sep, '_').replace('.', '_')
    
    downscaled_file = os.path.join(output_dir, "orthos", f"downscaled_{unique_id}.tif")
    mask_file = os.path.join(output_dir, "masks", f"seg_mask_{unique_id}.tif")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(downscaled_file), exist_ok=True)
    os.makedirs(os.path.dirname(mask_file), exist_ok=True)

    # Check if downscaled file already exists
    if os.path.exists(downscaled_file):
        print(f"Downscaled image already exists at {downscaled_file}. Reading it...")
        with rasterio.open(downscaled_file) as src:
            downscaled_image = src.read()
            output_profile = src.profile.copy()
    else:
        # Downscale
        downscaled_image, output_profile = downscale_tif(input_file, config)

        # Save downscaled image
        with rasterio.open(downscaled_file, 'w', **output_profile) as dst:
            dst.write(downscaled_image)
        print(f"Downscaled image saved to {downscaled_file}")

    # Rest of the function remains the same
    image = downscaled_image.transpose(1, 2, 0)
    initial_mask = segment_orchard(image, config)

    if debug:
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"1_initial_mask_{unique_id}.png"), initial_mask)

    # Apply adaptive closing
    closed_mask = adaptive_closing(initial_mask, config)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"2_closed_mask_{unique_id}.png"), closed_mask)

    # Trim edges and remove border segments
    border_width = config['segmentation'].get('border_width', 10)
    border_checked_mask = remove_border_segments(closed_mask, border_width)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"3_border_checked_mask_{unique_id}.png"), border_checked_mask)
    
  # Remove small segments with dynamic threshold
    large_segments_mask, final_ratio = remove_small_segments(border_checked_mask, config)

    print(f"Final min_segment_ratio used: {final_ratio}")

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"4_large_segments_{unique_id}.png"), large_segments_mask)


    nodata_threshold = config['segmentation'].get('nodata_threshold', 100)
    border_size = config['segmentation'].get('border_size', 50)
    nodata_checked_mask = remove_nodata_segments(large_segments_mask, downscaled_image, nodata_value=(0, 0, 0, 0), 
                                                 nodata_threshold=nodata_threshold,
                                                 border_size=border_size,
                                                 debug=debug, debug_dir=debug_dir, unique_id=unique_id)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"5_nodata_checked_mask_{unique_id}.png"), nodata_checked_mask)


    # Add buffer
    buffer_size = config['segmentation'].get('buffer_size', 10)
    buffered_mask = add_buffer(nodata_checked_mask, buffer_size)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"6_buffered_mask_{unique_id}.png"), buffered_mask)

    filled_mask = fill_holes_in_segments(buffered_mask)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"7_filled_mask_{unique_id}.png"), filled_mask)

    # Invert the mask
    final_mask = cv2.bitwise_not(filled_mask)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"8_final_mask_{unique_id}.png"), final_mask)

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