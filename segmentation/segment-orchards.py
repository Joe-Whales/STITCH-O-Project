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
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from skimage import measure


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
    min_segment_pixels = int(total_area * config['segmentation']['min_segment_ratio'])
    max_segment_pixels = int(total_area * 0.9)  # Prevent removing very large segments
    min_circularity = config['segmentation'].get('min_circularity', 0.1)  # Adjust this threshold as needed

    # Create structuring element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Fill holes in the mask
    filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find connected components
    labels = measure.label(filled_mask, connectivity=2)
    props = measure.regionprops(labels)
    
    # Create a new mask for acceptable segments
    acceptable_segments_mask = np.zeros_like(mask, dtype=np.uint8)
    
    if not props:
        return acceptable_segments_mask
    
    # Find the largest segment
    largest_segment_area = max(prop.area for prop in props)
    
    # Count acceptable segments
    acceptable_segment_count = 0
    
    for prop in props:
        area = prop.area
        perimeter = prop.perimeter
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        if (area >= min_segment_pixels and area <= max_segment_pixels and circularity >= min_circularity) or area == largest_segment_area:
            acceptable_segments_mask[labels == prop.label] = 255
            acceptable_segment_count += 1
            
    return acceptable_segments_mask

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
    
    # Initial check for large, unfilled segments
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    
    # Calculate areas and thresholds
    areas = stats[1:, cv2.CC_STAT_AREA]  # Exclude background
    large_threshold = large_segment_threshold * mask.size
    very_large_threshold = 2 * large_threshold
    
    # Count large and very large segments
    large_segments = np.sum(areas > large_threshold * 1.5)
    very_large_segments = np.sum(areas > very_large_threshold)
    # If there are at least 2 large segments or 1 very large segment, return the original mask
    if large_segments > 2 or very_large_segments >= 1:
        print(f"Detected {large_segments} large segments and {very_large_segments} very large segments. Skipping adaptive closing.")
        return mask
    
    closed_mask = mask.copy()
    prev_num_labels = 0
    
    for size in range(4, max_kernel_size + 1, kernel_step):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        closed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_CLOSE, kernel)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask)
        
        large_segments = np.sum(stats[1:, cv2.CC_STAT_AREA] > large_segment_threshold * mask.size)

        if large_segments > 2 or (num_labels - prev_num_labels) < 100:
            print(f"Detected {large_segments} large segments or small increase in number of segments. Stopping at kernel size {size}.")
            print(large_segment_threshold * mask.size)
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

def add_border(mask, border_size=50):
    return ndimage.binary_dilation(mask, iterations=border_size)

def fill_small_holes(mask, config):
    small_hole_max_size = config['segmentation'].get('small_hole_max_size', 0.001)  # 0.1% of total area by default
    total_area = mask.shape[0] * mask.shape[1]
    max_hole_size = int(total_area * small_hole_max_size)
    
    # Invert the mask to find holes
    inverted_mask = np.logical_not(mask)
    
    # Label the holes
    labeled_holes, num_holes = ndimage.label(inverted_mask)
    
    # Get sizes of holes
    hole_sizes = np.bincount(labeled_holes.ravel())[1:]
    
    # Create a mask of holes to fill (those smaller than max_hole_size)
    holes_to_fill = np.logical_and(labeled_holes > 0, 
                                   np.take(hole_sizes, labeled_holes - 1) <= max_hole_size)
    
    # Fill the small holes
    filled_mask = np.logical_or(mask, holes_to_fill)
    
    print(f"Number of holes filled: {np.sum(holes_to_fill > 0)}")
    print(f"Largest hole filled: {np.max(hole_sizes[hole_sizes <= max_hole_size])} pixels")
    
    return filled_mask.astype(np.uint8) * 255

def process_single_segment(segment, config, depth=0, max_depth=5):
    max_connection_width = config['segmentation'].get('min_connection_width', 10)
    min_segment_size_percentage = config['segmentation'].get('min_segment_size_percentage', 0.2)

    original_segment = segment.copy()
    total_pixels = np.sum(segment)

    print(f"Processing segment at depth {depth} with {total_pixels} pixels.")

    if depth >= max_depth:
        return [segment]  # Return the segment as is if max depth is reached

    for connection_width in range(3, max_connection_width + 1, 2):
        kernel = np.ones((connection_width, connection_width), np.uint8)
        
        eroded = cv2.erode(segment, kernel, iterations=1)
        num_labels, labels = cv2.connectedComponents(eroded)
        
        # Calculate min_segment_size based on the current eroded segment size
        eroded_total_pixels = np.sum(eroded)
        min_segment_size = int(eroded_total_pixels * min_segment_size_percentage)
        
        large_segments = [i for i in range(1, num_labels) if np.sum(labels == i) >= min_segment_size]
        
        print(f"Depth: {depth}, Connection width: {connection_width}, Large segments: {len(large_segments)}")
        
        if len(large_segments) > 1:
            # If we found a separation, process each part recursively
            processed_segments = []
            for label in large_segments:
                part = (labels == label).astype(np.uint8)
                # Dilate the part to recover its original size, but don't let it grow beyond the original segment
                dilated_part = cv2.dilate(part, kernel, iterations=1)
                dilated_part = np.logical_and(dilated_part, original_segment).astype(np.uint8)
                
                # Recursively process this part
                processed_segments.extend(process_single_segment(dilated_part, config, depth=depth+1, max_depth=max_depth))
            
            return processed_segments

    # If no separation was found, return the original segment
    return [segment]

def separate_weakly_connected_segments(mask, config, debug=False, debug_dir=None, unique_id=None):
    print(f"Input mask shape: {mask.shape}")
    print(f"Number of non-zero pixels in input mask: {np.count_nonzero(mask)}")

    num_labels, labels = cv2.connectedComponents(mask)
    max_recursion_depth = config['segmentation'].get('max_recursion_depth', 5)

    separated_mask = np.zeros_like(mask)
    all_processed_segments = []

    for label in range(1, num_labels):
        segment = (labels == label).astype(np.uint8)
        segment_size = np.sum(segment)
        
        processed_segments = process_single_segment(segment, config, depth=0, max_depth=max_recursion_depth)
        all_processed_segments.extend(processed_segments)
        
        # Replace the original segment with its processed parts in the separated mask
        for processed_segment in processed_segments:
            separated_mask = np.maximum(separated_mask, processed_segment)

    if debug:
        debug_images = {
            'original_mask': mask,
            'separated_mask': separated_mask
        }
        
        for name, image in debug_images.items():
            cv2.imwrite(os.path.join(debug_dir, f"{name}_{unique_id}.png"), image)
        
        print(f"Number of segments before separation: {num_labels - 1}")
        print(f"Number of segments after separation: {len(all_processed_segments)}")
    
    return separated_mask * 255, all_processed_segments

def remove_nodata_segments(mask, image, config, debug=False, debug_dir=None, unique_id=None):
    nodata_value = tuple(config['segmentation'].get('nodata_value', (0, 0, 0, 0)))
    nodata_threshold = config['segmentation'].get('nodata_threshold', 100)
    max_nodata_percentage = config['segmentation'].get('max_nodata_percentage', 0.15)
    border_size = config['segmentation'].get('border_size', 50)

    # Ensure the image is in (height, width, channels) format
    if image.shape[0] == 4:
        image = image.transpose(1, 2, 0)

    # Create a binary mask of no-data values
    nodata_mask = np.all(image == nodata_value, axis=-1)

    # Separate weakly connected segments
    separated_mask, all_segments = separate_weakly_connected_segments(mask, config, debug, debug_dir, unique_id)

    # Create a new mask without segments containing too many no-data values
    new_mask = np.zeros_like(mask, dtype=bool)

    for i, segment in enumerate(all_segments):
        # Add border to the segment for checking
        bordered_segment = add_border(segment, border_size)
        filled_bordered_segment = ndimage.binary_fill_holes(bordered_segment)
        
        # Count no-data pixels in the filled bordered segment
        nodata_count = np.sum(filled_bordered_segment & nodata_mask)
        total_pixels = np.sum(filled_bordered_segment)
        nodata_ratio = nodata_count / total_pixels if total_pixels > 0 else 0
        
        if nodata_count <= nodata_threshold and nodata_ratio <= max_nodata_percentage:
            # Keep the original segment shape if it passes both checks
            new_mask |= segment.astype(bool)
            print(f"Keeping segment {i} with {nodata_count} no-data pixels ({nodata_ratio:.2%} of segment)")
        else:
            print(f"Removing segment {i} with {nodata_count} no-data pixels ({nodata_ratio:.2%} of segment)")

    print(f"Number of segments in output mask: {np.max(cv2.connectedComponents(new_mask.astype(np.uint8))[1]) - 1}")
    print(f"Number of non-zero pixels in output mask: {np.count_nonzero(new_mask)}")

    if debug:
        debug_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        debug_mask[new_mask] = [0, 255, 0]  # Green for kept segments
        debug_mask[mask & ~new_mask] = [0, 0, 255]  # Red for removed segments
        cv2.imwrite(os.path.join(debug_dir, f"nodata_removal_debug_{unique_id}.png"), debug_mask)

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

    filled_mask = fill_small_holes(initial_mask, config)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"2_filled_mask_{unique_id}.png"), filled_mask)

    # Apply adaptive closing
    closed_mask = adaptive_closing(filled_mask, config)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"3_closed_mask_{unique_id}.png"), closed_mask)

  # Remove small segments with dynamic threshold
    large_segments_mask = remove_small_segments(closed_mask, config)
    
    filled_mask = fill_small_holes(large_segments_mask, config)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"4_large_segments_{unique_id}.png"), filled_mask)

    nodata_checked_mask = remove_nodata_segments(filled_mask, downscaled_image, config)
    
    large_segments_mask_2 = remove_small_segments(nodata_checked_mask, config)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"5_nodata_checked_mask_{unique_id}.png"), large_segments_mask_2)


    # Add buffer
    buffer_size = config['segmentation'].get('buffer_size', 10)
    buffered_mask = add_buffer(large_segments_mask_2, buffer_size)

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