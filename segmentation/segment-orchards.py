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
from skimage import measure

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def read_and_resample_block(input_file, window_data, scale_factor, overlap):
    """Read and resample a block of the input image."""
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
    """Downscale the input TIF file."""
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
    """Perform initial segmentation of the orchard."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_green = np.array(config['segmentation']['color_thresholds']['lower_green'])
    upper_green = np.array(config['segmentation']['color_thresholds']['upper_green'])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    kernel_size = config['segmentation']['morphology']['kernel_size']
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

def remove_small_segments(mask, config):
    """Remove small segments from the mask."""
    total_area = mask.shape[0] * mask.shape[1]
    min_segment_pixels = int(total_area * config['segmentation']['size_thresholds']['min_segment_ratio'])
    max_segment_pixels = int(total_area * 0.9)
    min_circularity = config['segmentation']['shape_thresholds']['min_circularity']

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    labels = measure.label(filled_mask, connectivity=2)
    props = measure.regionprops(labels)
    
    acceptable_segments_mask = np.zeros_like(mask, dtype=np.uint8)
    
    if not props:
        return acceptable_segments_mask
    
    largest_segment_area = max(prop.area for prop in props)
    
    for prop in props:
        area = prop.area
        perimeter = prop.perimeter
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        if (area >= min_segment_pixels and area <= max_segment_pixels and circularity >= min_circularity) or area == largest_segment_area:
            acceptable_segments_mask[labels == prop.label] = 255
            
    return acceptable_segments_mask

def fill_holes_in_segments(mask):
    """Fill holes in segmented mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filled_mask = np.zeros_like(mask)
    
    for label in range(1, num_labels):
        segment_mask = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(segment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(filled_mask, contours, -1, 255, -1)
    
    return filled_mask

def adaptive_closing(mask, config):
    """Perform adaptive closing on the mask."""
    max_kernel_size = config['segmentation']['adaptive_closing']['max_kernel_size']
    kernel_step = config['segmentation']['adaptive_closing']['kernel_step']
    large_segment_threshold = config['segmentation']['adaptive_closing']['large_segment_threshold']
    min_large_segments = config['segmentation']['adaptive_closing']['min_large_segments']
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    
    areas = stats[1:, cv2.CC_STAT_AREA]
    large_threshold = large_segment_threshold * mask.size
    very_large_threshold = 2 * large_threshold
    
    large_segments = np.sum(areas > large_threshold * 1.5)
    very_large_segments = np.sum(areas > very_large_threshold)
    if large_segments >= min_large_segments or very_large_segments >= 1:
        return mask
    
    closed_mask = mask.copy()
    prev_num_labels = 0
    
    for size in range(3, max_kernel_size + 1, kernel_step):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        closed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_CLOSE, kernel)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask)
        large_segments = np.sum(stats[1:, cv2.CC_STAT_AREA] > large_segment_threshold * mask.size)
        
        if large_segments >= min_large_segments or abs(prev_num_labels - num_labels) < 100:
            break
        
        prev_num_labels = num_labels
    
    return closed_mask

def add_buffer(mask, buffer_size=10):
    """Add buffer around segmented regions."""
    kernel = np.ones((buffer_size * 2 + 1, buffer_size * 2 + 1), np.uint8)
    buffered_mask = cv2.dilate(mask, kernel, iterations=1)
    return buffered_mask

def add_border(mask, border_size=50):
    """Add border to the mask."""
    return ndimage.binary_dilation(mask, iterations=border_size)

def fill_small_holes(mask, config):
    """Fill small holes in the mask."""
    small_hole_max_size = config['segmentation']['hole_filling']['small_hole_max_size']
    total_area = mask.shape[0] * mask.shape[1]
    max_hole_size = int(total_area * small_hole_max_size)
    
    inverted_mask = np.logical_not(mask)
    labeled_holes, num_holes = ndimage.label(inverted_mask)
    hole_sizes = np.bincount(labeled_holes.ravel())[1:]
    holes_to_fill = np.logical_and(labeled_holes > 0, 
                                   np.take(hole_sizes, labeled_holes - 1) <= max_hole_size)
    filled_mask = np.logical_or(mask, holes_to_fill)
    
    return filled_mask.astype(np.uint8) * 255

def process_single_segment(segment, config, depth=0, max_depth=5):
    """Process a single segment, potentially splitting it."""
    max_connection_width = config['segmentation']['segment_separation']['max_connection_width']
    min_segment_size_percentage = config['segmentation']['segment_separation']['min_segment_size_percentage']

    original_segment = segment.copy()
    total_pixels = np.sum(segment)

    if depth >= max_depth:
        return [segment]

    for connection_width in range(3, max_connection_width + 1, 2):
        kernel = np.ones((connection_width, connection_width), np.uint8)
        
        eroded = cv2.erode(segment, kernel, iterations=1)
        num_labels, labels = cv2.connectedComponents(eroded)
        
        eroded_total_pixels = np.sum(eroded)
        min_segment_size = int(eroded_total_pixels * min_segment_size_percentage)
        
        large_segments = [i for i in range(1, num_labels) if np.sum(labels == i) >= min_segment_size]
        
        if len(large_segments) > 1:
            processed_segments = []
            for label in large_segments:
                part = (labels == label).astype(np.uint8)
                dilated_part = cv2.dilate(part, kernel, iterations=1)
                dilated_part = np.logical_and(dilated_part, original_segment).astype(np.uint8)
                
                processed_segments.extend(process_single_segment(dilated_part, config, depth=depth+1, max_depth=max_depth))
            
            return processed_segments

    return [segment]

def separate_weakly_connected_segments(mask, config):
    """Separate weakly connected segments in the mask."""
    num_labels, labels = cv2.connectedComponents(mask)
    max_recursion_depth = config['segmentation']['segment_separation']['max_recursion_depth']

    separated_mask = np.zeros_like(mask)
    all_processed_segments = []

    for label in range(1, num_labels):
        segment = (labels == label).astype(np.uint8)
        processed_segments = process_single_segment(segment, config, depth=0, max_depth=max_recursion_depth)
        all_processed_segments.extend(processed_segments)
        
        for processed_segment in processed_segments:
            separated_mask = np.maximum(separated_mask, processed_segment)

    return separated_mask * 255, all_processed_segments

def remove_nodata_segments(mask, image, config):
    """Remove segments with high percentage of no-data pixels."""
    nodata_value = tuple(config['segmentation']['nodata_removal']['nodata_value'])
    nodata_threshold = config['segmentation']['nodata_removal']['nodata_threshold']
    max_nodata_percentage = config['segmentation']['nodata_removal']['max_nodata_percentage']
    border_size = config['segmentation']['nodata_removal']['border_size']

    if image.shape[0] == 4:
        image = image.transpose(1, 2, 0)

    nodata_mask = np.all(image == nodata_value, axis=-1)
    separated_mask, all_segments = separate_weakly_connected_segments(mask, config)
    new_mask = np.zeros_like(mask, dtype=bool)

    for segment in all_segments:
        bordered_segment = add_border(segment, border_size)
        filled_bordered_segment = ndimage.binary_fill_holes(bordered_segment)
        
        nodata_count = np.sum(filled_bordered_segment & nodata_mask)
        total_pixels = np.sum(filled_bordered_segment)
        nodata_ratio = nodata_count / total_pixels if total_pixels > 0 else 0
        
        if nodata_count <= nodata_threshold and nodata_ratio <= max_nodata_percentage:
            new_mask |= segment.astype(bool)

    return new_mask.astype(np.uint8) * 255

def process_single_file(input_file, output_dir, config, debug=False):
    """Process a single input file."""
    rel_path = os.path.relpath(input_file, config['root_dir'])
    unique_id = rel_path.replace(os.path.sep, '_').replace('.', '_')
    
    downscaled_file = os.path.join(output_dir, "orthos", f"downscaled_{unique_id}.tif")
    mask_file = os.path.join(output_dir, "masks", f"seg_mask_{unique_id}.tif")

    os.makedirs(os.path.dirname(downscaled_file), exist_ok=True)
    os.makedirs(os.path.dirname(mask_file), exist_ok=True)

    # Downscaling
    if os.path.exists(downscaled_file):
        with rasterio.open(downscaled_file) as src:
            downscaled_image = src.read()
            output_profile = src.profile.copy()
    else:
        downscaled_image, output_profile = downscale_tif(input_file, config)
        with rasterio.open(downscaled_file, 'w', **output_profile) as dst:
            dst.write(downscaled_image)

    image = downscaled_image.transpose(1, 2, 0)

    # Segmentation process
    initial_mask = segment_orchard(image, config)
    filled_mask = fill_small_holes(initial_mask, config)
    closed_mask = adaptive_closing(filled_mask, config)
    filled_mask = fill_small_holes(closed_mask, config)
    large_segments_mask = remove_small_segments(filled_mask, config)
    nodata_checked_mask = remove_nodata_segments(large_segments_mask, downscaled_image, config)
    large_segments_mask_2 = remove_small_segments(nodata_checked_mask, config)
    buffer_size = config['segmentation']['buffer']['size']
    buffered_mask = add_buffer(large_segments_mask_2, buffer_size)
    filled_mask = fill_holes_in_segments(buffered_mask)

    # Change: Invert the mask and set non-segmented areas to nodata
    nodata_value = config['segmentation'].get('nodata_value', -999)
    final_mask = np.where(filled_mask == 255, nodata_value, 0)

    # Save final mask
    mask_profile = output_profile.copy()
    mask_profile.update(dtype=rasterio.int16, count=1, nodata=nodata_value)
    with rasterio.open(mask_file, 'w', **mask_profile) as dst:
        dst.write(final_mask.astype(rasterio.int16), 1)

    # Debug output
    if debug:
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        debug_images = {
            "1_initial_mask": initial_mask,
            "2_filled_mask": filled_mask,
            "3_closed_mask": closed_mask,
            "4_large_segments": large_segments_mask,
            "5_nodata_checked_mask": large_segments_mask_2,
            "6_buffered_mask": buffered_mask,
            "7_filled_mask": filled_mask,
            "8_final_mask": final_mask
        }
        for name, img in debug_images.items():
            cv2.imwrite(os.path.join(debug_dir, f"{name}_{unique_id}.png"), img)

    # Calculate segmented percentage
    total_pixels = filled_mask.size
    segmented_pixels = np.sum(filled_mask > 0)
    segmented_percentage = segmented_pixels / total_pixels
    
    return  segmented_percentage > 0.08, downscaled_image, output_profile

def main(config_file, root_dir, debug=False):
    """Main function to process all files."""
    config = load_config(config_file)
    config['root_dir'] = root_dir
    target_filename = config.get('input', {}).get('target_filename', 'orthomosaic_visible.tif')

    matching_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if target_filename in filenames:
            matching_files.append(os.path.join(dirpath, target_filename))

    output_dir = os.path.join(os.path.dirname(root_dir), "segmentation")
    os.makedirs(os.path.join(output_dir, "orthos"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    for input_file in matching_files:
        try:
            segments_found, downscaled_image, output_profile = process_single_file(input_file, output_dir, config, debug)
            if segments_found:
                print(f"Successfully segmented {input_file}")
            else:
                print(f"No large segments found in {input_file}. Increasing kernel size and rerunning segmentation.")
                # Create a new config with increased kernel size
                new_config = config.copy()
                new_config['segmentation']['morphology']['kernel_size'] += 6
                
                # Rerun segmentation with new config
                segments_found, _, _ = process_single_file(input_file, output_dir, new_config, debug)
                
                if segments_found:
                    print(f"Successfully segmented {input_file} with increased kernel size")
                else:
                    print(f"Warning: No large segments found in {input_file} even after increasing kernel size")
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchard Downscaling and Segmentation")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument("root_dir", type=str, help="Root directory to search for files")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    main(args.config, args.root_dir, args.debug)