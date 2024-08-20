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
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
from scipy import ndimage
import time
import concurrent.futures
from functools import partial

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

def load_sam_model(config):
    """Load the SAM model."""
    sam_checkpoint = config['segmentation']['sam_checkpoint']
    model_type = config['segmentation']['model_type']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        pred_iou_thresh=config['segmentation']['pred_iou_thresh'],
        stability_score_thresh=config['segmentation']['stability_score_thresh'],
        box_nms_thresh=config['segmentation']['box_nms_thresh'],
        crop_nms_thresh=config['segmentation']['crop_nms_thresh']
    )
    return mask_generator

def create_threshold_mask(image, config):
    """Create a threshold mask based on color ranges."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_green = np.array(config['segmentation']['color_thresholds']['lower_green'])
    upper_green = np.array(config['segmentation']['color_thresholds']['upper_green'])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    kernel_size = config['segmentation']['morphology']['kernel_size']
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

def further_downscale_for_sam(image, target_size):
    """Further downscale the image for SAM processing."""
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    resized_3d = np.stack([resized, resized, resized], axis=-1)
    
    return resized_3d

def remove_small_segments(masks, min_size, image_shape):
    """Remove small segments from multiple masks and combine them."""    
    combined_mask = np.zeros(image_shape, dtype=bool)
    
    for mask in masks:
        segment = mask['segmentation']
        if np.sum(segment) >= min_size:
            combined_mask = np.logical_or(combined_mask, segment)
    
    return combined_mask

def process_segment(args):
    label, segment, nodata_mask, max_nodata_percentage, kernel = args
    dilated_segment = ndimage.binary_dilation(segment, structure=kernel)
    nodata_count = np.sum(dilated_segment & nodata_mask)
    segment_size = np.sum(dilated_segment)
    if (segment_size > 0 and (nodata_count / segment_size) > max_nodata_percentage) or segment_size < 10000:
        return label
    print(f"Segment {label}: {segment_size} pixels, {nodata_count} nodata pixels, {nodata_count*100 / segment_size:.2f}% nodata percentage")
    return None

def remove_nodata_segments(mask, rgb_image, nodata_value, max_nodata_percentage, border_size, num_threads=4):
    """Remove segments with high percentage of no-data pixels."""
    labeled, num_features = ndimage.label(mask)
    print(f"Total segments: {num_features}")
    
    if rgb_image.shape[0] == 4:
        rgb_image = rgb_image.transpose(1, 2, 0)
    
    # Create nodata mask
    nodata_mask = np.all(rgb_image == nodata_value, axis=-1)
    
    # Create morphological kernel
    kernel = np.ones((border_size, border_size), dtype=bool)
    
    # Prepare arguments for multiprocessing
    segments = [labeled == i for i in range(1, num_features + 1)]
    args = [(i+1, segment, nodata_mask, max_nodata_percentage, kernel) for i, segment in enumerate(segments)]
    
    # Process segments in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_segment, args))
    
    # Remove segments that exceed the nodata threshold
    segments_to_remove = [label for label in results if label is not None]
    mask[np.isin(labeled, segments_to_remove)] = 0
    
    return mask

def segment_image(image, mask_generator, config):
    """Apply the SAM model to the image and process the resulting masks."""    
    # Ensure image is in the correct format for SAM (3D, RGB)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    
    # Generate masks
    masks = mask_generator.generate(image)
    
    # Remove small segments and combine masks
    min_segment_size = config['segmentation']['min_segment_size']
    combined_mask = remove_small_segments(masks, min_segment_size, image.shape[:2])
    return combined_mask

def process_single_file(input_file, output_dir, config, mask_generator):
    """Process a single input file."""
    print(f"Processing file: {input_file}")
    rel_path = os.path.relpath(input_file, config['root_dir'])
    unique_id = rel_path.replace(os.path.sep, '_').replace('.', '_')
    
    downscaled_file = os.path.join(output_dir, "orthos", f"downscaled_{unique_id}.tif")
    mask_file = os.path.join(output_dir, "masks", f"seg_mask_{unique_id}.tif")

    os.makedirs(os.path.dirname(downscaled_file), exist_ok=True)
    os.makedirs(os.path.dirname(mask_file), exist_ok=True)

    # Downscaling
    if os.path.exists(downscaled_file):
        print("Loading existing downscaled image...")
        with rasterio.open(downscaled_file) as src:
            downscaled_image = src.read()
            output_profile = src.profile.copy()
    else:
        downscaled_image, output_profile = downscale_tif(input_file, config)
        print("Saving downscaled image...")
        with rasterio.open(downscaled_file, 'w', **output_profile) as dst:
            dst.write(downscaled_image)

    image = downscaled_image.transpose(1, 2, 0)

    # Create threshold mask
    threshold_mask = create_threshold_mask(image, config)

    # Further downscale for SAM
    sam_image = further_downscale_for_sam(threshold_mask, config['segmentation']['sam_target_size'])

    # Apply segmentation model
    segmentation_mask = segment_image(sam_image, mask_generator, config)

    # Upscale the mask to match the downscaled RGB image
    upscaled_mask = cv2.resize(segmentation_mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Remove segments with high no-data percentage
    nodata_value = np.array(config['segmentation']['nodata_value'])
    max_nodata_percentage = config['segmentation']['max_nodata_percentage']
    border_size = config['segmentation'].get('border_size', 3)
    final_mask = remove_nodata_segments(upscaled_mask, image, nodata_value, max_nodata_percentage, border_size)
    final_mask = np.logical_not(final_mask)
    
    temp = Image.fromarray(final_mask.astype(np.uint8) * 255)
    temp.save("mask.png")

    # Save final mask
    mask_profile = output_profile.copy()
    mask_profile.update(dtype=rasterio.uint8, count=1, nodata=0)
    with rasterio.open(mask_file, 'w', **mask_profile) as dst:
        dst.write(final_mask.astype(rasterio.uint8), 1)

    return True, downscaled_image, output_profile

def main(config_file, root_dir):
    """Main function to process all files."""
    print("Starting main process...")
    start_time = time.time()
    config = load_config(config_file)
    config['root_dir'] = root_dir
    target_filename = config.get('input', {}).get('target_filename', 'orthomosaic_visible.tif')

    # Load the SAM model
    mask_generator = load_sam_model(config)

    matching_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if target_filename in filenames:
            matching_files.append(os.path.join(dirpath, target_filename))

    output_dir = os.path.join(os.path.dirname(root_dir), "segmentation")
    os.makedirs(os.path.join(output_dir, "orthos"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    for input_file in matching_files:
        # try:
        success, _, _ = process_single_file(input_file, output_dir, config, mask_generator)
        if success:
            print(f"Successfully processed {input_file}")
        else:
            print(f"Failed to process {input_file}")
        # except Exception as e:
        #     print(f"Error processing {input_file}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchard Downscaling and Segmentation")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument("root_dir", type=str, help="Root directory to search for files")
    args = parser.parse_args()
    main(args.config, args.root_dir)