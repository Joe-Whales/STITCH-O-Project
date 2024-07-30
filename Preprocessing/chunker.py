import cv2
import numpy as np
import rasterio
import yaml
import argparse
import os
from tqdm import tqdm
import multiprocessing

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def segment_orchard(image, config):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_green = np.array(config['segmentation']['lower_green'])
    upper_green = np.array(config['segmentation']['upper_green'])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    kernel_size = config['segmentation']['morph_kernel_size']
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    mask_inverted = cv2.bitwise_not(mask)
    return mask_inverted

def remove_small_segments(mask, min_size_ratio=0.2):
    total_area = mask.shape[0] * mask.shape[1]
    min_size = int(total_area * min_size_ratio)
    
    # Identify connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Create a new mask with only large segments
    large_segments_mask = np.zeros_like(mask)
    for label in range(1, num_labels):  # Skip background (label 0)
        if stats[label, cv2.CC_STAT_AREA] >= min_size:
            large_segments_mask[labels == label] = 255
    
    return large_segments_mask

def fill_all_holes(mask):
    # Invert the mask
    inv_mask = cv2.bitwise_not(mask)
    
    # Flood fill from the corners
    h, w = mask.shape[:2]
    flood_mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(inv_mask, flood_mask, (0,0), 255)
    cv2.floodFill(inv_mask, flood_mask, (w-1,0), 255)
    cv2.floodFill(inv_mask, flood_mask, (0,h-1), 255)
    cv2.floodFill(inv_mask, flood_mask, (w-1,h-1), 255)
    
    # Invert back
    filled_mask = cv2.bitwise_not(inv_mask)
    return filled_mask

def process_chunk(args):
    chunk, config, chunk_id = args
    mask = segment_orchard(chunk, config)
    return (chunk_id, mask)

def process_tif(input_file, output_file, config):
    with rasterio.open(input_file) as src:
        image = src.read().transpose(1, 2, 0)
        profile = src.profile

    # Split the image into chunks for parallel processing
    chunks = np.array_split(image, multiprocessing.cpu_count())
    args_list = [(chunk, config, i) for i, chunk in enumerate(chunks)]

    # Process chunks in parallel
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(process_chunk, args_list), 
                            total=len(args_list), 
                            desc="Processing image chunks"))

    # Reassemble the mask from chunks
    masks = [mask for _, mask in sorted(results, key=lambda x: x[0])]
    full_mask = np.concatenate(masks, axis=0)

    # Post-processing steps
    min_segment_ratio = config['segmentation'].get('min_segment_ratio', 0.2)
    large_segments_mask = remove_small_segments(full_mask, min_size_ratio=min_segment_ratio)
    filled_mask = fill_all_holes(large_segments_mask)
    
    # Invert the mask and set non-segmented areas to no-data value
    final_mask = np.where(filled_mask == 0, 1, 0).astype(np.uint8)
    no_data_value = config['segmentation'].get('no_data_value', 255)
    final_mask = np.where(final_mask == 0, no_data_value, final_mask)

    # Save the mask
    profile.update(dtype=rasterio.uint8, count=1, nodata=no_data_value)
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(final_mask, 1)

    print(f"Segmentation mask saved to {output_file}")
    print(f"Mask unique values: {np.unique(final_mask)}")

def main(config_file):
    config = load_config(config_file)

    input_file = config['output_file']
    output_file = config['mask_output_file']
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    process_tif(input_file, output_file, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallelized Orchard Segmentation with Advanced Post-processing")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)