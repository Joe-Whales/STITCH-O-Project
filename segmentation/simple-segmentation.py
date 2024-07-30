import cv2
import numpy as np
import rasterio
import yaml
import argparse

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

def process_tif(input_file, output_file, config):
    with rasterio.open(input_file) as src:
        image = src.read().transpose(1, 2, 0)
        profile = src.profile

    full_mask = segment_orchard(image, config)

    min_segment_ratio = config['segmentation'].get('min_segment_ratio', 0.2)
    large_segments_mask = remove_small_segments(full_mask, min_size_ratio=min_segment_ratio)

    filled_mask = fill_holes_in_segments(large_segments_mask)

    final_mask = cv2.bitwise_not(filled_mask)

    profile.update(dtype=rasterio.uint8, count=1, nodata=0)
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(final_mask, 1)

    print(f"Segmentation mask saved to {output_file}")

def main(config_file):
    config = load_config(config_file)

    input_file = config['output_file']
    output_file = config['mask_output_file']
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    process_tif(input_file, output_file, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchard Segmentation with Advanced Post-processing")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)