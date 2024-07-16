import os
import numpy as np
import concurrent.futures
import multiprocessing
import random
import shutil
import argparse
from typing import List, Tuple

def is_orchard_folder(folder_path: str) -> bool:
    """Check if the folder is an orchard folder containing required subfolders."""
    required_folders = ['normal', 'case_1', 'case_2', 'case_3']
    return all(os.path.isdir(os.path.join(folder_path, subfolder)) for subfolder in required_folders)

def train_test_split(folder_path: str):
    """Perform train/test split on the data."""
    print(f"Performing train/test split for: {folder_path}")
    
    for dir_name in ['train', 'test']:
        if os.path.exists(os.path.join(folder_path, dir_name)):
            shutil.rmtree(os.path.join(folder_path, dir_name))
    
    os.makedirs(os.path.join(folder_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'test'), exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'test', 'normal'), exist_ok=True)
    
    if os.path.exists(os.path.join(folder_path, 'normal')):
        shutil.move(os.path.join(folder_path, 'normal'), os.path.join(folder_path, 'train', 'normal'))
    
    case_counts = []
    for case in ['case_1', 'case_2', 'case_3']:
        case_path = os.path.join(folder_path, case)
        if os.path.exists(case_path):
            if not os.listdir(case_path):
                print(f"{case} is empty. Removing it.")
                shutil.rmtree(case_path)
            else:
                shutil.move(case_path, os.path.join(folder_path, 'test', case))
                npy_count = len([f for f in os.listdir(os.path.join(folder_path, 'test', case)) if f.endswith('.npy')])
                case_counts.append(npy_count)
                print(f"{case} contains {npy_count} .npy files")
    
    if case_counts:
        avg_count = sum(case_counts) // len(case_counts)
        print(f"Average .npy count: {avg_count}")
        normal_files = [f for f in os.listdir(os.path.join(folder_path, 'train', 'normal')) if f.endswith('.npy')]
        files_to_move = random.sample(normal_files, min(avg_count, len(normal_files)//4))
        
        for file in files_to_move:
            shutil.move(os.path.join(folder_path, 'train', 'normal', file), 
                        os.path.join(folder_path, 'test', 'normal', file))
        
        print(f"Moved {len(files_to_move)} .npy files from train/normal to test/normal")
    else:
        print("No non-empty case folders found. No files moved to test/normal.")

def calculate_percentiles(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate min and max values (1st and 99th percentiles) for a single file."""
    data = np.load(file_path)
    min_vals = np.percentile(data, 1, axis=(0, 1))
    max_vals = np.percentile(data, 99, axis=(0, 1))
    return min_vals, max_vals

def scale_file(args: Tuple[str, np.ndarray, np.ndarray, str]):
    """Scale a single file based on its min and max values."""
    input_file, min_vals, max_vals, output_file = args
    data = np.load(input_file)
    scaled_data = (data - min_vals) / (max_vals - min_vals)
    scaled_data = np.clip(scaled_data, 0, 1)
    np.save(output_file, scaled_data)

def process_orchard(orchard_path: str, output_path: str):
    """Process a single orchard: calculate percentiles and scale data."""
    print(f"Processing orchard: {orchard_path}")
    
    # Calculate percentiles
    file_list = [os.path.join(root, f) for root, _, files in os.walk(orchard_path) 
                 for f in files if f.endswith('.npy')]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(calculate_percentiles, file_list))
    
    min_vals = np.min([r[0] for r in results], axis=0)
    max_vals = np.max([r[1] for r in results], axis=0)
    
    # Scale data
    os.makedirs(output_path, exist_ok=True)
    output_file_list = [os.path.join(output_path, os.path.relpath(f, orchard_path)) for f in file_list]
    
    for output_file in output_file_list:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(scale_file, zip(file_list, [min_vals]*len(file_list), 
                                     [max_vals]*len(file_list), output_file_list))

def calculate_mean_std(orchard_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate mean and standard deviation for an orchard."""
    file_list = [os.path.join(root, f) for root, _, files in os.walk(orchard_path) 
                 for f in files if f.endswith('.npy')]
    
    data_sum = np.zeros(3)
    data_sq_sum = np.zeros(3)
    total_pixels = 0
    
    for file in file_list:
        data = np.load(file)
        data_sum += np.sum(data, axis=(0, 1))
        data_sq_sum += np.sum(np.square(data), axis=(0, 1))
        total_pixels += data.shape[0] * data.shape[1]
    
    mean = data_sum / total_pixels
    std = np.sqrt(data_sq_sum / total_pixels - np.square(mean))
    
    return mean, std

def main(root_dir: str, output_dir: str):
    orchards = [orchard for orchard in os.listdir(root_dir) 
                if os.path.isdir(os.path.join(root_dir, orchard)) and 
                is_orchard_folder(os.path.join(root_dir, orchard))]
    
    for orchard in orchards:
        orchard_path = os.path.join(root_dir, orchard)
        output_path = os.path.join(output_dir, orchard)
        
        # Process orchard
        process_orchard(orchard_path, output_path)
        
        # Perform train/test split
        train_test_split(output_path)
        
        # Calculate mean and std
        mean, std = calculate_mean_std(output_path)
        
        # Save statistics
        stats_dir = os.path.join(output_path, 'data_stats')
        os.makedirs(stats_dir, exist_ok=True)
        np.save(os.path.join(stats_dir, 'train_means.npy'), mean)
        np.save(os.path.join(stats_dir, 'train_sdv.npy'), std)
        
        print(f"Orchard {orchard} processed. Mean: {mean}, Std: {std}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and scale image data for orchards.")
    parser.add_argument("root_dir", type=str, help="Path to the root directory containing the orchard folders")
    parser.add_argument("output_dir", type=str, help="Path to save the processed data")
    args = parser.parse_args()
    
    main(args.root_dir, args.output_dir)