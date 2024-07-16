import os
import sys
import numpy as np
from typing import List, Optional, Tuple
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_file(file_path: str, num_channels: int):
    data = np.load(file_path)
    channel_sums = np.sum(data, axis=(0, 1))
    channel_sq_sums = np.sum(np.square(data), axis=(0, 1))
    pixel_count = data.shape[0] * data.shape[1]
    
    # Collect a sample of values for percentile estimation
    sample_size = min(1000, pixel_count)  # Adjust sample size as needed
    sample_indices = np.random.choice(pixel_count, sample_size, replace=False)
    sample_data = data.reshape(-1, num_channels)[sample_indices]
    
    return channel_sums, channel_sq_sums, pixel_count, sample_data

def calc_statistics_multi(chunk_path: str, num_channels: int, percentiles: List[int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    print("Calculating statistics (mean, standard deviation, and percentiles)")

    all_files = []
    for root, dirs, files in os.walk(chunk_path):
        if os.path.basename(root) in ["normal", "case_1", "case_2", "case_3"]:
            all_files.extend([os.path.join(root, f) for f in files if f.endswith('.npy')])

    if not all_files:
        return None, None, None

    # Parallel processing
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(process_file, [(f, num_channels) for f in all_files])

    # Aggregate results
    channel_sums = np.sum([r[0] for r in results], axis=0)
    channel_sq_sums = np.sum([r[1] for r in results], axis=0)
    pixel_count = np.sum([r[2] for r in results])
    all_samples = np.concatenate([r[3] for r in results], axis=0)

    # Calculate statistics
    means = channel_sums / pixel_count
    variances = (channel_sq_sums / pixel_count) - np.square(means)
    stddevs = np.sqrt(variances)

    # Calculate percentiles
    percentile_values = np.percentile(all_samples, percentiles, axis=0).T

    return means, stddevs, percentile_values

def calc_mean(chunk_path: str, num_channels: int) -> Optional[np.ndarray]:
    """
    Calculate the mean of each channel across all .npy files in the given path.

    Args:
        chunk_path (str): Path to the directory containing the data.
        num_channels (int): Number of channels in the data.

    Returns:
        Optional[np.ndarray]: Array of mean values for each channel, or None if no data is found.
    """
    print("Calculating mean")
    pixel_count = 0
    channel_sums = np.zeros(shape=(num_channels,), dtype=np.float64)

    for root, dirs, files in os.walk(chunk_path):
        if os.path.basename(root) in ["normal", "case_1", "case_2", "case_3"]:
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    file_data = np.load(file_path)
                    channel_sums += np.sum(file_data, axis=(0, 1))
                    pixel_count += file_data.shape[0] * file_data.shape[1]

    return channel_sums / pixel_count if pixel_count > 0 else None

def calc_sdv(means: np.ndarray, chunk_path: str, num_channels: int) -> Optional[np.ndarray]:
    """
    Calculate the standard deviation of each channel across all .npy files in the given path.

    Args:
        means (np.ndarray): Array of mean values for each channel.
        chunk_path (str): Path to the directory containing the data.
        num_channels (int): Number of channels in the data.

    Returns:
        Optional[np.ndarray]: Array of standard deviation values for each channel, or None if no data is found.
    """
    print("Calculating standard deviation")
    pixel_count = 0
    channel_sums = np.zeros(shape=(num_channels,), dtype=np.float64)

    for root, dirs, files in os.walk(chunk_path):
        if os.path.basename(root) in ["normal", "case_1", "case_2", "case_3"]:
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    file_data = np.load(file_path)
                    
                    for channel in range(num_channels):
                        squared_diff = (file_data[:, :, channel] - means[channel]) ** 2
                        channel_sums[channel] += np.sum(squared_diff)

                    pixel_count += file_data.shape[0] * file_data.shape[1]

    return np.sqrt(channel_sums / pixel_count) if pixel_count > 0 else None

def calc_percentiles(chunk_path: str, num_channels: int, percentiles: List[int]) -> Optional[np.ndarray]:
    """
    Calculate specified percentiles of each channel across all .npy files in the given path.

    Args:
        chunk_path (str): Path to the directory containing the data.
        num_channels (int): Number of channels in the data.
        percentiles (List[int]): List of percentiles to calculate.

    Returns:
    Optional[np.ndarray]: Array of percentile values for each channel, or None if no data is found.
        """
    print("Calculating percentiles")

    result = np.zeros((num_channels, len(percentiles)), dtype=np.float64)
    for channel in range(num_channels):
        channel_data = []
        for root, dirs, files in os.walk(chunk_path):
            if os.path.basename(root) in ["normal", "case_1", "case_2", "case_3"]:
                for file in files:
                    if file.endswith('.npy'):
                        file_path = os.path.join(root, file)
                        file_data = np.load(file_path)
                        channel_data.extend(file_data[:, :, channel].flatten())
        
        if channel_data:
            result[channel, :] = np.percentile(channel_data, percentiles)
    
    return result if len(channel_data) > 0 else None

def process_folder(folder_path: str, num_channels: int, stats_folder: str) -> None:
    """
    Process a folder to calculate and save statistics (mean, standard deviation, percentiles).

    Args:
        folder_path (str): Path to the folder to process.
        num_channels (int): Number of channels in the data.
        stats_folder (str): Path to save the statistics.
    """
    print(f"Processing folder: {folder_path}")
    
    # train_means = calc_mean(folder_path, num_channels)
    # if train_means is None:
    #     print(f"No valid data found in {folder_path}")
    #     return

    # train_sdv = calc_sdv(train_means, folder_path, num_channels)
    # percentiles = calc_percentiles(folder_path, num_channels, [1, 99])
    
    train_means, train_sdv, percentiles = calc_statistics_multi(folder_path, num_channels, [1, 99])
    
    if not os.path.exists(stats_folder):
        os.makedirs(stats_folder)
    
    mean_path = os.path.join(stats_folder, "train_means.npy")
    sdv_path = os.path.join(stats_folder, "train_sdv.npy")
    percentiles_path = os.path.join(stats_folder, "train_percentiles.npy")
    
    np.save(mean_path, train_means)
    print("Means for each channel: ", train_means)
    np.save(sdv_path, train_sdv)
    print("Standard Deviation for each channel: ", train_sdv)
    np.save(percentiles_path, percentiles)
    print("Percentiles (1st and 99th) for each channel:", percentiles)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the mean, standard deviation, and percentiles of each channel in our dataset")
    parser.add_argument("root_path", type=str, help="The path to the root directory containing the dataset")
    parser.add_argument("num_channels", type=int, help="The number of channels in the dataset")
    parser.add_argument("--separate", action="store_true", help="Calculate statistics separately for each subfolder")
    args = parser.parse_args()
    root_path = args.root_path
    num_channels = args.num_channels
    separate = args.separate

    if separate:
        for subfolder in os.listdir(root_path):
            subfolder_path = os.path.join(root_path, subfolder)
            if os.path.isdir(subfolder_path):
                required_folders = ["normal", "case_1", "case_2", "case_3"]
                if all(os.path.isdir(os.path.join(subfolder_path, folder)) for folder in required_folders):
                    # Check if any of the required folders are not empty
                    if any(len(os.listdir(os.path.join(subfolder_path, folder))) > 0 for folder in required_folders):
                        stats_folder = os.path.join(subfolder_path, "data_stats")
                        process_folder(subfolder_path, num_channels, stats_folder)
                    else:
                        print(f"Skipping {subfolder_path} as all required folders are empty")
                else:
                    print(f"Skipping {subfolder_path} as it doesn't contain all required folders")
    else:
        stats_folder = os.path.join(root_path, "data_stats")
        process_folder(root_path, num_channels, stats_folder)