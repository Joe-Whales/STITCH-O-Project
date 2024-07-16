import os
import numpy as np
import concurrent.futures
import multiprocessing

def process_orchard(orchard_path):
    print(f"Processing orchard: {orchard_path}")
    
    # Load stats
    stats_path = os.path.join(orchard_path, 'data_stats')
    min_max = np.load(os.path.join(stats_path, 'train_percentiles.npy'))
    min_vals = min_max[:, 0].astype(np.float32)
    max_vals = min_max[:, 1].astype(np.float32)
    mean_vals = np.load(os.path.join(stats_path, 'train_means.npy')).astype(np.float32)
    std_vals = np.load(os.path.join(stats_path, 'train_sdv.npy')).astype(np.float32)

    # Calculate scaling factors
    scale = 255.0 / (max_vals - min_vals)
    offset = -min_vals * scale

    # Define normalization function
    def normalize_and_scale(img):
        img = img.astype(np.float32)
        img_scaled = np.clip(img * scale + offset, 0, 255)
        return img_scaled

    # Process train folder
    train_path = os.path.join(orchard_path, 'train', 'normal')
    process_folder(train_path, normalize_and_scale)

    # Process test folder
    test_path = os.path.join(orchard_path, 'test')
    for case in ['normal', 'case_1', 'case_2', 'case_3']:
        case_path = os.path.join(test_path, case)
        if os.path.exists(case_path):
            process_folder(case_path, normalize_and_scale)

    # Adjust mean and std for each channel
    new_mean = mean_vals * scale + offset
    new_std = std_vals * scale

    # Save new stats
    np.save(os.path.join(stats_path, 'train_means.npy'), new_mean)
    np.save(os.path.join(stats_path, 'train_sdv.npy'), new_std)

def process_file(args):
    file_path, process_func = args
    img = np.load(file_path)
    processed_img = process_func(img)
    np.save(file_path, processed_img)  # This overwrites the original file

def process_folder(folder_path, process_func):
    files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    args = [(os.path.join(folder_path, f), process_func) for f in files]

    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        list(executor.map(process_file, args))

def main():
    root_dir = 'chunks/'
    orchards = [orchard for orchard in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, orchard))]

    for orchard in orchards:
        orchard_path = os.path.join(root_dir, orchard)
        process_orchard(orchard_path)

if __name__ == "__main__":
    main()