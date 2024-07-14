import os
import json
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count

# Set the root directory where your dataset is located
root_dir = "chunks"

# Define the class names and their corresponding labels
class_labels = {
    "normal": 0,
    "case_1": 1,
    "case_2": 1,
    "case_3": 1
}

def process_chunk(args):
    file_path, relative_path, class_name, orchard_name = args
    try:
        data = np.load(file_path)
        
        # Ensure data is 3D
        if data.ndim != 3:
            print(f"Warning: {file_path} has unexpected dimensions {data.shape}, skipping")
            return None
        
        height, width, channels = data.shape
        
        # Process each channel independently
        scaled_data = np.zeros((height, width, channels), dtype=np.float32)
        channel_means = []
        channel_stds = []
        
        for c in range(channels):
            channel_data = data[:,:,c].astype(np.float32)
            
            # Calculate percentiles
            min_val, max_val = np.percentile(channel_data, [1, 99])
            
            # Scale data
            scale = 255.0 / (max_val - min_val)
            offset = -min_val * scale
            channel_scaled = np.clip(channel_data * scale + offset, 0, 255)
            
            scaled_data[:,:,c] = channel_scaled
            
            # Calculate mean and std of scaled data
            channel_means.append(float(np.mean(channel_scaled)))
            channel_stds.append(float(np.std(channel_scaled)))
        
        metadata_entry = {
            "filename": relative_path,
            "label": class_labels[class_name],
            "label_name": "good" if class_labels[class_name] == 0 else "defective",
            "clsname": orchard_name,
            "mean": channel_means,
            "std": channel_stds
        }
        
        # Save the scaled data
        np.save(file_path, scaled_data)
        
        return metadata_entry
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def process_directory(dir_path, class_name, split, orchard_name):
    file_list = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".npy"):
            file_path = os.path.join(dir_path, filename)
            relative_path = os.path.join(orchard_name, split, class_name, filename)
            file_list.append((file_path, relative_path, class_name, orchard_name))
    return file_list

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate metadata for orchard dataset")
    parser.add_argument("--case", choices=["case_1", "case_2", "case_3"], help="Filter test metadata by specific case")
    args = parser.parse_args()

    # Process all orchards
    all_files = []

    for orchard in os.listdir(root_dir):
        orchard_path = os.path.join(root_dir, orchard)
        if os.path.isdir(orchard_path):
            # Process training data
            train_dir = os.path.join(orchard_path, "train")
            if os.path.exists(train_dir):
                normal_dir = os.path.join(train_dir, "normal")
                if os.path.exists(normal_dir):
                    all_files.extend(process_directory(normal_dir, "normal", "train", orchard))
            
            # Process test data
            test_dir = os.path.join(orchard_path, "test")
            if os.path.exists(test_dir):
                for class_name in class_labels.keys():
                    if args.case and class_name != "normal" and class_name != args.case:
                        continue
                    class_dir = os.path.join(test_dir, class_name)
                    if os.path.exists(class_dir):
                        all_files.extend(process_directory(class_dir, class_name, "test", orchard))

    # Process chunks in parallel
    with Pool(processes=cpu_count()) as pool:
        metadata = pool.map(process_chunk, all_files)

    # Remove None values (failed processing)
    metadata = [entry for entry in metadata if entry is not None]

    # Separate train and test metadata
    train_metadata = [entry for entry in metadata if "train" in entry["filename"]]
    test_metadata = [entry for entry in metadata if "test" in entry["filename"]]

    # Create metadata folder if it doesn't exist
    metadata_dir = os.path.join(root_dir, "metadata")
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)

    # Save the training metadata to a JSON file
    train_output_file = os.path.join(metadata_dir, "train_metadata.json")
    with open(train_output_file, "w") as f:
        for entry in train_metadata:
            json.dump(entry, f)
            f.write("\n")

    # Save the test metadata to a JSON file
    test_output_file = os.path.join(metadata_dir, "test_metadata.json")
    with open(test_output_file, "w") as f:
        for entry in test_metadata:
            json.dump(entry, f)
            f.write("\n")

    print(f"Training metadata file '{train_output_file}' has been generated successfully.")
    print(f"Test metadata file '{test_output_file}' has been generated successfully.")

if __name__ == "__main__":
    main()