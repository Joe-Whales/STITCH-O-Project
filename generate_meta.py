import os
import json
import numpy as np
import argparse

# Set the root directory where your dataset is located
root_dir = "chunks"

# Define the class names and their corresponding labels
class_labels = {
    "normal": 0,
    "case_1": 1,
    "case_2": 1,
    "case_3": 1
}

# Function to read mean and std values from data_stats folder
def read_stats(orchard_path):
    stats_path = os.path.join(orchard_path, "data_stats")
    mean_path = os.path.join(stats_path, "train_means.npy")
    std_path = os.path.join(stats_path, "train_sdv.npy")
    
    if os.path.exists(mean_path) and os.path.exists(std_path):
        mean = np.load(mean_path).tolist()
        std = np.load(std_path).tolist()
        return mean, std
    else:
        return None, None

# Function to process files in a directory
def process_directory(dir_path, class_name, split, orchard_name, mean, std):
    metadata = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".npy"):
            relative_path = os.path.join(orchard_name, split, class_name, filename)
            metadata_entry = {
                "filename": relative_path,
                "label": class_labels[class_name],
                "label_name": "good" if class_labels[class_name] == 0 else "defective",
                "clsname": orchard_name,
                "mean": mean,
                "std": std
            }
            metadata.append(metadata_entry)
    return metadata

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate metadata for orchard dataset")
parser.add_argument("--case", choices=["case_1", "case_2", "case_3"], help="Filter test metadata by specific case")
args = parser.parse_args()

# Process all orchards
all_metadata = {"train": [], "test": []}

for orchard in os.listdir(root_dir):
    orchard_path = os.path.join(root_dir, orchard)
    if os.path.isdir(orchard_path):
        mean, std = read_stats(orchard_path)
        
        # Process training data
        train_dir = os.path.join(orchard_path, "train")
        if os.path.exists(train_dir):
            normal_dir = os.path.join(train_dir, "normal")
            if os.path.exists(normal_dir):
                all_metadata["train"].extend(process_directory(normal_dir, "normal", "train", orchard, mean, std))
        
        # Process test data
        test_dir = os.path.join(orchard_path, "test")
        if os.path.exists(test_dir):
            for class_name in class_labels.keys():
                if args.case and class_name != "normal" and class_name != args.case:
                    continue
                class_dir = os.path.join(test_dir, class_name)
                if os.path.exists(class_dir):
                    all_metadata["test"].extend(process_directory(class_dir, class_name, "test", orchard, mean, std))

# Create metadata folder if it doesn't exist
metadata_dir = os.path.join(root_dir, "metadata")
if not os.path.exists(metadata_dir):
    os.makedirs(metadata_dir)

# Save the training metadata to a JSON file
train_output_file = os.path.join(metadata_dir, "train_metadata.json")
with open(train_output_file, "w") as f:
    for entry in all_metadata["train"]:
        json.dump(entry, f)
        f.write("\n")

# Save the test metadata to a JSON file
test_output_file = os.path.join(metadata_dir, "test_metadata.json")
with open(test_output_file, "w") as f:
    for entry in all_metadata["test"]:
        json.dump(entry, f)
        f.write("\n")

print(f"Training metadata file '{train_output_file}' has been generated successfully.")
print(f"Test metadata file '{test_output_file}' has been generated successfully.")