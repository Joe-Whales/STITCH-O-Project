import os
import json
import numpy as np
import argparse
import random
from collections import defaultdict

# Define the class names and their corresponding labels
class_labels = {
    "normal": 0,
    "case_1": 1,
    "case_2": 1,
    "case_3": 1
}

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
                "std": std,
                "case": class_name
            }
            metadata.append(metadata_entry)
    return metadata

def main(root_dir, verbose, totals):
    all_metadata = {"train": [], "test": []}
    case_metadata = defaultdict(lambda: defaultdict(list))
    normal_metadata = defaultdict(list)

    for orchard in os.listdir(root_dir):
        orchard_path = os.path.join(root_dir, orchard)
        if os.path.isdir(orchard_path):
            mean, std = read_stats(orchard_path)
            
            train_dir = os.path.join(orchard_path, "train")
            if os.path.exists(train_dir):
                normal_dir = os.path.join(train_dir, "normal")
                if os.path.exists(normal_dir):
                    all_metadata["train"].extend(process_directory(normal_dir, "normal", "train", orchard, mean, std))
            
            test_dir = os.path.join(orchard_path, "test")
            if os.path.exists(test_dir):
                for class_name in class_labels.keys():
                    class_dir = os.path.join(test_dir, class_name)
                    if os.path.exists(class_dir):
                        processed_data = process_directory(class_dir, class_name, "test", orchard, mean, std)
                        if class_name != "normal":
                            case_metadata[class_name][orchard].extend(processed_data)
                        else:
                            normal_metadata[orchard].extend(processed_data)

    ordered_test_metadata = []

    # Collect statistics for each case and all cases
    statistics = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for case_name in sorted(case_metadata.keys()):
        all_case_data = []
        for orchard, case_data in case_metadata[case_name].items():
            if orchard in normal_metadata and normal_metadata[orchard]:
                needed_normal = len(case_data)
                sampled_normal = random.sample(normal_metadata[orchard], min(needed_normal, len(normal_metadata[orchard])))
                all_case_data.extend(case_data)
                all_case_data.extend(sampled_normal)
                
                statistics[case_name][orchard]['normal'] = len(sampled_normal)
                statistics[case_name][orchard]['defective'] = len(case_data)
                statistics[f'all_{case_name}'][orchard]['normal'] += len(sampled_normal)
                statistics[f'all_{case_name}'][orchard]['defective'] += len(case_data)

        if all_case_data:
            all_case_clsname = f"all_{case_name}"
            for entry in all_case_data:
                new_entry = entry.copy()
                new_entry["clsname"] = all_case_clsname
                ordered_test_metadata.append(new_entry)

    if not totals:
        # Then, add orchard_case_x entries
        for case_name in sorted(case_metadata.keys()):
            for orchard in sorted(case_metadata[case_name].keys()):
                case_data = case_metadata[case_name][orchard]
                if orchard in normal_metadata and normal_metadata[orchard]:
                    needed_normal = len(case_data)
                    sampled_normal = random.sample(normal_metadata[orchard], min(needed_normal, len(normal_metadata[orchard])))
                    
                    orchard_case_clsname = f"{orchard}_{case_name}"
                    for entry in case_data + sampled_normal:
                        new_entry = entry.copy()
                        new_entry["clsname"] = orchard_case_clsname
                        ordered_test_metadata.append(new_entry)
                else:
                    print(f"Warning: No normal samples for orchard {orchard}. Skipping {case_name} for this orchard.")

    # Print statistics if in verbose mode
    if verbose:
        print("\nStatistics for each case:")
        for case_name in sorted(statistics.keys()):
            print(f"\n{case_name.upper()}:")
            for orchard in sorted(statistics[case_name].keys()):
                print(f"  {orchard}:")
                print(f"    Normal: {statistics[case_name][orchard]['normal']}")
                print(f"    Defective: {statistics[case_name][orchard]['defective']}")

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

    # Save the ordered test metadata to a JSON file
    test_output_file = os.path.join(metadata_dir, "test_metadata.json")
    with open(test_output_file, "w") as f:
        for entry in ordered_test_metadata:
            json.dump(entry, f)
            f.write("\n")

    print(f"Training metadata file '{train_output_file}' has been generated successfully.")
    print(f"Test metadata file '{test_output_file}' has been generated successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metadata for orchard dataset")
    parser.add_argument("root_dir", type=str, help="Root directory where your dataset is located")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed statistics")
    parser.add_argument("-t", "--totals", action="store_true", help="Print total statistics only")
    args = parser.parse_args()

    main(args.root_dir, args.verbose, args.totals)
