import os
import json

# Set the root directory where your dataset is located
root_dir = "chunks"

# Define the class names and their corresponding labels
class_labels = {
    "normal": 0,
    "case_1": 1,
    "case_2": 1
}

# Function to process files in a directory
def process_directory(dir_path, class_name, split):
    metadata = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".npy"):
            relative_path = f"orchard/{split}/{class_name}/{filename}"
            metadata_entry = {
                "filename": relative_path,
                "label": class_labels[class_name],
                "label_name": "good" if class_labels[class_name] == 0 else "defective",
                "clsname": "orchard"
            }
            metadata.append(metadata_entry)
    return metadata

# Process the training data
train_metadata = []
train_dir = os.path.join(root_dir, "train")
normal_dir = os.path.join(train_dir, "normal")
train_metadata.extend(process_directory(normal_dir, "normal", "train"))

# Process the validation (test) data
test_metadata = []
val_dir = os.path.join(root_dir, "val")
for class_name in class_labels.keys():
    class_dir = os.path.join(val_dir, class_name)
    if os.path.exists(class_dir):
        test_metadata.extend(process_directory(class_dir, class_name, "test"))

# Save the training metadata to a JSON file
train_output_file = "train_metadata.json"
with open(train_output_file, "w") as f:
    for entry in train_metadata:
        json.dump(entry, f)
        f.write("\n")

# Save the test metadata to a JSON file
test_output_file = "test_metadata.json"
with open(test_output_file, "w") as f:
    for entry in test_metadata:
        json.dump(entry, f)
        f.write("\n")

print(f"Training metadata file '{train_output_file}' has been generated successfully.")
print(f"Test metadata file '{test_output_file}' has been generated successfully.")