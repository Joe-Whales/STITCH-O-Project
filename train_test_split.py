import os
import random
import shutil
import sys

def organize_folders(base_dir):
    for root, dirs, _ in os.walk(base_dir):
        if all(folder in dirs for folder in ['normal', 'case_1', 'case_2', 'case_3']):
            print(f"Organizing folder: {root}")
            reorganize_subfolder(root)

def reorganize_subfolder(folder_path):
    original_dir = os.getcwd()
    os.chdir(folder_path)
    
    # Remove existing train and test folders if they exist
    for dir_name in ['train', 'test']:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
    
    os.makedirs('train', exist_ok=True)
    os.makedirs('test', exist_ok=True)
    os.makedirs('test/normal', exist_ok=True)
    
    if os.path.exists('normal'):
        shutil.move('normal', 'train/normal')
    
    case_counts = []
    for case in ['case_1', 'case_2', 'case_3']:
        if os.path.exists(case):
            if not os.listdir(case):  # Check if the case folder is empty
                print(f"{case} is empty. Removing it.")
                shutil.rmtree(case)
            else:
                shutil.move(case, f'test/{case}')
                npy_count = len([f for f in os.listdir(f'test/{case}') if f.endswith('.npy')])
                case_counts.append(npy_count)
                print(f"{case} contains {npy_count} .npy files")
    
    if case_counts:
        avg_count = sum(case_counts) // len(case_counts)
        print(f"Average .npy count: {avg_count}")
        normal_files = [f for f in os.listdir('train/normal') if f.endswith('.npy')]
        files_to_move = random.sample(normal_files, min(avg_count, len(normal_files)//4))
        
        for file in files_to_move:
            shutil.move(os.path.join('train/normal', file), os.path.join('test/normal', file))
        
        print(f"Moved {len(files_to_move)} .npy files from train/normal to test/normal")
    else:
        print("No non-empty case folders found. No files moved to test/normal.")
    
    print("Folders reorganized successfully.")
    os.chdir(original_dir)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py /path/to/directory")
        sys.exit(1)
    
    base_directory = sys.argv[1]
    if not os.path.isdir(base_directory):
        print(f"Error: {base_directory} is not a valid directory")
        sys.exit(1)
    
    organize_folders(base_directory)