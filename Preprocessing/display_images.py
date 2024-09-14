import os
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse

def load_and_display_images(orchard_path, num_images=36):
    train_path = os.path.join(orchard_path, 'train', 'normal')
    test_path = os.path.join(orchard_path, 'test')
    
    train_files = [f for f in os.listdir(train_path) if f.endswith('.npy')]
    test_files = []
    for case in ['normal', 'case_1', 'case_2', 'case_3']:
        case_path = os.path.join(test_path, case)
        if os.path.exists(case_path):
            test_files.extend([os.path.join(case, f) for f in os.listdir(case_path) if f.endswith('.npy')])
    
    # Randomly select images
    num_train = num_images // 2
    num_test = num_images - num_train
    selected_train = random.sample(train_files, min(num_train, len(train_files)))
    selected_test = random.sample(test_files, min(num_test, len(test_files)))
    
    # Create a 6x6 grid
    fig, axes = plt.subplots(6, 6, figsize=(15, 15))
    fig.suptitle(f"Random Images from {os.path.basename(orchard_path)}", fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < len(selected_train):
            img_path = os.path.join(train_path, selected_train[i])
            title = 'Train'
        elif i < len(selected_train) + len(selected_test):
            img_path = os.path.join(test_path, selected_test[i - len(selected_train)])
            title = 'Test: ' + os.path.dirname(selected_test[i - len(selected_train)])
        else:
            ax.axis('off')
            continue
        
        # Load and display image
        img = (np.load(img_path))
        ax.imshow(img)
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def main(root_dir):
    orchards = [orchard for orchard in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, orchard))]
    
    for orchard in orchards:
        orchard_path = os.path.join(root_dir, orchard)
        if os.path.exists(os.path.join(orchard_path, 'train')) and os.path.exists(os.path.join(orchard_path, 'test')):
            load_and_display_images(orchard_path)
        else:
            print(f"Skipping {orchard}: missing train or test folder")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display random images from orchards.")
    parser.add_argument("root_dir", type=str, help="Path to the root directory containing the processed orchard folders")
    args = parser.parse_args()
    
    main(args.root_dir)