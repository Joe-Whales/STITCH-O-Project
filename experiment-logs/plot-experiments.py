import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def load_specific_files(root_dir):
    """
    Load specific CSV files for different models from the root directory.

    Args:
        root_dir (str): Path to the root directory containing experiment folders.

    Returns:
        dict: A dictionary where keys are model names and values are pandas DataFrames of the CSV data.
    """
    specific_files = {
        'Baseline': os.path.join(root_dir, 'UniAD vs Baseline Model', 'Baseline.csv'),
        'UniAD (dropout and jitter)':os.path.join(root_dir, 'UniAD vs Baseline Model', 'UniAD.csv'),
        'UniAD (ResNet50)': os.path.join(root_dir, 'ResNet50 vs EfficientNet-b4', 'UniAD-ResNet50.csv'),
        'UniAD Multi-Layer': os.path.join(root_dir, 'Multi-Layer vs Single-Layer', 'UniAD-Multi-Layer.csv'),
        'UniAD (dropout, no jitter)': os.path.join(root_dir, 'Jitter and Dropout', 'UniAD-dropout-no-jitter.csv'),
        'UniAD (jitter)': os.path.join(root_dir, 'Jitter and Dropout', 'UniAD-jitter.csv'),
        'UniAD (no jitter)': os.path.join(root_dir, 'Jitter and Dropout', 'UniAD-no-jitter.csv'),
        'UniAD Large Chunks': os.path.join(root_dir, 'Small vs Large chunks', 'Large chunks UniAD.csv')
    }
    
    data = {}
    for model, file_paths in specific_files.items():
        if isinstance(file_paths, list):
            # For UniAD, we'll use the first file that exists
            for file_path in file_paths:
                if os.path.exists(file_path):
                    data[model] = pd.read_csv(file_path)
                    break
        else:
            if os.path.exists(file_paths):
                data[model] = pd.read_csv(file_paths)
            else:
                print(f"Warning: File not found - {file_paths}")
    
    return data

def plot_overall_comparison(data):
    """
    Plot overall comparison graphs for Case 1 and Case 2 AUROC across all models.

    Args:
        data (dict): A dictionary where keys are model names and values are pandas DataFrames of the CSV data.

    This function creates two plots:
    1. Case 1 AUROC comparison across all models
    2. Case 2 AUROC comparison across all models
    Each plot shows the AUROC values over epochs for different models, with the highest point marked and annotated.
    """
    for case in ['Case 1', 'Case 2']:
        plt.figure(figsize=(15, 10))
        column = f'all_case_{case[-1]}_AUROC'
        
        for model, df in data.items():
            if column in df.columns:
                plt.plot(df['Epoch'], df[column], label=model, linewidth=2)
                
                # Mark the highest point
                max_idx = df[column].idxmax()
                plt.plot(df['Epoch'][max_idx], df[column][max_idx], 'o', markersize=8)
                plt.annotate(f'{df[column][max_idx]:.3f}', 
                             (df['Epoch'][max_idx], df[column][max_idx]),
                             xytext=(5, 5), textcoords='offset points', fontsize=12)
        
        plt.title(f'{case} AUROC Comparison Across All Models', fontsize=24)
        plt.xlabel('Epoch', fontsize=18)
        plt.ylabel('AUROC', fontsize=18)
        plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

def process_csv_files(directory):
    """
    Process all CSV files in the given directory and its subdirectories.

    Args:
        directory (str): Path to the root directory to search for CSV files.

    Returns:
        dict: A nested dictionary where the first level keys are experiment names,
              the second level keys are model names, and the values are pandas DataFrames of the CSV data.
    """
    experiments = {}
    for root, dirs, files in os.walk(directory):
        if files:
            experiment_name = os.path.basename(root)
            csv_files = [f for f in files if f.endswith('.csv')]
            if csv_files:
                experiments[experiment_name] = {}
                for csv_file in csv_files:
                    model_name = os.path.splitext(csv_file)[0]
                    df = pd.read_csv(os.path.join(root, csv_file))
                    experiments[experiment_name][model_name] = df
    return experiments

def plot_experiment_results(experiment_name, results):
    """
    Plot the results for a single experiment.

    Args:
        experiment_name (str): Name of the experiment.
        results (dict): A dictionary where keys are model names and values are pandas DataFrames of the CSV data.

    This function creates a figure with three subplots:
    1. Case 1 AUROC
    2. Case 2 AUROC
    3. Average AUROC
    Each subplot shows the AUROC values over epochs for different models in the experiment, 
    with the highest point marked and annotated.
    """
    fig = plt.figure(figsize=(15, 20))  # Adjusted figure height
    
    # Create 7 subplot spaces
    gs = fig.add_gridspec(7, 1)
    
    # Title takes up the first subplot space
    title_ax = fig.add_subplot(gs[0])
    title_ax.axis('off')
    title_ax.text(0.5, 0.5, f'Experiment: {experiment_name}', 
                  fontsize=24, ha='center', va='center')
    
    # Create the three main subplots, each taking up 2 subplot spaces
    axes = [fig.add_subplot(gs[1:3]), fig.add_subplot(gs[3:5]), fig.add_subplot(gs[5:7])]
    
    titles = ['Case 1 AUROC', 'Case 2 AUROC', 'Average AUROC']
    columns = ['all_case_1_AUROC', 'all_case_2_AUROC', 'average_AUROC']
    colors = plt.colormaps['Set1'](np.linspace(0, 1, len(results)))

    for i, (ax, title, column) in enumerate(zip(axes, titles, columns)):
        min_val, max_val = float('inf'), float('-inf')
        for (model, df), color in zip(results.items(), colors):
            if column == 'average_AUROC':
                if 'all_case_1_AUROC' in df.columns and 'all_case_2_AUROC' in df.columns:
                    df[column] = (df['all_case_1_AUROC'] + df['all_case_2_AUROC']) / 2
                else:
                    continue
            if column in df.columns:
                line = ax.plot(df['Epoch'], df[column], label=model, color=color, linewidth=2)
                
                # Update min and max values
                min_val = min(min_val, df[column].min())
                max_val = max(max_val, df[column].max())
                
                # Mark the highest point
                max_idx = df[column].idxmax()
                ax.plot(df['Epoch'][max_idx], df[column][max_idx], 'o', color=color, markersize=10)
                ax.annotate(f'{df[column][max_idx]:.3f}', 
                            (df['Epoch'][max_idx], df[column][max_idx]),
                            xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        # Set y-axis limits with a small padding
        y_range = max_val - min_val
        ax.set_ylim(max(0, min_val - 0.05 * y_range), min(1, max_val + 0.05 * y_range))
        
        ax.set_title(title, fontsize=18, pad=20)
        ax.set_ylabel('AUROC', fontsize=26, labelpad=10)
        ax.legend(fontsize=22, loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Improve tick label readability
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Only set xlabel for the bottom subplot
        if i == 2:  # Last subplot
            ax.set_xlabel('Epoch', fontsize=14, labelpad=10)
        else:
            ax.set_xlabel('')  # Remove x-label from other subplots

    plt.tight_layout()
    plt.subplots_adjust(hspace=1.5)  # Adjust space between subplots
    plt.show()
    
def main(root_dir):
    """
    Main function to process CSV files and generate plots.

    Args:
        root_dir (str): Path to the root directory containing experiment folders with CSV files.

    This function:
    1. Loads specific files for overall comparison
    2. Plots overall comparison graphs
    3. Processes all CSV files in the directory and its subdirectories
    4. Plots individual experiment results
    """
    # Load specific files for overall comparison
    specific_data = load_specific_files(root_dir)
    
    # Plot overall comparison graphs
    plot_overall_comparison(specific_data)

    # Process all CSV files for individual experiment plots
    experiments = process_csv_files(root_dir)

    if not experiments:
        print("No experiments with CSV files found in the specified directory or its subdirectories.")
        return

    # Plot individual experiment results
    for experiment_name, results in experiments.items():
        plot_experiment_results(experiment_name, results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot AUROC graphs from CSV files in experiment subfolders.")
    parser.add_argument("root_dir", type=str, help="Root directory containing experiment subfolders with CSV files")
    args = parser.parse_args()

    main(args.root_dir)