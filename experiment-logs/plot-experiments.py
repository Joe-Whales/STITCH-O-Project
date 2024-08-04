import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def process_csv_files(directory):
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
    fig = plt.figure(figsize=(15, 30))  # Adjusted figure height
    
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
    colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, len(results)))

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
        ax.set_ylabel('AUROC', fontsize=14, labelpad=10)
        ax.legend(fontsize=12, loc='lower right')
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
    experiments = process_csv_files(root_dir)

    if not experiments:
        print("No experiments with CSV files found in the specified directory or its subdirectories.")
        return

    for experiment_name, results in experiments.items():
        plot_experiment_results(experiment_name, results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot AUROC graphs from CSV files in experiment subfolders.")
    parser.add_argument("root_dir", type=str, help="Root directory containing experiment subfolders with CSV files")
    args = parser.parse_args()

    main(args.root_dir)