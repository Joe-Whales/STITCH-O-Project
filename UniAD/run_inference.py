import argparse
import os
import yaml
import torch
import numpy as np
from easydict import EasyDict
from tqdm import tqdm
from collections import defaultdict

from models.model_helper import ModelHelper
from utils.misc_helper import load_state, update_config
from datasets.data_builder import build_dataloader

def load_config(config_path):
    """
    Load and update the configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        EasyDict: Updated configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    return update_config(config)

def encode_pred(preds):
    """
    Encode predictions by reshaping and calculating the mean.

    Args:
        preds (numpy.ndarray): Prediction array of shape (batch_size, ...).

    Returns:
        numpy.ndarray: Encoded predictions of shape (batch_size,).

    Raises:
        ValueError: If the input shape is unexpected.
    """
    if len(preds.shape) == 4:
        return preds.reshape(preds.shape[0], -1).mean(axis=1)
    elif len(preds.shape) == 3:
        return preds.reshape(preds.shape[0], -1).mean(axis=1)
    else:
        raise ValueError(f"Unexpected shape for preds: {preds.shape}")

def classify_sample(anomaly_score, thresholds):
    """
    Classify a sample based on its anomaly score and given thresholds.

    Args:
        anomaly_score (float): The anomaly score of the sample.
        thresholds (dict): Dictionary containing 'all_case_1' and 'all_case_2' threshold values.

    Returns:
       
        str: Classification result - "Anomalous (Below Threshold)", "Anomalous (Above Threshold)", or "Normal".
    """
    lower_threshold = thresholds['all_case_1']
    upper_threshold = thresholds['all_case_2']
    
    if anomaly_score < lower_threshold:
        return "Anomalous (Below Threshold)"
    elif anomaly_score > upper_threshold:
        return "Anomalous (Above Threshold)"
    else:
        return "Normal"

def calculate_orchard_metrics(anomaly_scores, classifications):
    """
    Calculate various metrics for an orchard based on anomaly scores and classifications.

    Args:
        anomaly_scores (list): List of anomaly scores for the orchard.
        classifications (list): List of classifications for the orchard samples.

    Returns:
        tuple: Contains mean_score, top5_score, bottom5_score, and anomalous_percentage.
    """    
    mean_score = np.mean(anomaly_scores)
    top5_score = np.mean(sorted(anomaly_scores, reverse=True)[:5])
    bottom5_score = np.mean(sorted(anomaly_scores)[:5])
    anomalous_percentage = (sum(1 for result in classifications if result != "Normal") / len(classifications)) * 100
    return mean_score, top5_score, bottom5_score, anomalous_percentage

def classify_orchard(orchard_data):
    """
    Classify an orchard based on various metrics and provide insights.

    Args:
        orchard_data (dict): Dictionary containing orchard metrics including anomaly scores and percentages.

    Returns:
        str: Classification result and insights about the orchard.
    """
    bottom_5_avg = orchard_data['Bottom 5 mean anomaly score']
    top_5_avg = orchard_data['Top 5 mean anomaly score']
    mean_score = orchard_data['Mean anomaly score']
    anomalous_percentage = orchard_data['Percentage of anomalous chunks']
    percent_above = orchard_data['Anomalous samples (Above Threshold)']
    percent_below = orchard_data['Anomalous samples (Below Threshold)']
    
    score_range = top_5_avg - bottom_5_avg

    case_1_score = 0
    case_2_score = 0

    # Scoring
    if bottom_5_avg < 21:
        case_1_score += 2
    if percent_below > 10:
        case_1_score += 2
    if percent_above > 15:
        case_2_score += 2
    if top_5_avg > 75:
        case_2_score += 2
    elif top_5_avg > 70:
        case_2_score += 1
    if mean_score > 50:
        case_2_score += 1
    if anomalous_percentage > 50:
        case_1_score += 1
        case_2_score += 1

    # Classification
    classification = "Anomalous" if case_1_score + case_2_score > 1 else "Normal"

    # Insights
    insights = []
    if mean_score > 45:
        insights.append("High overall anomaly score")
    if case_1_score > case_2_score + 2:
        insights.append("High Case 1, low Case 2")
    elif case_2_score > case_1_score + 2:
        insights.append("High Case 2, low Case 1")
    if score_range > 40:
        insights.append("Wide range of anomaly scores")
    if top_5_avg > 70:
        insights.append("High top 5 average")
    if bottom_5_avg < 21:
        insights.append("Low bottom 5 average")

    result = f"{classification}\n"
    if insights:
        result += "Insights: " + ", ".join(insights)
    
    return result

def main(args):
    """
    Main function to run inference on the validation loader.

    Args:
        args (argparse.Namespace): Command-line arguments containing the configuration file path.

    This function performs the following steps:
    1. Load the configuration and model
    2. Process samples using the model
    3. Classify samples and calculate metrics
    4. Print results for each class and overall statistics
    """
    config = load_config(args.config)
    
    model = ModelHelper(config.net)
    model.cuda()
    
    checkpoint_path = os.path.join(config.saver.save_dir, "ckpt_best.pth.tar")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    thresholds = checkpoint.get('thresholds', {})
    model.eval()
    
    if 'all_case_1' not in thresholds or 'all_case_2' not in thresholds:
        raise ValueError("Checkpoint does not contain required thresholds for all_case_1 and all_case_2")
    
    data_loader = build_dataloader(config.dataset, distributed=False, inference=True)
    
    results = defaultdict(lambda: {
        "normal": 0, "anomalous_below": 0, "anomalous_above": 0, "total": 0,
        "anomaly_scores": [], "classifications": []
    })
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing samples"):
            outputs = model(batch)
            preds = outputs["pred"].cpu().numpy()
            
            anomaly_scores = encode_pred(preds)
            clsnames = batch["clsname"]
            
            for score, clsname in zip(anomaly_scores, clsnames):
                classification = classify_sample(score, thresholds)
                results[clsname]["total"] += 1
                results[clsname]["anomaly_scores"].append(score)
                results[clsname]["classifications"].append(classification)
                if classification == "Normal":
                    results[clsname]["normal"] += 1
                elif classification == "Anomalous (Below Threshold)":
                    results[clsname]["anomalous_below"] += 1
                else:
                    results[clsname]["anomalous_above"] += 1
    
    print("\nResults per class:")
    for clsname, counts in results.items():
        print(f"\nClass: {clsname}")
        print(f"Normal: {counts['normal']/counts['total']*100:.2f}% | "
            f"Case 1: {counts['anomalous_below']/counts['total']*100:.2f}% | "
            f"Case 2: {counts['anomalous_above']/counts['total']*100:.2f}%")

        mean_score, top5_score, bottom5_score, anomalous_percentage = calculate_orchard_metrics(counts['anomaly_scores'], counts['classifications'])

        
        # New: Classify the orchard
        orchard_data = {
            "Bottom 5 mean anomaly score": bottom5_score,
            "Top 5 mean anomaly score": top5_score,
            "Mean anomaly score": mean_score,
            "Percentage of anomalous chunks": anomalous_percentage,
            "Anomalous samples (Below Threshold)": counts['anomalous_below']/counts['total']*100,
            "Anomalous samples (Above Threshold)": counts['anomalous_above']/counts['total']*100
        }
        orchard_classification = classify_orchard(orchard_data)
        print(f"Orchard Classification: {orchard_classification}")
    
    print(f"\nThresholds used:")
    print(f"Lower (all_case_1): {thresholds['all_case_1']}")
    print(f"Upper (all_case_2): {thresholds['all_case_2']}")

    overall_results = {
        "normal": sum(counts["normal"] for counts in results.values()),
        "anomalous_below": sum(counts["anomalous_below"] for counts in results.values()),
        "anomalous_above": sum(counts["anomalous_above"] for counts in results.values()),
    }
    total_samples = sum(counts["total"] for counts in results.values())
    
    print("\nOverall Results:")
    print(f"Total samples processed: {total_samples}")
    print(f"Normal samples: {overall_results['normal']} ({overall_results['normal']/total_samples*100:.2f}%)")
    print(f"Anomalous samples (Below Threshold): {overall_results['anomalous_below']} ({overall_results['anomalous_below']/total_samples*100:.2f}%)")
    print(f"Anomalous samples (Above Threshold): {overall_results['anomalous_above']} ({overall_results['anomalous_above']/total_samples*100:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on validation loader")
    parser.add_argument("--config", default="./train_config.yaml", help="Path to the configuration file")
    args = parser.parse_args()
    
    main(args)