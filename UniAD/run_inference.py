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
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    return update_config(config)

def encode_pred(preds):
    # Assuming preds shape is (B, C, H, W) or (B, H, W)
    if len(preds.shape) == 4:
        return preds.reshape(preds.shape[0], -1).mean(axis=1)  # (B, )
    elif len(preds.shape) == 3:
        return preds.reshape(preds.shape[0], -1).mean(axis=1)  # (B, )
    else:
        raise ValueError(f"Unexpected shape for preds: {preds.shape}")

def classify_sample(anomaly_score, thresholds):
    lower_threshold = thresholds['all_case_1']
    upper_threshold = thresholds['all_case_2']
    
    if anomaly_score < lower_threshold:
        return "Anomalous (Below Threshold)"
    elif anomaly_score > upper_threshold:
        return "Anomalous (Above Threshold)"
    else:
        return "Normal"

def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Initialize model
    model = ModelHelper(config.net)
    model.cuda()
    
    # Load best checkpoint and thresholds
    checkpoint_path = os.path.join(config.saver.save_dir, "ckpt_best.pth.tar")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    thresholds = checkpoint.get('thresholds', {})
    model.eval()
    
    # Ensure we have the required thresholds
    if 'all_case_1' not in thresholds or 'all_case_2' not in thresholds:
        raise ValueError("Checkpoint does not contain required thresholds for all_case_1 and all_case_2")
    
    # Build data loader
    train_loader, val_loader = build_dataloader(config.dataset, distributed=False)
    
    # Run inference
    results = defaultdict(lambda: {"normal": 0, "anomalous_below": 0, "anomalous_above": 0, "total": 0})
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Processing samples"):
            outputs = model(batch)
            preds = outputs["pred"].cpu().numpy()
            
            anomaly_scores = encode_pred(preds)
            clsnames = batch["clsname"]
            
            for score, clsname in zip(anomaly_scores, clsnames):
                classification = classify_sample(score, thresholds)
                results[clsname]["total"] += 1
                if classification == "Normal":
                    results[clsname]["normal"] += 1
                elif classification == "Anomalous (Below Threshold)":
                    results[clsname]["anomalous_below"] += 1
                else:
                    results[clsname]["anomalous_above"] += 1
    
    # Print results
    print("\nResults per class:")
    for clsname, counts in results.items():
        print(f"\nClass: {clsname}")
        print(f"Total samples: {counts['total']}")
        print(f"Normal samples: {counts['normal']} ({counts['normal']/counts['total']*100:.2f}%)")
        print(f"Anomalous samples (Below Threshold): {counts['anomalous_below']} ({counts['anomalous_below']/counts['total']*100:.2f}%)")
        print(f"Anomalous samples (Above Threshold): {counts['anomalous_above']} ({counts['anomalous_above']/counts['total']*100:.2f}%)")
    
    print(f"\nThresholds used:")
    print(f"Lower (all_case_1): {thresholds['all_case_1']}")
    print(f"Upper (all_case_2): {thresholds['all_case_2']}")

    # Calculate and print overall results
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