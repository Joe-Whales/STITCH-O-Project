import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
    
def get_metrics(labels, losses):
    """
    Calculate various performance metrics based on true labels and predicted losses.

    Args:
        labels (array-like): True binary labels.
        losses (array-like): Predicted loss scores.

    Returns:
        tuple: A tuple containing:
            - accuracy (float): The accuracy score.
            - precision (float): The precision score.
            - recall (float): The recall score.
            - f1 (float): The F1 score.
            - best_threshold (float): The best threshold for classification based on Youden's J statistic.
    """
    fpr, tpr, thresholds = roc_curve(labels, losses)
    j_scores = tpr - fpr
    best_threshold_index = np.argmax(j_scores)
    best_threshold = thresholds[best_threshold_index]
    predicted_labels = (losses >= best_threshold).astype(int)
    accuracy = accuracy_score(labels, predicted_labels)
    precision = precision_score(labels, predicted_labels)
    recall = recall_score(labels, predicted_labels)
    f1 = f1_score(labels, predicted_labels)
    return (accuracy, precision, recall, f1, best_threshold)

def plot_roc_curve(labels, losses, auroc):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    Args:
        labels (array-like): True binary labels.
        losses (array-like): Predicted loss scores.
        auroc (float): The Area Under the ROC Curve score.

    This function creates a plot showing:
    - The ROC curve
    - A diagonal reference line
    - The point corresponding to the best threshold
    - An annotation showing the best threshold value
    The plot includes labels, a title, and a legend.
    """
    fpr, tpr, thresholds = roc_curve(labels, losses)
    j_scores = tpr - fpr
    best_threshold_index = np.argmax(j_scores)
    best_threshold = thresholds[best_threshold_index]
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, label=f'ROC curve (AUROC = {auroc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.scatter(fpr[best_threshold_index], tpr[best_threshold_index], color='red', s=80, label='Best threshold')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    plt.annotate(f'Best threshold: {best_threshold:.2f}', 
             xy=(fpr[best_threshold_index], tpr[best_threshold_index]), 
             xytext=(0.6, 0.2),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )

    plt.show()

def performances(labels, losses, verbose=False):
    """
    Calculate and print performance metrics, optionally plot the ROC curve.

    Args:
        labels (array-like): True binary labels.
        losses (array-like): Predicted loss scores.
        verbose (bool, optional): If True, print additional metrics and plot the ROC curve. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - auroc (float): The Area Under the ROC Curve score.
            - accuracy (float): The accuracy score.
            - best_threshold (float): The best threshold for classification.

    This function calculates the AUROC score, accuracy, and best threshold.
    If verbose is True, it also prints precision, recall, and F1-score, and plots the ROC curve.
    """
    auroc = roc_auc_score(labels, losses)
    print(f"AUROC: {auroc}")
    metrics = get_metrics(labels, losses)
    
    print(f"Accuracy: {metrics[0]}")
    if verbose:
        print(f"Precision: {metrics[1]}")
        print(f"Recall: {metrics[2]}")
        print(f"F1-score: {metrics[3]}")
        plot_roc_curve(labels, losses, auroc)

    return auroc, metrics[0], metrics[4]