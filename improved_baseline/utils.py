import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
    
def get_metrics(labels, losses):
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