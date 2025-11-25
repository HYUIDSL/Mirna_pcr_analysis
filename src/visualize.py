import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

def plot_confusion_matrix_heatmap(cm, class_labels, title, filename):
    """
    Generates and saves a heatmap for the given confusion matrix.

    Args:
        cm (numpy.ndarray): The confusion matrix.
        class_labels (list): The list of class names.
        title (str): The title for the plot.
        filename (str): The filename to save the plot as.
    """
    plt.figure(figsize=(8, 6))
    
    # Normalize the confusion matrix to show percentages of true labels
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    
    plt.title(title, fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Save the figure
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Successfully saved {filename}")
    except Exception as e:
        print(f"Error saving figure: {e}")
    
    plt.close() # Close the figure to free up memory

def plot_roc_curves(y_true, y_proba, class_labels, title, filename):
    """
    Generates and saves multiclass ROC curves.

    Args:
        y_true (np.ndarray): True labels.
        y_proba (np.ndarray): Predicted probabilities for each class.
        class_labels (list): The list of class names.
        title (str): The title for the plot.
        filename (str): The filename to save the plot as.
    """
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=range(len(class_labels)))
    n_classes = y_true_bin.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(class_labels[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=16)
    plt.legend(loc="lower right")
    
    # Save the figure
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Successfully saved {filename}")
    except Exception as e:
        print(f"Error saving figure: {e}")
        
    plt.close()
