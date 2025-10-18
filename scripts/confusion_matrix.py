#!/usr/bin/env python3
"""
Confusion Matrix Plotter for b-hive analysis
Plots confusion matrices with various normalization options and styling
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from scipy.special import softmax
import mplhep as hep

plt.style.use(hep.cms.style.CMS)


def plot_confusion_matrix(truth, predictions, display_labels, normalize="true", 
                         output_path=None, figsize=(10, 8), cmap="Blues"):
    """
    Plot confusion matrix with various options
    
    Args:
        truth: Ground truth labels
        predictions: Predicted labels
        display_labels: List of class labels for display
        normalize: Normalization mode ("true", "pred", or None)
        output_path: Path to save the plot (optional)
        figsize: Figure size tuple
        cmap: Colormap for the heatmap
    """
    # Calculate confusion matrix
    cm = confusion_matrix(truth, predictions, normalize=normalize)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create confusion matrix display
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=display_labels
    )
    
    # Plot the confusion matrix
    cm_display.plot(
        ax=ax, 
        cmap=cmap, 
        values_format='.2f',
        colorbar=True
    )
    
    # Customize the plot
    ax.set_xlabel("Predicted Label", fontsize=20)
    ax.set_ylabel("True Label", fontsize=20)
    
    # Add CMS label
    #hep.cms.label(loc=0, data=True, llabel="Preliminary", rlabel="13.6 TeV", ax=ax)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Confusion matrix saved to: {output_path}")
    plt.close(fig)
    
    return fig, ax


def main():
    parser = argparse.ArgumentParser(description="Plot confusion matrix for b-hive analysis")
    
    # Required arguments
    parser.add_argument("--truth", "-t", type=str, required=True,
                       help="Path to truth.npy file")
    parser.add_argument("--prediction", "-p", type=str, required=True,
                       help="Path to prediction.npy file")
    
    # Optional arguments
    parser.add_argument("--output", "-o", type=str, default="cm.png",
                       help="Output path for the plot")
    parser.add_argument("--labels", "-l", type=str, nargs="+", 
                       default=["ttbb", "ttb", "ttcc", "ttc", "ttother"],
                       help="Class labels for display")
    parser.add_argument("--figsize", type=float, nargs=2, default=[10, 8],
                       help="Figure size (width, height)")
    parser.add_argument("--cmap", type=str, default="Blues",
                       help="Colormap for the heatmap")
    parser.add_argument("--apply-softmax", action="store_true",
                       help="Apply softmax to predictions (for logits)")  
    parser.add_argument("--normalize", "-n", type=str, 
                       choices=["true", "prediction", "both"], default="both",
                       help="Normalization mode for confusion matrix")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading truth from: {args.truth}")
    truth = np.load(args.truth)
    
    print(f"Loading predictions from: {args.prediction}")
    pred = np.load(args.prediction)
    
    # Apply softmax if requested
    if args.apply_softmax:
        print("Applying softmax to convert logits to probabilities")
        pred_softmax = softmax(pred, axis=1)
        predictions = np.argmax(pred_softmax, axis=1)
    else:
        predictions = np.argmax(pred, axis=1)
    
    # Print some statistics
    print(f"Number of samples: {len(truth)}")
    print(f"Number of classes: {len(args.labels)}")
    print(f"Truth distribution: {np.bincount(truth)}")
    print(f"Prediction distribution: {np.bincount(predictions)}")
    
    # Handle normalization argument
    normalize_config = {
        "true": ["true"],
        "prediction": ["prediction"],
        "both": ["true", "prediction"]
    }
    
    normalizations = normalize_config[args.normalize]
    
    for norm_type in normalizations:
        # Map to sklearn normalization parameter
        sklearn_norm = "true" if norm_type == "true" else "pred"
        
        # Set output path
        base, ext = os.path.splitext(args.output)
        output_path = f"{base}_{sklearn_norm}_norm{ext}"
        
        plot_confusion_matrix(
            truth, predictions, args.labels,
            normalize=sklearn_norm,
            output_path=output_path,
            figsize=args.figsize,
            cmap=args.cmap
        )


if __name__ == "__main__":
    main()
