#!/usr/bin/env python3
"""
Create a summary table of model performance across all seeds.
Shows accuracy and AUC metrics averaged across 3 seeds with standard deviations.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any


def load_aggregated_results(filepath: str = "./training_results/aggregated_results.json") -> Dict[str, Any]:
    """Load aggregated results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_summary_table(results: Dict[str, Any]) -> pd.DataFrame:
    """Create a summary table with accuracy, AUC, and timing metrics."""
    data = []
    
    for model_name, metrics in results.items():
        row = {
            "Model": model_name,
            "Accuracy (Mean)": f"{metrics['accuracy']['mean']:.4f}",
            "Accuracy (STD)": f"{metrics['accuracy']['std']:.4f}",
            "AUC (Mean)": f"{metrics['auc_roc']['mean']:.4f}",
            "AUC (STD)": f"{metrics['auc_roc']['std']:.4f}",
            "Loss (Mean)": f"{metrics.get('final_val_loss', {}).get('mean', 0):.4f}",
            "Loss (STD)": f"{metrics.get('final_val_loss', {}).get('std', 0):.4f}",
            "Train Time (s)": f"{metrics['training_time']['mean']:.2f}±{metrics['training_time']['std']:.2f}",
            "Inference Time (s)": f"{metrics['inference_time']['mean']:.4f}±{metrics['inference_time']['std']:.4f}",
        }
        data.append(row)
    
    # Sort by accuracy (descending)
    df = pd.DataFrame(data)
    # Extract numeric values for sorting
    df['sort_key'] = df["Accuracy (Mean)"].astype(float)
    df = df.sort_values('sort_key', ascending=False).drop('sort_key', axis=1)
    df = df.reset_index(drop=True)
    
    return df


# def get_hyperparameters() -> Dict[str, str]:
#     """Extract hyperparameters from training code."""
#     hyperparams = {
#         "Number of Seeds": "3",
#         "Random Seeds": "",
#         "Epochs": "50",
#         "Batch Size": "32",
#         "Learning Rate": "0.001",
#         "Optimizer": "Adam",
#         "Optimizer Betas": "(0.9, 0.999)",
#         "Loss Function": "CrossEntropyLoss",
#         "Learning Rate Scheduler": "ReduceLROnPlateau",
#         "  - Factor": "0.5",
#         "  - Patience": "5",
#         "  - Mode": "min (on validation loss)",
#         "Early Stopping Patience": "5 epochs",
#         "Image Size": "224x224",
#         "Train/Val/Test Split": "70% / 20% / 10%",
#         "Input Channels": "1 (grayscale)",
#         "Number of Classes": "2 (binary classification)",
#         "Data Augmentation (Train)": "Resize → Grayscale → RandomHorizontalFlip(p=0.5) → RandomRotation(±15°) → Normalize",
#         "Data Augmentation (Val/Test)": "Resize → Grayscale → Normalize",
#         "Normalization": "Mean: 0.5, Std: 0.5",
#     }
#       return hyperparams


def save_summary(df: pd.DataFrame, 
                 output_path: str = "./training_results/model_performance_summary.txt"):
    """Save summary table and hyperparameters to file."""
    with open(output_path, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("MODEL PERFORMANCE SUMMARY (Averaged over 3 Seeds)\n")
        f.write("=" * 120 + "\n\n")
        
        # Write table
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        f.write("=" * 120 + "\n")
        f.write("HYPERPARAMETERS USED\n")
        f.write("=" * 120 + "\n\n")
        
        # # Write hyperparameters
        # for key, value in hyperparams.items():
        #     f.write(f"{key:.<45} {value}\n")
        
        f.write("\n" + "=" * 120 + "\n")
        f.write("NOTES:\n")
        f.write("=" * 120 + "\n")
        f.write("- All metrics are averaged across 3 random seeds\n")
        f.write("- Standard deviations show variance across the 3 runs\n")
        f.write("- Loss: Final validation loss at best model checkpoint\n")
        f.write("- Training Time: Average time to train one model (seconds)\n")
        f.write("- Inference Time: Average time to run inference on entire test set (seconds)\n")
        f.write("- Models are sorted by accuracy (highest to lowest)\n")
        f.write("- All metrics evaluated on the test set (10% of data)\n")
        f.write("- Early stopping patience is for validation loss improvement (5 epochs without improvement stops training)\n")
    
    print(f"Summary saved to: {output_path}")


def main():
    """Main execution."""
    try:
        # Load results
        print("Loading aggregated results...")
        results = load_aggregated_results()
        
        # Create summary table
        print("Creating summary table...")
        summary_df = create_summary_table(results)
        
        # Get hyperparameters
       # hyperparams = get_hyperparameters()
        
        # Display in console
        print("\n" + "=" * 120)
        print("MODEL PERFORMANCE SUMMARY (Averaged over 3 Seeds)")
        print("=" * 120)
        print(summary_df.to_string(index=False))
        
        print("\n" + "=" * 120)
        print("HYPERPARAMETERS USED")
        print("=" * 120)
        # for key, value in hyperparams.items():
        #     print(f"{key:.<45} {value}")
        
        # Save to file
        print("\n" + "=" * 120)
        save_summary(summary_df)
        
        # Also save as CSV for easy import
        csv_path = "./training_results/model_performance_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"CSV summary saved to: {csv_path}")
        
        print("\n✓ Summary generation complete!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
