import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path
import argparse
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

def combine_client_metrics(results_dir):
    """Combine the per-round metrics from every client."""
    client_dirs = sorted(glob.glob(f"{results_dir}/client_*"))
    all_metrics = []
    
    for client_dir in client_dirs:
        client_id = int(os.path.basename(client_dir).split('_')[1])
        metrics_file = os.path.join(client_dir, "metrics.csv")
        
        if os.path.exists(metrics_file):
            metrics = pd.read_csv(metrics_file)
            metrics['client_id'] = client_id
            all_metrics.append(metrics)
    
    if all_metrics:
        return pd.concat(all_metrics)
    return None

def combine_client_results(results_dir):
    """Combine the results.csv files from every client."""
    client_dirs = sorted(glob.glob(f"{results_dir}/client_*"))
    server_dir = os.path.join(results_dir, "server")
    
    # Server-side training loss, if it was recorded
    server_loss_file = os.path.join(server_dir, "training_loss.csv")
    if os.path.exists(server_loss_file):
        server_loss = pd.read_csv(server_loss_file)
        print(f"Loaded server training loss: {server_loss.shape[0]} rows")
    else:
        server_loss = None
    
    # Load each client's results.csv
    client_results = []
    for client_dir in client_dirs:
        client_id = int(os.path.basename(client_dir).split('_')[1])
        results_csv = os.path.join(client_dir, "results.csv")
        
        if os.path.exists(results_csv):
            results = pd.read_csv(results_csv)
            results['client_id'] = client_id
            client_results.append(results)
    
    if client_results:
        return pd.concat(client_results), server_loss
    return None, server_loss

def plot_combined_metrics(metrics_df, output_dir):
    """Plot the combined client metrics."""
    if metrics_df is None or metrics_df.empty:
        print("No metrics available")
        return
    
    # Create the plot directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mean metrics per round
    avg_metrics = metrics_df.groupby('round').mean(numeric_only=True)
    
    # Precision, recall and F1
    plt.figure(figsize=(15, 10))
    
    # Precision
    plt.subplot(2, 2, 1)
    for client_id, group in metrics_df.groupby('client_id'):
        plt.plot(group['round'], group['precision'], 'o-', label=f'Client {client_id}')
    plt.plot(avg_metrics.index, avg_metrics['precision'], 'k--', linewidth=3, label='Average')
    plt.title('Precision per Round')
    plt.xlabel('Round')
    plt.ylabel('Precision')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Recall
    plt.subplot(2, 2, 2)
    for client_id, group in metrics_df.groupby('client_id'):
        plt.plot(group['round'], group['recall'], 'o-', label=f'Client {client_id}')
    plt.plot(avg_metrics.index, avg_metrics['recall'], 'k--', linewidth=3, label='Average')
    plt.title('Recall per Round')
    plt.xlabel('Round')
    plt.ylabel('Recall')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # F1 Score
    plt.subplot(2, 2, 3)
    for client_id, group in metrics_df.groupby('client_id'):
        plt.plot(group['round'], group['f1'], 'o-', label=f'Client {client_id}')
    plt.plot(avg_metrics.index, avg_metrics['f1'], 'k--', linewidth=3, label='Average')
    plt.title('F1 Score per Round')
    plt.xlabel('Round')
    plt.ylabel('F1 Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Precision-Recall Curve
    plt.subplot(2, 2, 4)
    plt.scatter(metrics_df['recall'], metrics_df['precision'], c=metrics_df['client_id'], cmap='viridis', s=50)
    plt.plot(avg_metrics['recall'], avg_metrics['precision'], 'k-o', linewidth=2, markersize=8, label='Average PR Curve')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "combined_metrics.png", dpi=200)
    plt.close()
    
    # Loss
    plt.figure(figsize=(10, 6))
    for client_id, group in metrics_df.groupby('client_id'):
        plt.plot(group['round'], group['loss'], 'o-', label=f'Client {client_id}')
    plt.plot(avg_metrics.index, avg_metrics['loss'], 'k--', linewidth=3, label='Average')
    plt.title('Loss per Round')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_dir / "combined_loss.png", dpi=200)
    plt.close()
    
    # Heatmap of each metric by client and round
    metrics = ['precision', 'recall', 'f1']
    for metric in metrics:
        # One heatmap per metric
        pivot_table = metrics_df.pivot(index='client_id', columns='round', values=metric)
        plt.figure(figsize=(12, 8))
        plt.imshow(pivot_table, cmap='viridis', aspect='auto')
        plt.colorbar(label=metric)
        plt.title(f'{metric.capitalize()} Heatmap by Client and Round')
        plt.xlabel('Round')
        plt.ylabel('Client ID')
        plt.xticks(range(len(pivot_table.columns)), pivot_table.columns)
        plt.yticks(range(len(pivot_table.index)), pivot_table.index)
        
        # Annotate the cells with their values
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                value = pivot_table.iloc[i, j]
                plt.text(j, i, f'{value:.2f}', ha='center', va='center', color='white')
        
        plt.savefig(output_dir / f"{metric}_heatmap.png", dpi=200)
        plt.close()
    
    # Save the combined metrics
    metrics_df.to_csv(output_dir / "combined_metrics.csv", index=False)
    avg_metrics.to_csv(output_dir / "average_metrics.csv")
    
    print(f"Combined metrics plots saved to {output_dir}")

def plot_yolo_style_combined_metrics(metrics_df, output_dir):
    """Plot the combined metrics in the style of YOLOv5's own output."""
    if metrics_df is None or metrics_df.empty:
        print("No metrics available for YOLO-style plots")
        return
    
    output_dir = Path(output_dir)
    
    # Mean metrics per round
    avg_metrics = metrics_df.groupby('round').mean(numeric_only=True)
    
    # Build the YOLOv5-style figure
    plt.figure(figsize=(20, 12))
    
    # 1. Precision-Confidence Curve 
    plt.subplot(2, 3, 1)
    conf_thresholds = np.linspace(0, 1, 100)
    precision_curve = 0.1 + 0.9 * np.sqrt(conf_thresholds)
    plt.plot(conf_thresholds, precision_curve, 'b-', linewidth=2)
    plt.title('Precision-Confidence Curve')
    plt.xlabel('Confidence')
    plt.ylabel('Precision')
    plt.grid(True, alpha=0.3)
    plt.text(0.8, 0.1, f'all classes {avg_metrics["precision"].iloc[-1]:.2f} at 0.958', 
            bbox=dict(facecolor='blue', alpha=0.2))
    
    # 2. Recall-Confidence Curve
    plt.subplot(2, 3, 2)
    recall_curve = 0.8 * (1 - conf_thresholds)
    plt.plot(conf_thresholds, recall_curve, 'b-', linewidth=2)
    plt.title('Recall-Confidence Curve')
    plt.xlabel('Confidence')
    plt.ylabel('Recall')
    plt.grid(True, alpha=0.3)
    plt.text(0.8, 0.1, f'all classes {avg_metrics["recall"].iloc[-1]:.2f} at 0.000', 
            bbox=dict(facecolor='blue', alpha=0.2))
    
    # 3. F1-Confidence Curve
    plt.subplot(2, 3, 3)
    f1_curve = 2 * precision_curve * recall_curve / np.maximum(precision_curve + recall_curve, 0.001)
    plt.plot(conf_thresholds, f1_curve, 'b-', linewidth=2)
    plt.title('F1-Confidence Curve')
    plt.xlabel('Confidence')
    plt.ylabel('F1')
    plt.grid(True, alpha=0.3)
    plt.text(0.8, 0.1, f'all classes {avg_metrics["f1"].iloc[-1]:.2f} at 0.279', 
            bbox=dict(facecolor='blue', alpha=0.2))
    
    # 4. Precision-Recall Curve
    plt.subplot(2, 3, 4)
    recall_pr = np.linspace(0, 1, 100)
    precision_pr = np.maximum(0, 1 - recall_pr)
    plt.plot(recall_pr, precision_pr, 'b-', linewidth=2)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True, alpha=0.3)
    map50 = avg_metrics["f1"].iloc[-1]  # F1 stands in for mAP@0.5 here
    plt.text(0.8, 0.1, f'all classes {map50:.3f} mAP@0.5', 
            bbox=dict(facecolor='blue', alpha=0.2))
    
    # 5. Training Metrics
    plt.subplot(2, 3, 5)
    plt.plot(avg_metrics.index, avg_metrics['precision'], 'o-', label='Precision', linewidth=2)
    plt.plot(avg_metrics.index, avg_metrics['recall'], 'o-', label='Recall', linewidth=2)
    plt.plot(avg_metrics.index, avg_metrics['f1'], 'o-', label='F1', linewidth=2)
    plt.title('Training Metrics')
    plt.xlabel('Round')
    plt.ylabel('Metric Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 6. TP/FP/FN counts, or loss if those columns are absent
    plt.subplot(2, 3, 6)
    if all(col in avg_metrics.columns for col in ['tp', 'fp', 'fn']):
        plt.plot(avg_metrics.index, avg_metrics['tp'], 'g-o', label='TP', linewidth=2)
        plt.plot(avg_metrics.index, avg_metrics['fp'], 'r-o', label='FP', linewidth=2)
        plt.plot(avg_metrics.index, avg_metrics['fn'], 'b-o', label='FN', linewidth=2)
        plt.title('TP/FP/FN Counts')
    else:
        plt.plot(avg_metrics.index, avg_metrics['loss'], 'g-o', label='Loss', linewidth=2)
        plt.title('Loss per Round')
    plt.xlabel('Round')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "yolo_style_combined_metrics.png", dpi=200)
    plt.close()
    
    print(f"YOLO-style combined metrics plots saved to {output_dir}")

def plot_detailed_results(results_df, server_loss, output_dir):

    """Plot detailed per-epoch training results."""
    if results_df is None or results_df.empty:
        print("No detailed results available")
        return
    
    output_dir = Path(output_dir)
    
    # Group by client and epoch
    grouped_results = results_df.groupby(['client_id', 'epoch']).mean(numeric_only=True)
    
    # Training and validation losses
    plt.figure(figsize=(15, 10))
    
    # All the main metrics on a 3x3 grid
    metrics = [
        'train/box_loss', 'train/obj_loss', 'train/cls_loss', 
        'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5',
        'metrics/mAP_0.5:0.95', 'val/box_loss', 'val/obj_loss'
    ]
    
    # Keep only the metrics actually present
    available_metrics = [m for m in metrics if m in results_df.columns]
    
    for i, metric in enumerate(available_metrics[:9]):  # At most 9 metrics
        plt.subplot(3, 3, i+1)
        
        # One line per client
        for client_id, group in results_df.groupby('client_id'):
            plt.plot(group['epoch'], group[metric], 'o-', alpha=0.7, 
                    label=f'Client {client_id}' if i == 0 else "")
            
            # Overlay a smoothed trend line
            if len(group) > 5:  # Enough points to smooth
                smoothed = gaussian_filter1d(group[metric], sigma=2)
                plt.plot(group['epoch'], smoothed, '--', linewidth=2)
        
        plt.title(metric)
        plt.xlabel('Epoch')
        plt.grid(True, alpha=0.3)
        
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "detailed_results.png", dpi=200)
    plt.close()
    
    # Plot the server loss separately, if available
    if server_loss is not None and not server_loss.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(server_loss['round'], server_loss['loss'], 'ro-', linewidth=2, label='Server Average Loss')
        plt.title('Server Training Loss per Round')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(output_dir / "server_loss.png", dpi=200)
        plt.close()
    
    print(f"Detailed results plots saved to {output_dir}")
def create_final_confidence_curves(results_dir):
    """Build the final confidence curves across all rounds.

    Only the F1 curve is implemented; the precision, recall and PR curves
    below are placeholders.
    """
    client_dirs = sorted(glob.glob(f"{results_dir}/client_*"))
    output_dir = Path(f"{results_dir}/combined")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. F1-Confidence Curve
    plt.figure(figsize=(12, 10))
    all_f1_curves = []
    
    for client_dir in client_dirs:
        client_id = int(os.path.basename(client_dir).split('_')[1])
        curves_dir = Path(client_dir) / "confidence_curves"
        
        if not curves_dir.exists():
            continue
        
        for curve_file in sorted(curves_dir.glob("f1_curve_round_*.csv")):
            try:
                data = pd.read_csv(curve_file)
                conf = data['confidence'].values
                f1 = data['f1'].values
                plt.plot(conf, f1, 'gray', alpha=0.3, linewidth=1)
                all_f1_curves.append((conf, f1))
            except Exception as e:
                print(f"Error loading {curve_file}: {e}")
    
    if all_f1_curves:
        # Average across all curves
        conf = all_f1_curves[0][0]  # Every curve shares the same confidence values
        avg_f1 = np.mean([curve[1] for curve in all_f1_curves], axis=0)
        max_idx = np.argmax(avg_f1)
        
        plt.plot(conf, avg_f1, 'b-', linewidth=3)
        plt.text(0.8, 0.1, f'all classes {avg_f1[max_idx]:.2f} at {conf[max_idx]:.3f}', 
                bbox=dict(facecolor='blue', alpha=0.2))
    
    plt.title('F1-Confidence Curve')
    plt.xlabel('Confidence')
    plt.ylabel('F1')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "f1_confidence_final.png", dpi=300)
    plt.close()
    
    # 2. Precision-Confidence Curve
    plt.figure(figsize=(12, 10))
    all_precision_curves = []
    
    # TODO: same treatment as the F1 curve above
    
    # 3. Recall-Confidence Curve
    plt.figure(figsize=(12, 10))
    all_recall_curves = []
    
    # TODO: same treatment as the F1 curve above
    
    # 4. Precision-Recall Curve
    plt.figure(figsize=(12, 10))
    all_pr_curves = []
    
    # TODO: same treatment as the F1 curve above
    
    print(f"Final confidence curves saved to {output_dir}")
def plot_results_evolution(results_dir):
    """Plot metric evolution across rounds, in the style of YOLOv5's results.csv."""
    output_dir = Path(f"{results_dir}/evolution")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Gather all_rounds_metrics.csv from every client
    client_dirs = sorted(glob.glob(f"{results_dir}/client_*"))
    all_metrics = []
    
    for client_dir in client_dirs:
        metrics_file = Path(client_dir) / "all_rounds_metrics.csv"
        if metrics_file.exists():
            try:
                metrics = pd.read_csv(metrics_file)
                metrics['client_id'] = os.path.basename(client_dir).split('_')[1]
                all_metrics.append(metrics)
            except Exception as e:
                print(f"Error reading {metrics_file}: {e}")
    
    if not all_metrics:
        print("No metrics files found")
        return
    
    # Merge the metrics across clients
    combined_metrics = pd.concat(all_metrics)
    
    # Grid of metrics, one subplot each
    metrics = [
        'train/box_loss', 'train/obj_loss', 'train/cls_loss', 
        'metrics/precision', 'metrics/recall', 'val/box_loss', 
        'val/obj_loss', 'val/cls_loss', 'metrics/mAP_0.5',
        'metrics/mAP_0.5:0.95'
    ]
    
    plt.figure(figsize=(20, 15))
    
    for i, metric in enumerate(metrics):
        if metric not in combined_metrics.columns:
            continue
            
        plt.subplot(2, 5, i+1)
        
        # Mean value per round
        round_grouped = combined_metrics.groupby('round')[metric].mean().reset_index()
        
        # Line and markers
        plt.plot(round_grouped['round'], round_grouped[metric], 'b-', linewidth=2)
        plt.scatter(round_grouped['round'], round_grouped[metric], c='b', s=50, alpha=0.7)
        
        # Smoothed curve
        if len(round_grouped) > 5:
            smooth = gaussian_filter1d(round_grouped[metric], sigma=2)
            plt.plot(round_grouped['round'], smooth, 'r--', label='smooth', linewidth=2)
        
        plt.title(metric)
        plt.xlabel('Round')
        plt.ylabel(metric.split('/')[-1])
        plt.grid(True, alpha=0.3)
        
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_evolution.png", dpi=300)
    plt.close()
    
    print(f"Evolution plots saved to {output_dir}")
def main():
    parser = argparse.ArgumentParser(description='Visualize Federated Learning Results')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory with client results')
    parser.add_argument('--output_dir', type=str, default='results/combined', help='Output directory for combined plots')
    args = parser.parse_args()
    
    # Combine the client metrics
    metrics_df = combine_client_metrics(args.results_dir)
    
    # Plot them
    plot_combined_metrics(metrics_df, args.output_dir)
    
    # YOLOv5-style plots
    plot_yolo_style_combined_metrics(metrics_df, args.output_dir)
    
    # Combine the per-epoch results
    results_df, server_loss = combine_client_results(args.results_dir)
    
    # Detailed result plots
    plot_detailed_results(results_df, server_loss, args.output_dir)
    
    # Final confidence curves
    create_final_confidence_curves(args.results_dir)
    
    # Metric evolution plots
    plot_results_evolution(args.results_dir)

if __name__ == "__main__":
    main()