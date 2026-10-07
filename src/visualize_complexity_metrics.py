"""
Visualization script for Complexity and Performance Metrics

This script creates comprehensive plots for:
- Inference time distribution
- Memory usage timeline
- Communication costs
- Device comparison (GPU vs CPU)
- Efficiency analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_metrics(results_dir):
    """Load all metrics from CSV files."""
    results_path = Path(results_dir)
    
    metrics = {}
    
    # Find all client directories
    client_dirs = list(results_path.glob("client_*/metrics"))
    
    for client_dir in client_dirs:
        client_id = client_dir.parent.name
        
        # Load CSV files
        csv_files = {
            'inference': client_dir / 'inference_metrics.csv',
            'training': client_dir / 'training_metrics.csv',
            'memory': client_dir / 'memory_metrics.csv',
            'communication': client_dir / 'communication_metrics.csv',
        }
        
        # Load JSON summary
        json_file = client_dir / 'metrics_summary.json'
        
        client_metrics = {'client_id': client_id}
        
        for key, filepath in csv_files.items():
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    client_metrics[key] = df
                except Exception as e:
                    print(f"Warning: Could not load {filepath}: {e}")
        
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    client_metrics['summary'] = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")
        
        metrics[client_id] = client_metrics
    
    return metrics


def plot_inference_time_distribution(metrics, output_dir):
    """Plot inference time distribution across clients."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Inference Time Analysis', fontsize=16, fontweight='bold')
    
    all_clients_data = []
    
    for client_id, client_metrics in metrics.items():
        if 'inference' not in client_metrics:
            continue
        
        df = client_metrics['inference']
        
        if 'inference_time_mean_ms' in df.columns:
            all_clients_data.append({
                'client': client_id,
                'times': df['inference_time_mean_ms'].values,
                'device': df['device'].iloc[0] if 'device' in df.columns else 'unknown'
            })
    
    if not all_clients_data:
        print("No inference data available for plotting")
        return
    
    # Plot 1: Boxplot of inference times per client
    ax = axes[0, 0]
    data_for_box = [d['times'] for d in all_clients_data]
    labels = [d['client'] for d in all_clients_data]
    ax.boxplot(data_for_box, labels=labels)
    ax.set_title('Inference Time Distribution per Client')
    ax.set_ylabel('Time (ms)')
    ax.set_xlabel('Client')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Time evolution across rounds
    ax = axes[0, 1]
    for data in all_clients_data:
        client_id = data['client']
        df = metrics[client_id]['inference']
        if 'round' in df.columns and 'inference_time_mean_ms' in df.columns:
            ax.plot(df['round'], df['inference_time_mean_ms'], marker='o', label=client_id)
    ax.set_title('Inference Time Evolution')
    ax.set_xlabel('Round')
    ax.set_ylabel('Mean Inference Time (ms)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Throughput (FPS)
    ax = axes[1, 0]
    for data in all_clients_data:
        client_id = data['client']
        df = metrics[client_id]['inference']
        if 'round' in df.columns and 'throughput_fps' in df.columns:
            ax.plot(df['round'], df['throughput_fps'], marker='s', label=client_id)
    ax.set_title('Throughput (FPS) Evolution')
    ax.set_xlabel('Round')
    ax.set_ylabel('FPS')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Latency percentiles comparison
    ax = axes[1, 1]
    clients = [d['client'] for d in all_clients_data]
    
    p50_values = []
    p95_values = []
    p99_values = []
    
    for data in all_clients_data:
        client_id = data['client']
        df = metrics[client_id]['inference']
        
        if all(col in df.columns for col in ['inference_time_p50_ms', 'inference_time_p95_ms', 'inference_time_p99_ms']):
            p50_values.append(df['inference_time_p50_ms'].mean())
            p95_values.append(df['inference_time_p95_ms'].mean())
            p99_values.append(df['inference_time_p99_ms'].mean())
        else:
            p50_values.append(0)
            p95_values.append(0)
            p99_values.append(0)
    
    x = np.arange(len(clients))
    width = 0.25
    
    ax.bar(x - width, p50_values, width, label='P50', alpha=0.8)
    ax.bar(x, p95_values, width, label='P95', alpha=0.8)
    ax.bar(x + width, p99_values, width, label='P99', alpha=0.8)
    
    ax.set_xlabel('Client')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency Percentiles Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(clients)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'inference_time_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_memory_usage(metrics, output_dir):
    """Plot memory usage analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Memory Usage Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: GPU Memory Usage Timeline
    ax = axes[0, 0]
    for client_id, client_metrics in metrics.items():
        if 'memory' not in client_metrics:
            continue
        
        df = client_metrics['memory']
        if 'round' in df.columns and 'gpu_memory_allocated_mb' in df.columns:
            ax.plot(df['round'], df['gpu_memory_allocated_mb'], marker='o', label=client_id)
    
    ax.set_title('GPU Memory Usage Over Time')
    ax.set_xlabel('Round')
    ax.set_ylabel('Memory (MB)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Peak Memory Comparison
    ax = axes[0, 1]
    clients = []
    gpu_peak = []
    cpu_peak = []
    
    for client_id, client_metrics in metrics.items():
        if 'inference' not in client_metrics:
            continue
        
        df = client_metrics['inference']
        clients.append(client_id)
        
        if 'gpu_memory_peak_mb' in df.columns:
            gpu_peak.append(df['gpu_memory_peak_mb'].max())
        else:
            gpu_peak.append(0)
        
        if 'cpu_memory_rss_mb' in df.columns:
            cpu_peak.append(df['cpu_memory_rss_mb'].max())
        else:
            cpu_peak.append(0)
    
    x = np.arange(len(clients))
    width = 0.35
    
    ax.bar(x - width/2, gpu_peak, width, label='GPU Peak', alpha=0.8)
    ax.bar(x + width/2, cpu_peak, width, label='CPU Peak', alpha=0.8)
    
    ax.set_xlabel('Client')
    ax.set_ylabel('Peak Memory (MB)')
    ax.set_title('Peak Memory Usage Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(clients)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Memory efficiency (Memory per image)
    ax = axes[1, 0]
    for client_id, client_metrics in metrics.items():
        if 'inference' not in client_metrics:
            continue
        
        df = client_metrics['inference']
        if 'round' in df.columns and 'gpu_memory_allocated_mb' in df.columns and 'batch_size_avg' in df.columns:
            memory_per_image = df['gpu_memory_allocated_mb'] / df['batch_size_avg']
            ax.plot(df['round'], memory_per_image, marker='o', label=client_id)
    
    ax.set_title('Memory Efficiency (Memory per Image)')
    ax.set_xlabel('Round')
    ax.set_ylabel('MB per Image')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: System Memory Utilization
    ax = axes[1, 1]
    for client_id, client_metrics in metrics.items():
        if 'memory' not in client_metrics:
            continue
        
        df = client_metrics['memory']
        if 'round' in df.columns and 'system_memory_percent' in df.columns:
            ax.plot(df['round'], df['system_memory_percent'], marker='s', label=client_id)
    
    ax.set_title('System Memory Utilization %')
    ax.set_xlabel('Round')
    ax.set_ylabel('Utilization %')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'memory_usage_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_communication_costs(metrics, output_dir):
    """Plot communication costs analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Communication Costs Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Communication per round
    ax = axes[0, 0]
    for client_id, client_metrics in metrics.items():
        if 'communication' not in client_metrics:
            continue
        
        df = client_metrics['communication']
        if 'round' in df.columns and 'parameters_size_mb' in df.columns:
            # Group by round and sum
            comm_per_round = df.groupby('round')['parameters_size_mb'].sum()
            ax.plot(comm_per_round.index, comm_per_round.values, marker='o', label=client_id)
    
    ax.set_title('Total Communication per Round')
    ax.set_xlabel('Round')
    ax.set_ylabel('Data Transferred (MB)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Upload vs Download
    ax = axes[0, 1]
    for client_id, client_metrics in metrics.items():
        if 'communication' not in client_metrics:
            continue
        
        df = client_metrics['communication']
        if 'direction' in df.columns and 'parameters_size_mb' in df.columns:
            upload = df[df['direction'] == 'upload']['parameters_size_mb'].sum()
            download = df[df['direction'] == 'download']['parameters_size_mb'].sum()
            
            ax.bar(client_id, upload, label='Upload' if client_id == list(metrics.keys())[0] else '', alpha=0.8, color='blue')
            ax.bar(client_id, download, bottom=upload, label='Download' if client_id == list(metrics.keys())[0] else '', alpha=0.8, color='orange')
    
    ax.set_title('Total Upload vs Download')
    ax.set_ylabel('Data (MB)')
    ax.set_xlabel('Client')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Cumulative communication
    ax = axes[1, 0]
    for client_id, client_metrics in metrics.items():
        if 'communication' not in client_metrics:
            continue
        
        df = client_metrics['communication']
        if 'round' in df.columns and 'parameters_size_mb' in df.columns:
            cumsum = df.groupby('round')['parameters_size_mb'].sum().cumsum()
            ax.plot(cumsum.index, cumsum.values, marker='o', label=client_id)
    
    ax.set_title('Cumulative Communication Cost')
    ax.set_xlabel('Round')
    ax.set_ylabel('Total Data (MB)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Communication summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    table_data = []
    for client_id, client_metrics in metrics.items():
        if 'summary' not in client_metrics or 'communication_summary' not in client_metrics['summary']:
            continue
        
        comm_summary = client_metrics['summary']['communication_summary']
        table_data.append([
            client_id,
            f"{comm_summary.get('total_communication_mb', 0):.2f}",
            f"{comm_summary.get('total_uploaded_mb', 0):.2f}",
            f"{comm_summary.get('total_downloaded_mb', 0):.2f}"
        ])
    
    if table_data:
        table = ax.table(cellText=table_data,
                        colLabels=['Client', 'Total (MB)', 'Upload (MB)', 'Download (MB)'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax.set_title('Communication Summary', pad=20)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'communication_costs_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_device_comparison(metrics, output_dir):
    """Plot GPU vs CPU comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Device Comparison (GPU vs CPU)', fontsize=16, fontweight='bold')
    
    # Separate clients by device
    gpu_clients = {}
    cpu_clients = {}
    
    for client_id, client_metrics in metrics.items():
        if 'inference' not in client_metrics:
            continue
        
        df = client_metrics['inference']
        if 'device' in df.columns:
            device = df['device'].iloc[0] if len(df) > 0 else 'unknown'
            if 'cuda' in str(device).lower():
                gpu_clients[client_id] = client_metrics
            else:
                cpu_clients[client_id] = client_metrics
    
    # Plot 1: Inference time comparison
    ax = axes[0, 0]
    
    gpu_times = []
    cpu_times = []
    
    for client_id, client_metrics in gpu_clients.items():
        df = client_metrics['inference']
        if 'inference_time_mean_ms' in df.columns:
            gpu_times.extend(df['inference_time_mean_ms'].values)
    
    for client_id, client_metrics in cpu_clients.items():
        df = client_metrics['inference']
        if 'inference_time_mean_ms' in df.columns:
            cpu_times.extend(df['inference_time_mean_ms'].values)
    
    if gpu_times or cpu_times:
        data_to_plot = []
        labels = []
        if gpu_times:
            data_to_plot.append(gpu_times)
            labels.append('GPU')
        if cpu_times:
            data_to_plot.append(cpu_times)
            labels.append('CPU')
        
        ax.boxplot(data_to_plot, labels=labels)
        ax.set_title('Inference Time: GPU vs CPU')
        ax.set_ylabel('Time (ms)')
        ax.grid(True, alpha=0.3)
        
        # Add speedup annotation
        if gpu_times and cpu_times:
            speedup = np.mean(cpu_times) / np.mean(gpu_times)
            ax.text(0.5, 0.95, f'GPU Speedup: {speedup:.1f}x', 
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Throughput comparison
    ax = axes[0, 1]
    
    if gpu_clients:
        avg_gpu_fps = []
        for client_id, client_metrics in gpu_clients.items():
            df = client_metrics['inference']
            if 'throughput_fps' in df.columns:
                avg_gpu_fps.append(df['throughput_fps'].mean())
        ax.bar('GPU', np.mean(avg_gpu_fps) if avg_gpu_fps else 0, alpha=0.8, color='green')
    
    if cpu_clients:
        avg_cpu_fps = []
        for client_id, client_metrics in cpu_clients.items():
            df = client_metrics['inference']
            if 'throughput_fps' in df.columns:
                avg_cpu_fps.append(df['throughput_fps'].mean())
        ax.bar('CPU', np.mean(avg_cpu_fps) if avg_cpu_fps else 0, alpha=0.8, color='blue')
    
    ax.set_title('Average Throughput: GPU vs CPU')
    ax.set_ylabel('FPS')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Memory usage comparison
    ax = axes[1, 0]
    
    gpu_memory = []
    cpu_memory = []
    
    for client_id, client_metrics in gpu_clients.items():
        df = client_metrics['inference']
        if 'gpu_memory_peak_mb' in df.columns:
            gpu_memory.append(df['gpu_memory_peak_mb'].max())
    
    for client_id, client_metrics in cpu_clients.items():
        df = client_metrics['inference']
        if 'cpu_memory_rss_mb' in df.columns:
            cpu_memory.append(df['cpu_memory_rss_mb'].max())
    
    devices = []
    memory_values = []
    
    if gpu_memory:
        devices.append('GPU')
        memory_values.append(np.mean(gpu_memory))
    if cpu_memory:
        devices.append('CPU')
        memory_values.append(np.mean(cpu_memory))
    
    if devices:
        ax.bar(devices, memory_values, alpha=0.8, color=['green', 'blue'][:len(devices)])
        ax.set_title('Peak Memory Usage: GPU vs CPU')
        ax.set_ylabel('Memory (MB)')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    table_data = []
    
    if gpu_times:
        table_data.append([
            'GPU',
            f"{np.mean(gpu_times):.2f}",
            f"{np.mean([df['throughput_fps'].mean() for _, cm in gpu_clients.items() for df in [cm['inference']] if 'throughput_fps' in df.columns]):.1f}" if gpu_clients else "N/A",
            f"{len(gpu_clients)}"
        ])
    
    if cpu_times:
        table_data.append([
            'CPU',
            f"{np.mean(cpu_times):.2f}",
            f"{np.mean([df['throughput_fps'].mean() for _, cm in cpu_clients.items() for df in [cm['inference']] if 'throughput_fps' in df.columns]):.1f}" if cpu_clients else "N/A",
            f"{len(cpu_clients)}"
        ])
    
    if table_data:
        table = ax.table(cellText=table_data,
                        colLabels=['Device', 'Avg Time (ms)', 'Avg FPS', '# Clients'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax.set_title('Device Performance Summary', pad=20)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'device_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_efficiency_analysis(metrics, output_dir):
    """Plot efficiency metrics analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Efficiency Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Time per GFLOP
    ax = axes[0, 0]
    for client_id, client_metrics in metrics.items():
        if 'inference' not in client_metrics:
            continue
        
        df = client_metrics['inference']
        if 'round' in df.columns and 'time_per_gflop_ms' in df.columns:
            ax.plot(df['round'], df['time_per_gflop_ms'], marker='o', label=client_id)
    
    ax.set_title('Computational Efficiency (Time per GFLOP)')
    ax.set_xlabel('Round')
    ax.set_ylabel('ms / GFLOP')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: FPS per GFLOP
    ax = axes[0, 1]
    for client_id, client_metrics in metrics.items():
        if 'inference' not in client_metrics:
            continue
        
        df = client_metrics['inference']
        if 'round' in df.columns and 'fps_per_gflop' in df.columns:
            ax.plot(df['round'], df['fps_per_gflop'], marker='s', label=client_id)
    
    ax.set_title('Throughput Efficiency (FPS per GFLOP)')
    ax.set_xlabel('Round')
    ax.set_ylabel('FPS / GFLOP')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Model complexity summary
    ax = axes[1, 0]
    ax.axis('off')
    
    table_data = []
    for client_id, client_metrics in metrics.items():
        if 'summary' not in client_metrics or 'complexity' not in client_metrics['summary']:
            continue
        
        complexity = client_metrics['summary']['complexity']
        table_data.append([
            client_id,
            f"{complexity.get('gflops', 0):.2f}",
            f"{complexity.get('parameters_millions', 0):.2f}",
            f"{complexity.get('model_size_mb', 0):.2f}"
        ])
    
    if table_data:
        table = ax.table(cellText=table_data,
                        colLabels=['Client', 'GFLOPs', 'Params (M)', 'Size (MB)'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax.set_title('Model Complexity', pad=20)
    
    # Plot 4: Overall efficiency score
    ax = axes[1, 1]
    
    clients = []
    efficiency_scores = []
    
    for client_id, client_metrics in metrics.items():
        if 'inference' not in client_metrics:
            continue
        
        df = client_metrics['inference']
        if 'fps_per_gflop' in df.columns:
            # Efficiency score = average FPS per GFLOP
            score = df['fps_per_gflop'].mean()
            clients.append(client_id)
            efficiency_scores.append(score)
    
    if clients:
        colors = ['green' if score > np.median(efficiency_scores) else 'orange' for score in efficiency_scores]
        ax.barh(clients, efficiency_scores, color=colors, alpha=0.8)
        ax.set_xlabel('Efficiency Score (FPS/GFLOP)')
        ax.set_title('Overall Efficiency Ranking')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'efficiency_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_summary_report(metrics, output_dir):
    """Create a text summary report of all metrics."""
    output_path = Path(output_dir) / 'metrics_summary_report.txt'
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FEDERATED LEARNING PERFORMANCE METRICS SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        for client_id, client_metrics in metrics.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"CLIENT: {client_id}\n")
            f.write(f"{'='*80}\n\n")
            
            # Complexity
            if 'summary' in client_metrics and 'complexity' in client_metrics['summary']:
                complexity = client_metrics['summary']['complexity']
                f.write("MODEL COMPLEXITY:\n")
                f.write(f"  GFLOPs:           {complexity.get('gflops', 'N/A'):.3f}\n")
                f.write(f"  Parameters:       {complexity.get('parameters_millions', 'N/A'):.2f} M\n")
                f.write(f"  Model Size:       {complexity.get('model_size_mb', 'N/A'):.2f} MB\n")
                f.write(f"  FLOPs/Param:      {complexity.get('flops_per_parameter', 'N/A'):.2f}\n\n")
            
            # Inference
            if 'summary' in client_metrics and 'inference_summary' in client_metrics['summary']:
                inf_summary = client_metrics['summary']['inference_summary']
                f.write("INFERENCE PERFORMANCE:\n")
                f.write(f"  Mean Time:        {inf_summary.get('inference_time_mean_ms_mean', 'N/A'):.2f} ± "
                       f"{inf_summary.get('inference_time_std_ms_mean', 0):.2f} ms\n")
                f.write(f"  Throughput:       {inf_summary.get('throughput_fps_mean', 'N/A'):.1f} FPS\n")
                f.write(f"  P95 Latency:      {inf_summary.get('inference_time_p95_ms_mean', 'N/A'):.2f} ms\n")
                f.write(f"  P99 Latency:      {inf_summary.get('inference_time_p99_ms_mean', 'N/A'):.2f} ms\n\n")
            
            # Memory
            if 'summary' in client_metrics and 'inference_summary' in client_metrics['summary']:
                inf_summary = client_metrics['summary']['inference_summary']
                f.write("MEMORY USAGE:\n")
                if 'gpu_memory_peak_mb_max' in inf_summary:
                    f.write(f"  Peak GPU Memory:  {inf_summary.get('gpu_memory_peak_mb_max', 'N/A'):.2f} MB\n")
                if 'cpu_memory_rss_mb_max' in inf_summary:
                    f.write(f"  Peak CPU Memory:  {inf_summary.get('cpu_memory_rss_mb_max', 'N/A'):.2f} MB\n")
                f.write("\n")
            
            # Communication
            if 'summary' in client_metrics and 'communication_summary' in client_metrics['summary']:
                comm_summary = client_metrics['summary']['communication_summary']
                f.write("COMMUNICATION:\n")
                f.write(f"  Total Data:       {comm_summary.get('total_communication_mb', 'N/A'):.2f} MB\n")
                f.write(f"  Uploaded:         {comm_summary.get('total_uploaded_mb', 'N/A'):.2f} MB\n")
                f.write(f"  Downloaded:       {comm_summary.get('total_downloaded_mb', 'N/A'):.2f} MB\n\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Complexity and Performance Metrics')
    parser.add_argument('--results_dir', type=str, default='results', 
                       help='Directory containing results')
    parser.add_argument('--output_dir', type=str, default='results/metrics_plots',
                       help='Directory to save plots')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading metrics from: {args.results_dir}")
    metrics = load_metrics(args.results_dir)
    
    if not metrics:
        print("No metrics found!")
        return
    
    print(f"Found metrics for {len(metrics)} clients")
    
    # Generate all plots
    print("\nGenerating visualizations...")
    
    try:
        plot_inference_time_distribution(metrics, output_dir)
    except Exception as e:
        print(f"Error plotting inference time: {e}")
    
    try:
        plot_memory_usage(metrics, output_dir)
    except Exception as e:
        print(f"Error plotting memory usage: {e}")
    
    try:
        plot_communication_costs(metrics, output_dir)
    except Exception as e:
        print(f"Error plotting communication: {e}")
    
    try:
        plot_device_comparison(metrics, output_dir)
    except Exception as e:
        print(f"Error plotting device comparison: {e}")
    
    try:
        plot_efficiency_analysis(metrics, output_dir)
    except Exception as e:
        print(f"Error plotting efficiency: {e}")
    
    try:
        create_summary_report(metrics, output_dir)
    except Exception as e:
        print(f"Error creating summary report: {e}")
    
    print(f"\nAll visualizations saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()