#!/usr/bin/env python
# -*- coding: utf-8 -*-

import flwr as fl
import argparse
import logging
import os
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
from flwr.common import Parameters, FitRes, ndarrays_to_parameters, parameters_to_ndarrays

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('server.log'), logging.StreamHandler()]
)
logger = logging.getLogger("FL-Server")

# INT8-Aware Federated Learning Strategy
class INT8SmartFedAvg(fl.server.strategy.FedAvg):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 1,
        min_evaluate_clients: int = 1,
        min_available_clients: int = 1,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        initial_parameters=None,
        use_int8: bool = False,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            initial_parameters=initial_parameters,
        )
        
        self.use_int8 = use_int8
        if use_int8:
            logger.info("="*60)
            logger.info("INT8 SMART MODE Server Enabled")
            logger.info("  Features:")
            logger.info("  - INT8-aware parameter aggregation")
            logger.info("  - INT16 support for BatchNorm statistics")
            logger.info("  - Automatic dequantization and requantization")
            logger.info("  - 75% bandwidth reduction")
            logger.info("="*60)
        
        # For storing metrics
        self.fit_metrics_history = []
        self.evaluate_metrics_history = []
        
        # Create results directory
        self.results_dir = Path("results/server")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Track compression statistics
        self.compression_stats = {
            'total_original_size_mb': 0,
            'total_compressed_size_mb': 0,
            'rounds_processed': 0
        }
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[Parameters]:
        """INT8-aware aggregation of training results."""
        if not results:
            return None
        
        # Collect metrics from clients
        fit_metrics = []
        for _, fit_res in results:
            client_metrics = fit_res.metrics
            if client_metrics:
                fit_metrics.append(client_metrics)
        
        # Calculate average metrics
        if fit_metrics:
            avg_metrics = self._calculate_average_metrics(fit_metrics, 'fit')
            logger.info(f"Round {server_round} fit metrics: {avg_metrics}")
            
            # Store metrics
            self.fit_metrics_history.append((server_round, avg_metrics))
            
            # Create training plots
            self._plot_fit_metrics(server_round)
        
        # INT8-aware aggregation
        if self.use_int8:
            logger.info(f"Round {server_round}: Performing INT8 Smart Mode aggregation")
            
            # Extract parameters from results
            weights_results = []
            for _, fit_res in results:
                params = parameters_to_ndarrays(fit_res.parameters)
                weights_results.append((params, fit_res.num_examples))
            
            # Perform INT8-aware aggregation
            aggregated_params = self._aggregate_int8_smart(weights_results, server_round)
            
            # Return aggregated parameters
            return ndarrays_to_parameters(aggregated_params)
        else:
            # Standard FedAvg aggregation
            return super().aggregate_fit(server_round, results, failures)
    
    def _aggregate_int8_smart(self, results: List[Tuple[List[np.ndarray], int]], 
                              server_round: int) -> List[np.ndarray]:
        """Smart INT8/INT16 aware parameter aggregation."""
        
        num_clients = len(results)
        logger.info(f"Aggregating parameters from {num_clients} clients")
        
        # Get first client to determine structure
        first_params, _ = results[0]
        
        # Check for metadata (dtype markers and scales)
        has_metadata = False
        if len(first_params) >= 2:
            last_elem = first_params[-1]
            second_last = first_params[-2]
            
            # Check if we have dtype markers and scales
            if isinstance(last_elem, np.ndarray) and last_elem.dtype == np.uint8:
                if isinstance(second_last, np.ndarray) and second_last.dtype == np.float32:
                    has_metadata = True
        
        if not has_metadata:
            logger.warning("No INT8 metadata found, using standard aggregation")
            return self._standard_aggregate(results)
        
        # Extract metadata
        num_params = len(first_params) - 2  # Exclude scales and dtype markers
        logger.info(f"Found {num_params} model parameters with INT8/INT16 encoding")
        
        aggregated_params = []
        aggregated_scales = []
        dtype_markers = first_params[-1]  # Use first client's dtype markers
        
        original_size = 0
        compressed_size = 0
        
        for param_idx in range(num_params):
            # Collect this parameter from all clients
            client_params = []
            client_weights = []
            
            for params, num_examples in results:
                param = params[param_idx]
                scales = params[-2]  # Scales are second to last
                dtypes = params[-1]  # Dtype markers are last
                
                scale = scales[param_idx] if param_idx < len(scales) else 1.0
                dtype_marker = dtypes[param_idx] if param_idx < len(dtypes) else 1
                
                # Dequantize based on dtype
                if dtype_marker == 2:  # INT16
                    if param.dtype == np.int16:
                        dequantized = param.astype(np.float32) * scale
                    else:
                        dequantized = param.astype(np.float32)
                    original_size += param.size * 4
                    compressed_size += param.size * 2
                else:  # INT8
                    if param.dtype == np.int8:
                        dequantized = param.astype(np.float32) * scale
                    else:
                        dequantized = param.astype(np.float32)
                    original_size += param.size * 4
                    compressed_size += param.size
                
                client_params.append(dequantized)
                client_weights.append(num_examples)
            
            # Weighted average of dequantized parameters
            total_examples = sum(client_weights)
            if total_examples > 0:
                weighted_sum = np.zeros_like(client_params[0], dtype=np.float32)
                
                for param, weight in zip(client_params, client_weights):
                    weighted_sum += param * (weight / total_examples)
                
                # Re-quantize based on original dtype
                dtype_marker = dtype_markers[param_idx] if param_idx < len(dtype_markers) else 1
                
                if dtype_marker == 2:  # INT16
                    if weighted_sum.size > 0:
                        max_val = np.abs(weighted_sum).max()
                        scale = max_val / 32767.0 if max_val > 0 else 1.0
                        quantized = np.round(weighted_sum / scale).clip(-32768, 32767).astype(np.int16)
                        aggregated_params.append(quantized)
                        aggregated_scales.append(scale)
                    else:
                        aggregated_params.append(np.zeros_like(weighted_sum, dtype=np.int16))
                        aggregated_scales.append(1.0)
                else:  # INT8
                    if weighted_sum.size > 0:
                        max_val = np.abs(weighted_sum).max()
                        scale = max_val / 127.0 if max_val > 0 else 1.0
                        quantized = np.round(weighted_sum / scale).clip(-128, 127).astype(np.int8)
                        aggregated_params.append(quantized)
                        aggregated_scales.append(scale)
                    else:
                        aggregated_params.append(np.zeros_like(weighted_sum, dtype=np.int8))
                        aggregated_scales.append(1.0)
            else:
                # No examples, keep first client's parameter
                aggregated_params.append(client_params[0].astype(np.int8))
                aggregated_scales.append(1.0)
        
        # Add metadata (scales and dtype markers)
        aggregated_params.append(np.array(aggregated_scales, dtype=np.float32))
        aggregated_params.append(dtype_markers)  # Keep original dtype markers
        
        # Update compression statistics
        self.compression_stats['total_original_size_mb'] += original_size / (1024**2)
        self.compression_stats['total_compressed_size_mb'] += compressed_size / (1024**2)
        self.compression_stats['rounds_processed'] += 1
        
        avg_compression = (1 - self.compression_stats['total_compressed_size_mb'] / 
                          self.compression_stats['total_original_size_mb']) * 100
        
        logger.info(f"Round {server_round} INT8 aggregation complete:")
        logger.info(f"  - Parameters: {num_params}")
        logger.info(f"  - This round: {original_size/(1024**2):.1f}MB → {compressed_size/(1024**2):.1f}MB")
        logger.info(f"  - Overall compression: {avg_compression:.1f}%")
        
        return aggregated_params
    
    def _standard_aggregate(self, results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """Standard FP32 aggregation fallback."""
        logger.info("Using standard FP32 aggregation")
        
        weights_list = []
        num_examples_list = []
        
        for params, num_examples in results:
            weights_list.append(params)
            num_examples_list.append(num_examples)
        
        total_examples = sum(num_examples_list)
        aggregated = []
        
        for layer_idx in range(len(weights_list[0])):
            layer_aggregated = np.zeros_like(weights_list[0][layer_idx])
            
            for client_idx, client_weights in enumerate(weights_list):
                weight = num_examples_list[client_idx] / total_examples
                layer_aggregated += client_weights[layer_idx] * weight
            
            aggregated.append(layer_aggregated)
        
        return aggregated
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation results from clients."""
        if not results:
            return None
        
        # Collect metrics from clients
        eval_metrics = []
        for _, eval_res in results:
            client_metrics = eval_res.metrics
            if client_metrics:
                eval_metrics.append(client_metrics)
        
        # Calculate average metrics
        if eval_metrics:
            avg_metrics = self._calculate_average_metrics(eval_metrics, 'eval')
            logger.info(f"Round {server_round} evaluation metrics: {avg_metrics}")
            
            # Store metrics
            self.evaluate_metrics_history.append((server_round, avg_metrics))
            
            # Create evaluation plots
            self._plot_eval_metrics(server_round)
        
        return super().aggregate_evaluate(server_round, results, failures)
    
    def _calculate_average_metrics(self, metrics_list: List[Dict], metric_type: str) -> Dict:
        """Calculate average metrics."""
        avg_metrics = {}
        
        # Define numeric metrics based on type
        if metric_type == 'fit':
            numeric_metrics = ['loss', 'box_loss', 'obj_loss', 'cls_loss', 
                             'training_time', 'epochs_completed', 'best_loss', 
                             'final_lr', 'batch_size']
        else:  # eval
            numeric_metrics = ['loss', 'precision', 'recall', 'mAP@0.5', 'mAP@0.5:0.95']
        
        # Calculate averages for numeric metrics
        for metric in numeric_metrics:
            values = []
            for metrics in metrics_list:
                if metric in metrics:
                    try:
                        value = float(metrics[metric])
                        if np.isfinite(value):
                            values.append(value)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert {metric} to float: {metrics.get(metric)}")
            
            if values:
                avg_metrics[metric] = np.mean(values)
        
        # Handle string metrics for fit
        if metric_type == 'fit':
            for metric in ['device', 'execution_precision', 'communication']:
                values = []
                for metrics in metrics_list:
                    if metric in metrics:
                        values.append(str(metrics[metric]))
                
                if values:
                    counter = Counter(values)
                    most_common = counter.most_common(1)[0][0]
                    avg_metrics[f'{metric}_info'] = f"{most_common} ({len(values)} clients)"
        
        return avg_metrics
    
    def _plot_fit_metrics(self, current_round: int):
        """Create training metrics plot."""
        if not self.fit_metrics_history:
            return
            
        rounds = [r for r, _ in self.fit_metrics_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        mode_str = "INT8 Smart Mode" if self.use_int8 else "Standard"
        fig.suptitle(f'Training Metrics - Round {current_round} ({mode_str})', fontsize=16)
        
        # Total Loss
        losses = [m.get("loss", 0) for _, m in self.fit_metrics_history]
        axes[0, 0].plot(rounds, losses, 'b-o', linewidth=2, markersize=8)
        axes[0, 0].set_title('Average Training Loss')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Individual Losses
        box_losses = [m.get("box_loss", 0) for _, m in self.fit_metrics_history]
        obj_losses = [m.get("obj_loss", 0) for _, m in self.fit_metrics_history]
        cls_losses = [m.get("cls_loss", 0) for _, m in self.fit_metrics_history]
        
        axes[0, 1].plot(rounds, box_losses, 'r-o', label='Box Loss', linewidth=2, markersize=6)
        axes[0, 1].plot(rounds, obj_losses, 'g-o', label='Obj Loss', linewidth=2, markersize=6)
        axes[0, 1].plot(rounds, cls_losses, 'm-o', label='Cls Loss', linewidth=2, markersize=6)
        axes[0, 1].set_title('Individual Loss Components')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training Time
        times = [m.get("training_time", 0) for _, m in self.fit_metrics_history]
        axes[1, 0].plot(rounds, times, 'c-o', linewidth=2, markersize=8)
        axes[1, 0].set_title('Average Training Time')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        lrs = [m.get("final_lr", 0) for _, m in self.fit_metrics_history]
        axes[1, 1].plot(rounds, lrs, 'orange', marker='o', linewidth=2, markersize=8)
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        suffix = "_int8_smart" if self.use_int8 else "_standard"
        plt.savefig(self.results_dir / f"training_metrics{suffix}.png", dpi=200)
        plt.close()
    
    def _plot_eval_metrics(self, current_round: int):
        """Create evaluation metrics plot."""
        if not self.evaluate_metrics_history:
            return
        
        rounds = [r for r, _ in self.evaluate_metrics_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        mode_str = "INT8 Smart Mode" if self.use_int8 else "Standard"
        fig.suptitle(f'Evaluation Metrics - Round {current_round} ({mode_str})', fontsize=16)
        
        # Precision & Recall
        precisions = [m.get("precision", 0) for _, m in self.evaluate_metrics_history]
        recalls = [m.get("recall", 0) for _, m in self.evaluate_metrics_history]
        
        axes[0, 0].plot(rounds, precisions, 'b-o', label='Precision', linewidth=2, markersize=8)
        axes[0, 0].plot(rounds, recalls, 'r-o', label='Recall', linewidth=2, markersize=8)
        axes[0, 0].set_title('Precision & Recall')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1])
        
        # mAP scores
        map50s = [m.get("mAP@0.5", 0) for _, m in self.evaluate_metrics_history]
        map50_95s = [m.get("mAP@0.5:0.95", 0) for _, m in self.evaluate_metrics_history]
        
        axes[0, 1].plot(rounds, map50s, 'g-o', label='mAP@0.5', linewidth=2, markersize=8)
        axes[0, 1].plot(rounds, map50_95s, 'm-o', label='mAP@0.5:0.95', linewidth=2, markersize=8)
        axes[0, 1].set_title('Mean Average Precision')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])
        
        # Validation Loss
        val_losses = [m.get("loss", 0) for _, m in self.evaluate_metrics_history]
        axes[1, 0].plot(rounds, val_losses, 'orange', marker='o', linewidth=2, markersize=8)
        axes[1, 0].set_title('Validation Loss')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Compression Statistics (if INT8)
        if self.use_int8 and self.compression_stats['rounds_processed'] > 0:
            rounds_list = list(range(1, self.compression_stats['rounds_processed'] + 1))
            compression_ratio = (1 - self.compression_stats['total_compressed_size_mb'] / 
                               self.compression_stats['total_original_size_mb']) * 100
            
            axes[1, 1].bar(['Original', 'Compressed'], 
                          [self.compression_stats['total_original_size_mb'], 
                           self.compression_stats['total_compressed_size_mb']],
                          color=['red', 'green'])
            axes[1, 1].set_title(f'Cumulative Data Transfer ({compression_ratio:.1f}% reduction)')
            axes[1, 1].set_ylabel('Size (MB)')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Precision-Recall Curve
            if len(precisions) > 0 and len(recalls) > 0:
                axes[1, 1].plot(recalls, precisions, 'purple', marker='o', linewidth=2, markersize=8)
                axes[1, 1].set_title('Precision-Recall Curve')
                axes[1, 1].set_xlabel('Recall')
                axes[1, 1].set_ylabel('Precision')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_xlim([0, 1])
                axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        suffix = "_int8_smart" if self.use_int8 else "_standard"
        plt.savefig(self.results_dir / f"evaluation_metrics{suffix}.png", dpi=200)
        plt.close()

def get_on_fit_config(epochs: int = 4, images_per_client: int = 2000):
    """Return training configuration for each round."""
    def on_fit_config(server_round: int):
        return {
            "epochs": epochs,
            "round": server_round,
            "batch_size": 16,
            "images_per_client": images_per_client
        }
    return on_fit_config

def get_on_evaluate_config():
    """Return evaluation configuration for each round."""
    def on_evaluate_config(server_round: int):
        return {
            "round": server_round,
        }
    return on_evaluate_config

def main():
    parser = argparse.ArgumentParser(description='INT8 Smart Mode Federated Learning Server')
    parser.add_argument('--rounds', type=int, default=10, help='Number of federated learning rounds')
    parser.add_argument('--epochs', type=int, default=2, help='Number of local epochs for each round')
    parser.add_argument('--min_clients', type=int, default=1, help='Minimum number of clients to start a round')
    parser.add_argument('--port', type=int, default=8080, help='Port to use for the server')
    parser.add_argument('--images_per_client', type=int, default=2000, 
                       help='Number of images per client per round')
    parser.add_argument('--use_int8', action='store_true', 
                       help='Enable INT8 Smart Mode aggregation')
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Starting Federated Learning Server")
    logger.info(f"Configuration:")
    logger.info(f"  - Rounds: {args.rounds}")
    logger.info(f"  - Epochs per round: {args.epochs}")
    logger.info(f"  - Min clients: {args.min_clients}")
    logger.info(f"  - Images per client: {args.images_per_client}")
    logger.info(f"  - INT8 Smart Mode: {args.use_int8}")
    if args.use_int8:
        logger.info("INT8 Features Enabled:")
        logger.info("  - Smart quantization (INT8 for weights, INT16 for BatchNorm)")
        logger.info("  - Gradient caching optimization")
        logger.info("  - Checkpoint compression")
        logger.info("  - 75% bandwidth reduction")
    logger.info("="*60)
    
    # Use INT8-aware strategy
    strategy = INT8SmartFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        on_fit_config_fn=get_on_fit_config(args.epochs, args.images_per_client),
        on_evaluate_config_fn=get_on_evaluate_config(),
        use_int8=args.use_int8,
    )
    
    # Start Flower server
    server_address = f"[::]:{args.port}"
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    
    # Print final statistics if INT8 was used
    if args.use_int8 and strategy.compression_stats['rounds_processed'] > 0:
        logger.info("="*60)
        logger.info("Final INT8 Smart Mode Statistics:")
        logger.info(f"  Total Original Size: {strategy.compression_stats['total_original_size_mb']:.1f} MB")
        logger.info(f"  Total Compressed Size: {strategy.compression_stats['total_compressed_size_mb']:.1f} MB")
        logger.info(f"  Overall Compression: {(1 - strategy.compression_stats['total_compressed_size_mb'] / strategy.compression_stats['total_original_size_mb']) * 100:.1f}%")
        logger.info(f"  Rounds Processed: {strategy.compression_stats['rounds_processed']}")
        logger.info("="*60)
    
    logger.info("Server finished successfully")

if __name__ == "__main__":
    main()