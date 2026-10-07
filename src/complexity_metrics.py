"""
Complexity and Performance Metrics Module for Federated Learning YOLOv5

This module provides comprehensive metrics for:
- Model Complexity (FLOPs, Parameters, Size)
- Inference Performance (Time, Throughput, Latency)
- Memory Usage (GPU/CPU)
- Hardware Utilization
- Communication Costs
- Efficiency Ratios

Handles FP16 and INT8 models when measuring FLOPs.
"""

import torch
import numpy as np
import time
import os
import sys
import json
import csv
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Setup logger
logger = logging.getLogger("ComplexityMetrics")

# Try to import optional dependencies
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    logger.warning("thop not available. Install with: pip install thop")

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False
    logger.warning("pynvml not available. GPU monitoring disabled.")


class ComplexityTracker:
    """
    Tracks and manages all complexity and performance metrics.
    """
    
    def __init__(self, client_id: int, results_dir: str = "results"):
        """
        Initialize the complexity tracker.
        
        Args:
            client_id: ID of the client
            results_dir: Base directory for storing results (should already include client_id if needed)
        """
        self.client_id = client_id
        # Don't append client_id again if results_dir already contains it
        if f"client_{client_id}" in str(results_dir):
            self.results_dir = Path(results_dir) / "metrics"
        else:
            self.results_dir = Path(results_dir) / f"client_{client_id}" / "metrics"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for metrics
        self.complexity_metrics = {}
        self.inference_metrics_history = []
        self.training_metrics_history = []
        self.memory_metrics_history = []
        self.communication_metrics_history = []
        
        # Flags
        self.complexity_measured = False
        
        # Process info
        self.process = psutil.Process(os.getpid())
        
        # GPU info
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available and NVML_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                self.gpu_handle = None
        else:
            self.gpu_handle = None
        
        logger.info(f"ComplexityTracker initialized for client {client_id}")
        logger.info(f"Results directory: {self.results_dir}")
    
    # ========== MODEL COMPLEXITY MEASUREMENT ==========
    
    def measure_model_complexity(self, model: torch.nn.Module, 
                                 input_size: Tuple[int, int, int, int] = (1, 3, 640, 640),
                                 device: torch.device = None) -> Dict:
        """
        Measure model complexity metrics (one-time measurement).
        
        Args:
            model: PyTorch model
            input_size: Input tensor size (batch, channels, height, width)
            device: Device to use for measurement
            
        Returns:
            Dictionary with complexity metrics
        """
        if self.complexity_measured:
            logger.info("Complexity already measured, returning cached results")
            return self.complexity_metrics
        
        logger.info("Measuring model complexity...")
        
        if device is None:
            device = next(model.parameters()).device
        
        metrics = {}
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        metrics['total_parameters'] = int(total_params)
        metrics['trainable_parameters'] = int(trainable_params)
        metrics['non_trainable_parameters'] = int(non_trainable_params)
        metrics['parameters_millions'] = float(total_params / 1e6)
        
        # Measure FLOPs with thop
        if THOP_AVAILABLE:
            try:
                dummy_input = torch.randn(input_size).to(device)
                
                # Match input dtype to model dtype for FP16/INT8 compatibility
                model_dtype = next(model.parameters()).dtype
                
                # Check if model is quantized (INT8)
                is_quantized = any(hasattr(m, 'qconfig') and m.qconfig for m in model.modules())
                
                if model_dtype == torch.float16:
                    dummy_input = dummy_input.half()
                    logger.info("Model is FP16, converting input to FP16 for FLOPs measurement")
                elif is_quantized:
                    logger.info("Model is INT8 QAT, using FP32 input (fake quantization handles conversion)")
                    # INT8 QAT models expect FP32 input, fake quant nodes handle conversion
                
                # Profile a deep copy so the original model is left untouched
                import copy
                model_copy = copy.deepcopy(model)
                model_copy.eval()
                
                flops, params = profile(model_copy, inputs=(dummy_input,), verbose=False)
                
                # Clean up the copy
                del model_copy
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                metrics['flops'] = int(flops)
                metrics['gflops'] = float(flops / 1e9)
                metrics['macs'] = int(flops / 2)  # MACs ≈ FLOPs/2
                
                logger.info(f"FLOPs: {metrics['gflops']:.2f} G, Params: {metrics['parameters_millions']:.2f} M")
                
            except Exception as e:
                logger.warning(f"Could not measure FLOPs with thop: {e}")
                
                # For fully converted INT8 models, thop may fail entirely
                # This is expected - quantized models have different ops
                if is_quantized or 'quantized' in str(type(model)).lower():
                    logger.info("Note: FLOPs measurement not available for fully quantized INT8 models")
                    logger.info("      FLOPs should be measured during QAT phase (before convert)")
                
                metrics['flops'] = None
                metrics['gflops'] = None
                metrics['macs'] = None
        else:
            metrics['flops'] = None
            metrics['gflops'] = None
            metrics['macs'] = None
        
        # Measure model size (in memory)
        try:
            param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
            total_size = param_size + buffer_size
            
            metrics['model_size_bytes'] = int(total_size)
            metrics['model_size_mb'] = float(total_size / (1024 ** 2))
            metrics['model_size_mb_fp32'] = float(total_params * 4 / (1024 ** 2))  # Assuming FP32
            metrics['model_size_mb_fp16'] = float(total_params * 2 / (1024 ** 2))  # Assuming FP16
            
            logger.info(f"Model size: {metrics['model_size_mb']:.2f} MB")
            
        except Exception as e:
            logger.warning(f"Could not measure model size: {e}")
            metrics['model_size_bytes'] = None
            metrics['model_size_mb'] = None
        
        # Calculate efficiency ratios
        if metrics['gflops'] is not None and metrics['parameters_millions'] > 0:
            metrics['flops_per_parameter'] = float(metrics['flops'] / total_params)
        else:
            metrics['flops_per_parameter'] = None
        
        self.complexity_metrics = metrics
        self.complexity_measured = True
        
        # Save to file
        self._save_complexity_report()
        
        return metrics
    
    # ========== INFERENCE PERFORMANCE MEASUREMENT ==========
    
    def benchmark_inference(self, model: torch.nn.Module, 
                           dataloader: torch.utils.data.DataLoader,
                           warmup_runs: int = 10,
                           test_runs: int = 100,
                           device: torch.device = None) -> Dict:
        """
        Benchmark inference performance with detailed timing.
        
        Args:
            model: PyTorch model
            dataloader: DataLoader for inference
            warmup_runs: Number of warmup iterations
            test_runs: Number of test iterations
            device: Device to use
            
        Returns:
            Dictionary with inference metrics
        """
        logger.info(f"Benchmarking inference (warmup={warmup_runs}, test={test_runs})...")
        
        if device is None:
            device = next(model.parameters()).device
        
        model.eval()
        
        # Import time_sync if available
        try:
            from yolov5.utils.general import time_sync
        except:
            # Fallback to regular time
            def time_sync():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                return time.time()
        
        times = []
        batch_sizes = []
        
        # Warmup phase
        logger.info("Running warmup iterations...")
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= warmup_runs:
                    break
                
                imgs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                # Normalize images to [0, 1] if needed
                if imgs.dtype == torch.uint8:
                    imgs = imgs.float() / 255.0
                _ = model(imgs)
        
        # Actual benchmark
        logger.info("Running benchmark iterations...")
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= test_runs:
                    break
                
                imgs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                # Normalize images to [0, 1] if needed
                if imgs.dtype == torch.uint8:
                    imgs = imgs.float() / 255.0
                batch_size = imgs.shape[0]
                
                start = time_sync()
                _ = model(imgs)
                end = time_sync()
                
                elapsed_ms = (end - start) * 1000
                times.append(elapsed_ms)
                batch_sizes.append(batch_size)
        
        # Calculate statistics
        times = np.array(times)
        total_images = sum(batch_sizes)
        
        metrics = {
            'inference_time_mean_ms': float(np.mean(times)),
            'inference_time_std_ms': float(np.std(times)),
            'inference_time_min_ms': float(np.min(times)),
            'inference_time_max_ms': float(np.max(times)),
            'inference_time_median_ms': float(np.median(times)),
            'inference_time_p50_ms': float(np.percentile(times, 50)),
            'inference_time_p95_ms': float(np.percentile(times, 95)),
            'inference_time_p99_ms': float(np.percentile(times, 99)),
            'throughput_fps': float(1000 / np.mean(times)) if np.mean(times) > 0 else 0,
            'images_per_second': float(total_images / (np.sum(times) / 1000)),
            'total_images_processed': int(total_images),
            'batch_size_avg': float(np.mean(batch_sizes)),
        }
        
        logger.info(f"Inference: {metrics['inference_time_mean_ms']:.2f}±{metrics['inference_time_std_ms']:.2f} ms, "
                   f"{metrics['throughput_fps']:.1f} FPS")
        
        return metrics
    
    # ========== MEMORY MEASUREMENT ==========
    
    def measure_memory_usage(self, stage: str = "inference") -> Dict:
        """
        Measure current memory usage (GPU and CPU).
        
        Args:
            stage: Stage of measurement (e.g., 'training', 'inference')
            
        Returns:
            Dictionary with memory metrics
        """
        metrics = {
            'stage': stage,
            'timestamp': time.time()
        }
        
        # CPU Memory
        try:
            mem_info = self.process.memory_info()
            metrics['cpu_memory_rss_mb'] = float(mem_info.rss / (1024 ** 2))
            metrics['cpu_memory_vms_mb'] = float(mem_info.vms / (1024 ** 2))
            
            # System-wide memory
            sys_mem = psutil.virtual_memory()
            metrics['system_memory_used_mb'] = float(sys_mem.used / (1024 ** 2))
            metrics['system_memory_percent'] = float(sys_mem.percent)
            
        except Exception as e:
            logger.warning(f"Could not measure CPU memory: {e}")
        
        # GPU Memory
        if self.gpu_available:
            try:
                metrics['gpu_memory_allocated_mb'] = float(torch.cuda.memory_allocated() / (1024 ** 2))
                metrics['gpu_memory_reserved_mb'] = float(torch.cuda.memory_reserved() / (1024 ** 2))
                metrics['gpu_memory_peak_mb'] = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
                
                if self.gpu_handle and NVML_AVAILABLE:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    metrics['gpu_memory_total_mb'] = float(mem_info.total / (1024 ** 2))
                    metrics['gpu_memory_used_mb'] = float(mem_info.used / (1024 ** 2))
                    metrics['gpu_memory_free_mb'] = float(mem_info.free / (1024 ** 2))
                    metrics['gpu_memory_utilization_percent'] = float(mem_info.used / mem_info.total * 100)
                
            except Exception as e:
                logger.warning(f"Could not measure GPU memory: {e}")
        
        return metrics
    
    # ========== GPU UTILIZATION ==========
    
    def measure_gpu_utilization(self) -> Dict:
        """
        Measure GPU utilization metrics.
        
        Returns:
            Dictionary with GPU utilization metrics
        """
        metrics = {}
        
        if not self.gpu_available or not self.gpu_handle or not NVML_AVAILABLE:
            return metrics
        
        try:
            # Utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            metrics['gpu_utilization_percent'] = float(utilization.gpu)
            metrics['gpu_memory_utilization_percent'] = float(utilization.memory)
            
            # Temperature
            temperature = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            metrics['gpu_temperature_celsius'] = float(temperature)
            
            # Power
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                metrics['gpu_power_draw_watts'] = float(power / 1000)  # Convert mW to W
            except:
                pass
            
        except Exception as e:
            logger.warning(f"Could not measure GPU utilization: {e}")
        
        return metrics
    
    # ========== CPU UTILIZATION ==========
    
    def measure_cpu_utilization(self) -> Dict:
        """
        Measure CPU utilization metrics.
        
        Returns:
            Dictionary with CPU utilization metrics
        """
        metrics = {}
        
        try:
            metrics['cpu_percent'] = float(psutil.cpu_percent(interval=0.1))
            metrics['cpu_percent_per_core'] = [float(x) for x in psutil.cpu_percent(interval=0.1, percpu=True)]
            metrics['cpu_count_physical'] = int(psutil.cpu_count(logical=False))
            metrics['cpu_count_logical'] = int(psutil.cpu_count(logical=True))
            
        except Exception as e:
            logger.warning(f"Could not measure CPU utilization: {e}")
        
        return metrics
    
    # ========== COMMUNICATION COST ==========
    
    def measure_communication_cost(self, parameters: List[np.ndarray]) -> Dict:
        """
        Measure the size of model parameters being communicated.
        
        Args:
            parameters: List of numpy arrays representing model parameters
            
        Returns:
            Dictionary with communication metrics
        """
        metrics = {}
        
        try:
            # Calculate total size
            total_bytes = sum(p.nbytes for p in parameters)
            metrics['parameters_size_bytes'] = int(total_bytes)
            metrics['parameters_size_mb'] = float(total_bytes / (1024 ** 2))
            metrics['parameters_size_kb'] = float(total_bytes / 1024)
            metrics['num_parameters_arrays'] = len(parameters)
            
            logger.info(f"Communication cost: {metrics['parameters_size_mb']:.2f} MB")
            
        except Exception as e:
            logger.warning(f"Could not measure communication cost: {e}")
        
        return metrics
    
    # ========== RECORD METRICS ==========
    
    def record_inference_metrics(self, round_num: int, metrics: Dict):
        """Record inference metrics for a specific round."""
        metrics['round'] = round_num
        metrics['client_id'] = self.client_id
        self.inference_metrics_history.append(metrics)
    
    def record_training_metrics(self, round_num: int, metrics: Dict):
        """Record training metrics for a specific round."""
        metrics['round'] = round_num
        metrics['client_id'] = self.client_id
        self.training_metrics_history.append(metrics)
    
    def record_memory_metrics(self, round_num: int, metrics: Dict):
        """Record memory metrics for a specific round."""
        metrics['round'] = round_num
        metrics['client_id'] = self.client_id
        self.memory_metrics_history.append(metrics)
    
    def record_communication_metrics(self, round_num: int, direction: str, metrics: Dict):
        """
        Record communication metrics.
        
        Args:
            round_num: Round number
            direction: 'upload' or 'download'
            metrics: Communication metrics
        """
        metrics['round'] = round_num
        metrics['client_id'] = self.client_id
        metrics['direction'] = direction
        self.communication_metrics_history.append(metrics)
    
    # ========== EFFICIENCY CALCULATIONS ==========
    
    def calculate_efficiency_metrics(self, inference_metrics: Dict) -> Dict:
        """
        Calculate efficiency ratios combining complexity and performance metrics.
        
        Args:
            inference_metrics: Inference performance metrics
            
        Returns:
            Dictionary with efficiency metrics
        """
        efficiency = {}
        
        if not self.complexity_metrics:
            logger.warning("Complexity metrics not available for efficiency calculation")
            return efficiency
        
        gflops = self.complexity_metrics.get('gflops')
        params_millions = self.complexity_metrics.get('parameters_millions')
        inference_time_ms = inference_metrics.get('inference_time_mean_ms')
        fps = inference_metrics.get('throughput_fps')
        
        if gflops and inference_time_ms:
            efficiency['time_per_gflop_ms'] = float(inference_time_ms / gflops)
        
        if gflops and fps:
            efficiency['fps_per_gflop'] = float(fps / gflops)
        
        if gflops and params_millions:
            efficiency['flops_per_million_params'] = float(gflops / params_millions)
        
        # Memory efficiency (if available)
        if 'gpu_memory_allocated_mb' in inference_metrics and params_millions:
            efficiency['memory_per_million_params_mb'] = float(
                inference_metrics['gpu_memory_allocated_mb'] / params_millions
            )
        
        return efficiency
    
    # ========== SAVE METHODS ==========
    
    def _save_complexity_report(self):
        """Save complexity metrics to a text report."""
        report_path = self.results_dir / "complexity_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"MODEL COMPLEXITY REPORT - Client {self.client_id}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("PARAMETERS:\n")
            f.write(f"  Total Parameters:       {self.complexity_metrics.get('parameters_millions', 0):.2f} M\n")
            f.write(f"  Trainable Parameters:   {self.complexity_metrics.get('trainable_parameters', 0):,}\n")
            f.write(f"  Non-trainable Params:   {self.complexity_metrics.get('non_trainable_parameters', 0):,}\n\n")
            
            f.write("COMPUTATIONAL COMPLEXITY:\n")
            gflops = self.complexity_metrics.get('gflops')
            if gflops:
                f.write(f"  FLOPs:                  {gflops:.3f} G\n")
                f.write(f"  MACs:                   {self.complexity_metrics.get('macs', 0) / 1e9:.3f} G\n")
            else:
                f.write(f"  FLOPs:                  Not measured (thop not available)\n")
            
            f.write("\nMODEL SIZE:\n")
            f.write(f"  Memory (actual):        {self.complexity_metrics.get('model_size_mb', 0):.2f} MB\n")
            f.write(f"  Size (FP32):            {self.complexity_metrics.get('model_size_mb_fp32', 0):.2f} MB\n")
            f.write(f"  Size (FP16):            {self.complexity_metrics.get('model_size_mb_fp16', 0):.2f} MB\n\n")
            
            f.write("EFFICIENCY:\n")
            flops_per_param = self.complexity_metrics.get('flops_per_parameter')
            if flops_per_param:
                f.write(f"  FLOPs per Parameter:    {flops_per_param:.2f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        logger.info(f"Complexity report saved to {report_path}")
    
    def save_all_metrics(self):
        """Save all collected metrics to CSV files."""
        
        # Save inference metrics
        if self.inference_metrics_history:
            csv_path = self.results_dir / "inference_metrics.csv"
            self._save_to_csv(self.inference_metrics_history, csv_path)
            logger.info(f"Inference metrics saved to {csv_path}")
        
        # Save training metrics
        if self.training_metrics_history:
            csv_path = self.results_dir / "training_metrics.csv"
            self._save_to_csv(self.training_metrics_history, csv_path)
            logger.info(f"Training metrics saved to {csv_path}")
        
        # Save memory metrics
        if self.memory_metrics_history:
            csv_path = self.results_dir / "memory_metrics.csv"
            self._save_to_csv(self.memory_metrics_history, csv_path)
            logger.info(f"Memory metrics saved to {csv_path}")
        
        # Save communication metrics
        if self.communication_metrics_history:
            csv_path = self.results_dir / "communication_metrics.csv"
            self._save_to_csv(self.communication_metrics_history, csv_path)
            logger.info(f"Communication metrics saved to {csv_path}")
        
        # Save summary JSON
        self._save_summary_json()
    
    def _save_to_csv(self, data: List[Dict], filepath: Path):
        """Save list of dictionaries to CSV."""
        if not data:
            return
        
        # Get all unique keys
        all_keys = set()
        for item in data:
            all_keys.update(item.keys())
        
        fieldnames = sorted(all_keys)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    
    def _save_summary_json(self):
        """Save a summary of all metrics to JSON."""
        summary = {
            'client_id': self.client_id,
            'complexity': self.complexity_metrics,
            'inference_summary': self._summarize_metrics(self.inference_metrics_history),
            'training_summary': self._summarize_metrics(self.training_metrics_history),
            'memory_summary': self._summarize_metrics(self.memory_metrics_history),
            'communication_summary': self._summarize_communication(),
        }
        
        json_path = self.results_dir / "metrics_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary JSON saved to {json_path}")
    
    def _summarize_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Calculate summary statistics for a list of metrics."""
        if not metrics_list:
            return {}
        
        summary = {}
        
        # Get numeric fields
        numeric_fields = set()
        for item in metrics_list:
            for key, value in item.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_fields.add(key)
        
        # Calculate statistics
        for field in numeric_fields:
            values = [item[field] for item in metrics_list if field in item and item[field] is not None]
            if values:
                summary[f'{field}_mean'] = float(np.mean(values))
                summary[f'{field}_std'] = float(np.std(values))
                summary[f'{field}_min'] = float(np.min(values))
                summary[f'{field}_max'] = float(np.max(values))
        
        return summary
    
    def _summarize_communication(self) -> Dict:
        """Summarize communication metrics."""
        if not self.communication_metrics_history:
            return {}
        
        uploads = [m for m in self.communication_metrics_history if m.get('direction') == 'upload']
        downloads = [m for m in self.communication_metrics_history if m.get('direction') == 'download']
        
        summary = {}
        
        if uploads:
            upload_sizes = [m.get('parameters_size_mb', 0) for m in uploads]
            summary['total_uploaded_mb'] = float(sum(upload_sizes))
            summary['avg_upload_size_mb'] = float(np.mean(upload_sizes))
        
        if downloads:
            download_sizes = [m.get('parameters_size_mb', 0) for m in downloads]
            summary['total_downloaded_mb'] = float(sum(download_sizes))
            summary['avg_download_size_mb'] = float(np.mean(download_sizes))
        
        if uploads or downloads:
            all_sizes = [m.get('parameters_size_mb', 0) for m in self.communication_metrics_history]
            summary['total_communication_mb'] = float(sum(all_sizes))
        
        return summary


# Utility function for timing with GPU synchronization
def time_sync():
    """Accurate time measurement with GPU synchronization."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()