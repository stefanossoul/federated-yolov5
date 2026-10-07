import flwr as fl
import torch
import time
from yolov5.val import run as validate
import yaml
import os
import argparse
import logging
import numpy as np
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.optim as optim
from pathlib import Path
import pandas as pd
import torch.nn as nn  
import sys
import traceback
from tqdm import tqdm
import seaborn as sn

# Import complexity metrics module
from complexity_metrics import ComplexityTracker

# Add YOLOv5 to the Python path
yolov5_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'flower', 'yolov5')
if yolov5_path not in sys.path:
    sys.path.insert(0, yolov5_path)
from yolov5.utils.metrics import ConfusionMatrix
from yolov5.utils.plots import plot_images

# Initialise the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('client.log'), logging.StreamHandler()]
)
logger = logging.getLogger("FL-Client")

# Import the required YOLOv5 modules
from yolov5.models.yolo import Model
from yolov5.utils.dataloaders import LoadImagesAndLabels
from yolov5.utils.loss import ComputeLoss
from yolov5.utils.general import non_max_suppression

# Disable inference mode so tensors can be updated in place
import torch._dynamo
torch._dynamo.config.suppress_errors = True
try:
    torch._C._debug_only_display_inference_mode_tensor_overwarn = False
except:
    pass


class FederatedClient(fl.client.NumPyClient):
    def __init__(self, client_id, config=None, precision='fp32'):
        self.client_id = client_id
        self.config = config
        self.precision = precision
        
        # Initialize paths
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = f'{self.base_path}/flower/yolov5/datasets/coco/client_{client_id}'
        self.model_path = f'{self.base_path}/flower/yolov5/models/yolov5n.yaml'
        self.hyp_path = f'{self.base_path}/flower/yolov5/data/hyps/hyp.scratch-high.yaml'
        self.val_path = f'{self.base_path}/flower/yolov5/datasets/coco/images/val2017'
        self.coco_yaml_path = f'{self.base_path}/flower/yolov5/data/coco.yaml'
        self.logger = logging.getLogger("FL-Client")
        
        logger.info(f"Client {client_id} using precision: {self.precision.upper()}")
        
        # Smart device selection based on client_id
        self.device = self._select_device()
        
        # Initialize model
        self.model = self._init_model()
        
        # Initialize dataloaders with optimized settings
        self.train_loader, self.val_loader = self._init_dataloaders()
        
        # Initialize tracking variables
        self.last_train_loss_items = np.array([0.0, 0.0, 0.0])
        self.last_val_loss_items = np.array([0.0, 0.0, 0.0])
        
        # Initialize loss function
        self.compute_loss = ComputeLoss(self.model)
        
        # Initialize confusion matrix
        with open(self.coco_yaml_path, 'r') as f:
            data_dict = yaml.safe_load(f)
            nc = len(data_dict['names'])
        self.confusion_matrix = ConfusionMatrix(nc=nc, conf=0.25, iou_thres=0.45)
        
        self.results_dir = os.path.join(
            self.base_path, 'flower', 'results', 
            f'client_{self.client_id}_{self.precision}'
        )
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize ComplexityTracker for performance metrics
        try:
            self.metrics_tracker = ComplexityTracker(client_id=self.client_id, results_dir=self.results_dir)
            logger.info(f"ComplexityTracker initialized for client {self.client_id}")
            
            # Measure model complexity (one-time measurement)
            complexity = self.metrics_tracker.measure_model_complexity(
                model=self.model,
                input_size=(1, 3, 640, 640),
                device=self.device
            )
            logger.info(f"Model complexity ({self.precision.upper()}): {complexity.get('gflops', 'N/A')} GFLOPs, "
                       f"{complexity.get('parameters_millions', 'N/A')} M params, "
                       f"{complexity.get('model_size_mb', 'N/A')} MB")
        except Exception as e:
            logger.warning(f"Could not initialize ComplexityTracker: {e}")
            self.metrics_tracker = None
    
    def _select_device(self):
        """Smart device selection to avoid GPU overcrowding."""
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            gpu_memory_free = gpu_memory_total - gpu_memory_used
            
            logger.info(f"GPU Memory: {gpu_memory_used:.1f}/{gpu_memory_total:.1f} GB used")
            
            if gpu_memory_free < 12:
                logger.info(f"Client {self.client_id}: Not enough GPU memory, using CPU")
                return torch.device('cpu')
            
            # Limit GPU memory per process
            torch.cuda.empty_cache()
            memory_fraction = 0.18
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            
            # Additional memory optimizations
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            
            logger.info(f"Client {self.client_id}: Using GPU with {memory_fraction*100}% memory limit")
            return torch.device('cuda')
        
        logger.info(f"Client {self.client_id}: Using CPU")
        return torch.device('cpu')
    
    def _init_model(self):
        """Initialize YOLOv5 model with proper settings."""
        logger.info(f"Initializing YOLOv5 model with {self.precision.upper()} precision...")
        
        # Load parameters
        with open(self.coco_yaml_path, 'r') as f:
            data_dict = yaml.safe_load(f)
            nc = len(data_dict['names'])
            names = data_dict['names']
        
        # Load hyperparameters
        with open(self.hyp_path, 'r') as f:
            hyp = yaml.safe_load(f)
        
        # Create model
        model = Model(cfg=self.model_path, ch=3, nc=nc)
        model.hyp = hyp
        model.nc = nc
        model.names = names
        
        # ========== FP16: Convert model to half precision ==========
        if self.precision == 'fp16':
            logger.info("Converting model to TRUE FP16 (half precision)...")
            model.half()
            logger.info("Model converted to FP16")
            logger.info("  → Model weights: FP16")
            logger.info("  → Computations: FP16")
            logger.info("  → Gradients: FP16")
            logger.info("  → Memory: ~50% reduction")
            logger.info("  → Model size: ~50% reduction")
        
        # ========== INT8: QAT Setup ==========
        elif self.precision == 'int8':
            logger.info("Preparing model for INT8 Quantization-Aware Training...")
            
            # ========== Use qnnpack for GPU compatibility ==========
            # fbgemm: per-channel quantization (CPU only)
            # qnnpack: per-tensor quantization (GPU compatible)
            model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
            logger.info("Using qnnpack backend (GPU-compatible quantization)")
            # =======================================================
            
            # Fuse layers (Conv+BN) - optional for YOLOv5
            try:
                for m in model.modules():
                    if hasattr(m, 'conv') and hasattr(m, 'bn'):
                        torch.quantization.fuse_modules(m, [['conv', 'bn']], inplace=True)
            except Exception as e:
                logger.warning(f"Layer fusing skipped: {e}")
            
            # Prepare for Quantization-Aware Training
            torch.quantization.prepare_qat(model, inplace=True)
            
            logger.info("Model prepared for INT8 QAT")
            logger.info("   - Fake quantization nodes added")
            logger.info("   - Model will simulate INT8 during training")
        # ===========================================================
        
        # Move to device
        model.to(self.device)
        model.train()
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model initialized with {total_params/1e6:.1f}M parameters ({self.precision.upper()})")
        
        return model
    
    def _init_dataloaders(self):
        """Initialize optimized dataloaders."""
        logger.info("Loading datasets with optimized settings...")
        
        # Load hyperparameters
        with open(self.hyp_path, 'r') as f:
            hyp = yaml.safe_load(f)
        
        # ========== INT8: FORCE SMALLER BATCH SIZE ==========
        if self.precision == 'int8':
            # INT8 QAT needs reduced batch size due to memory overhead
            batch_size = 8
            val_batch_size = 4
            logger.info(f"INT8 QAT: Using reduced batch size (train={batch_size}, val={val_batch_size})")
            logger.info(f"Effective batch size maintained at 16 via gradient accumulation")
        
        # ========== FP32/FP16: ADAPTIVE BATCH SIZE ==========
        elif self.device.type == 'cuda':
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            
            if free_memory > 20 * 1024**3:
                batch_size = 16
                val_batch_size = 8
            elif free_memory > 10 * 1024**3:
                batch_size = 8
                val_batch_size = 4
            else:
                batch_size = 4
                val_batch_size = 2
        else:
            batch_size = 4
            val_batch_size = 2
        # ====================================================
        
        logger.info(f"Using batch sizes: train={batch_size}, val={val_batch_size}")
        
        img_size = 640
        cache_images = False
        
        # Random subset selection
        images_per_client = self.config.get('images_per_client', 9000) if self.config else 9000
        current_round = self.config.get('round', 0) if self.config else 0
        
        import glob
        import random
        import tempfile
        
        all_images = glob.glob(f"{self.data_path}/images/*.jpg")
        all_images.extend(glob.glob(f"{self.data_path}/images/*.png"))
        
        logger.info(f"Client {self.client_id} has {len(all_images)} total images")
        
        random.seed(self.client_id * 100 + current_round)
        
        if len(all_images) > images_per_client:
            selected_images = random.sample(all_images, images_per_client)
            logger.info(f"Round {current_round}: Randomly selected {images_per_client} images")
        else:
            selected_images = all_images
        
        # Create temporary text file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        for img_path in selected_images:
            temp_file.write(f"{img_path}\n")
        temp_file.close()
        
        self.temp_txt_file = temp_file.name
        logger.info(f"Created temporary file with {len(selected_images)} image paths: {self.temp_txt_file}")
        
        # Training dataset
        train_dataset = LoadImagesAndLabels(
            path=self.temp_txt_file,
            img_size=img_size,
            batch_size=batch_size,
            augment=True,
            hyp=hyp,
            rect=False,
            cache_images=cache_images,
            stride=32
        )
        
        # Validation dataset
        val_dataset = LoadImagesAndLabels(
            path=self.val_path,
            img_size=img_size,
            batch_size=val_batch_size,
            augment=False,
            hyp=hyp,
            rect=True,
            cache_images=False,
            stride=32
        )
        
        num_workers = 2 if self.device.type == 'cuda' else 1
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_dataset.collate_fn,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=1 if num_workers > 0 else None,
            timeout=60
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=val_dataset.collate_fn,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=1 if num_workers > 0 else None,
            timeout=60
        )
        
        logger.info(f"Datasets loaded: train={len(train_dataset)}, val={len(val_dataset)}")
        
        self.train_batch_size = batch_size
        self.val_batch_size = val_batch_size
        
        return train_loader, val_loader
    
    def get_parameters(self, config):
        """Return model parameters as numpy arrays."""
        logger.info("Getting model parameters")
        params = [param.cpu().numpy() for param in self.model.state_dict().values()]
        
        # Measure communication cost (upload to server)
        if self.metrics_tracker:
            try:
                comm_metrics = self.metrics_tracker.measure_communication_cost(params)
                current_round = config.get('round', 0)
                self.metrics_tracker.record_communication_metrics(current_round, 'upload', comm_metrics)
                logger.info(f"Upload size: {comm_metrics.get('parameters_size_mb', 0):.2f} MB")
            except Exception as e:
                logger.warning(f"Could not measure communication cost: {e}")
        
        return params
    
    def set_parameters(self, parameters):
        """Load model parameters from a list of NumPy arrays."""
        logger.info("Setting model parameters")
        
        try:
            # Build a fresh model
            new_model = self._init_model()
            
            # Load the incoming parameters into it
            params_dict = zip(new_model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v, device=self.device) for k, v in params_dict}
            
            # Load with inference mode disabled
            with torch.inference_mode(False):
                new_model.load_state_dict(state_dict, strict=False)
            
            # Swap in the new model
            self.model = new_model
            
            # Rebuild the loss function against the new model
            self.compute_loss = ComputeLoss(self.model)
            
            # Put the model back into training mode
            self.model.train()
            
            logger.info("Model parameters set successfully")
        except Exception as e:
            logger.error(f"Error setting parameters: {e}")
            logger.error(traceback.format_exc())
            raise e

    def update_metrics_csv(self, round_num, precision, recall, f1, tp, fp, fn):
        """Append this round's metrics to the CSV."""
        save_dir = Path(self.results_dir)
        metrics_file = save_dir / "metrics.csv"
        
        train_box = getattr(self, 'last_train_loss_items', [0, 0, 0])[0]
        train_obj = getattr(self, 'last_train_loss_items', [0, 0, 0])[1]
        train_cls = getattr(self, 'last_train_loss_items', [0, 0, 0])[2]
        
        val_box = getattr(self, 'last_val_loss_items', [0, 0, 0])[0]
        val_obj = getattr(self, 'last_val_loss_items', [0, 0, 0])[1]
        val_cls = getattr(self, 'last_val_loss_items', [0, 0, 0])[2]
        
        if not metrics_file.exists():
            with open(metrics_file, 'w') as f:
                f.write("round,precision,recall,f1,tp,fp,fn,train/box_loss,train/obj_loss,train/cls_loss,val/box_loss,val/obj_loss,val/cls_loss\n")
        
        with open(metrics_file, 'a') as f:
            f.write(f"{round_num},{precision},{recall},{f1},{tp},{fp},{fn},"
                    f"{train_box:.6f},{train_obj:.6f},{train_cls:.6f},"
                    f"{val_box:.6f},{val_obj:.6f},{val_cls:.6f}\n")
        
        logger.info(f"Updated metrics CSV for round {round_num}")
    
    def fit(self, parameters, config):
        """Optimized training with better memory management."""
        current_round = config.get("round", 0)
        logger.info(f"Starting training for round {current_round} ({self.precision.upper()})")
        
        start_time = time.time()
        
        # Memory cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Set parameters
        self.set_parameters(parameters)
        
        # Training settings
        epochs = 4
        
        # Learning rate schedule
        if current_round < 5:
            lr = 0.001 * (1 + current_round / 5)
        elif current_round < 40:
            lr = 0.01
        elif current_round < 60:
            lr = 0.01 * (1 - (current_round - 40) / 20)
        else:
            lr = 0.001
        
        logger.info(f"Learning rate: {lr:.6f}")
        
        # Optimizer with proper weight decay
        g = [], [], []
        for v in self.model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                g[2].append(v.bias)
            if isinstance(v, nn.BatchNorm2d):
                g[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                g[0].append(v.weight)
        
        optimizer = optim.SGD(g[0], lr=lr, momentum=0.937, nesterov=True, weight_decay=0.0005)
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})
        optimizer.add_param_group({'params': g[2], 'weight_decay': 0.0})
        
        # ========== AMP: Only for FP32, NOT for FP16 or INT8! ==========
        # FP16 uses model.half() instead of AMP
        # INT8 uses QAT
        use_amp = False
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        logger.info(f"AMP enabled: {use_amp} (precision: {self.precision})")
        # ================================================================
        
        # Gradient accumulation
        base_batch_size = 16
        accumulation_steps = max(1, base_batch_size // self.train_batch_size)
        
        losses = []
        best_loss = float('inf')
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            mloss = torch.zeros(3, device=self.device)
            
            pbar = tqdm(enumerate(self.train_loader), 
                       total=len(self.train_loader), 
                       desc=f'Client {self.client_id} [{self.precision.upper()}] Epoch {epoch+1}/{epochs}',
                       ncols=120)
            
            optimizer.zero_grad()
            
            for batch_idx, (imgs, targets, paths, _) in pbar:
                # Move data to device
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0
                
                # ========== FP16: Convert input to half ==========
                if self.precision == 'fp16':
                    imgs = imgs.half()
                # =================================================
                
                targets = targets.to(self.device, non_blocking=True)
                
                if len(targets) == 0:
                    continue
                
                # ========== Forward pass: autocast only for FP32+AMP ==========
                with torch.cuda.amp.autocast(enabled=use_amp):
                    preds = self.model(imgs)
                    loss, loss_items = self.compute_loss(preds, targets)
                    loss = loss / accumulation_steps
                # ==============================================================
                
                if not torch.isfinite(loss):
                    logger.warning(f"Non-finite loss detected, skipping batch")
                    optimizer.zero_grad()
                    continue
                
                # ========== Backward pass: scale only if AMP ==========
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()  # Simple backward for FP16/INT8
                # ======================================================
                
                # ========== Gradient accumulation: conditional on AMP ==========
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    
                    if use_amp:
                        # AMP path with scaler
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=10.0
                        )
                        
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            logger.warning(f"NaN/Inf gradients detected, skipping update")
                            scaler.update()
                            optimizer.zero_grad()
                            continue
                            
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Simple path for FP16/INT8 (no scaler)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=10.0
                        )
                        
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            logger.warning(f"NaN/Inf gradients detected, skipping update")
                            optimizer.zero_grad()
                            continue
                            
                        optimizer.step()
                    
                    optimizer.zero_grad()
                # ===============================================================
                
                # Record metrics
                actual_loss = loss.item() * accumulation_steps
                self.last_train_loss_items = loss_items.detach().cpu().numpy()
                
                if np.isfinite(actual_loss):
                    epoch_loss += actual_loss
                    batch_count += 1
                    mloss = (mloss * batch_idx + loss_items.detach()) / (batch_idx + 1)
                    
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                    pbar.set_postfix_str(
                        f'Loss: {actual_loss:.4f} ({mloss[0]:.4f} box, {mloss[1]:.4f} obj, {mloss[2]:.4f} cls) '
                        f'Mem: {mem}'
                    )
                
                if self.device.type == 'cuda' and batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
            
            pbar.close()
            
            avg_epoch_loss = epoch_loss / max(1, batch_count)
            losses.append(avg_epoch_loss)
            
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.4f} "
                f"({mloss[0]:.4f} box, {mloss[1]:.4f} obj, {mloss[2]:.4f} cls)"
            )
        
        # Cleanup
        optimizer.zero_grad()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s")
        
        avg_loss = sum(losses) / len(losses) if losses else float('inf')
        
        # Record training metrics
        if self.metrics_tracker:
            try:
                training_metrics = {
                    'training_time_seconds': float(total_time),
                    'avg_loss': float(avg_loss),
                    'best_loss': float(best_loss),
                    'epochs_completed': len(losses),
                    'learning_rate': float(lr),
                    'batch_size': self.train_batch_size,
                    'device': str(self.device),
                    'precision': self.precision
                }
                
                memory_metrics = self.metrics_tracker.measure_memory_usage(stage='training')
                training_metrics.update(memory_metrics)
                
                gpu_util = self.metrics_tracker.measure_gpu_utilization()
                cpu_util = self.metrics_tracker.measure_cpu_utilization()
                training_metrics.update(gpu_util)
                training_metrics.update(cpu_util)
                
                self.metrics_tracker.record_training_metrics(current_round, training_metrics)
                logger.info(f"Training metrics recorded for round {current_round}")
            except Exception as e:
                logger.warning(f"Could not record training metrics: {e}")
        
        # Cleanup temp file
        if hasattr(self, 'temp_txt_file') and os.path.exists(self.temp_txt_file):
            os.unlink(self.temp_txt_file)
            logger.info(f"Cleaned up temp file")
        
        return self.get_parameters({"round": current_round}), len(self.train_loader.dataset), {
            "loss": float(avg_loss),
            "box_loss": float(self.last_train_loss_items[0]),
            "obj_loss": float(self.last_train_loss_items[1]),
            "cls_loss": float(self.last_train_loss_items[2]),
            "training_time": float(total_time),
            "epochs_completed": len(losses),
            "best_loss": float(best_loss),
            "final_lr": float(lr),
            "device": str(self.device),
            "batch_size": self.train_batch_size,
            "precision": self.precision
        }
    
    def evaluate(self, parameters, config):
        """Optimized evaluation."""
        current_round = config.get("round", 0)
        logger.info(f"Starting evaluation for round {current_round} ({self.precision.upper()})")
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Set parameters
        self.set_parameters(parameters)
        
        # ========== INT8: Convert QAT to Real INT8 (CPU ONLY!) ==========
        if self.precision == 'int8':
            if not hasattr(self.model, '_is_quantized'):
                logger.info("Converting QAT model to actual INT8...")
                
                # CRITICAL: Move to CPU for INT8 conversion & evaluation
                # PyTorch quantized inference not fully supported on GPU
                self.model.cpu()
                self.device = torch.device('cpu')
                logger.info("Moved model to CPU for INT8 conversion and evaluation")
                logger.info("   (GPU quantized inference not fully supported in PyTorch)")
                
                # MUST be in eval mode!
                self.model.eval()
                
                # Convert: FP32+fake_quant → Real INT8
                torch.quantization.convert(self.model, inplace=True)
                
                # Mark as converted
                self.model._is_quantized = True
                
                # Check size
                import tempfile
                with tempfile.NamedTemporaryFile(delete=True, suffix='.pt') as tmp:
                    torch.save(self.model.state_dict(), tmp.name)
                    size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
                    logger.info(f"INT8 model converted. Size: {size_mb:.2f} MB")
                    logger.info(f"INT8 evaluation running on CPU (slower but accurate)")
            else:
                self.model.eval()
        else:
            self.model.eval()
        # =================================================================
    
        # Create the output directory for this round
        round_save_dir = Path(self.results_dir) / f"round_{current_round}"
        round_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Run validation
        try:
            with open(self.coco_yaml_path, 'r') as f:
                data_dict = yaml.safe_load(f)
                class_names = data_dict['names']
            
            validate_data = {
                'nc': len(class_names),
                'names': class_names
            }
            
            # Run validation (uses self.device, which is CPU for INT8)
            results, maps, times = validate(
                data=validate_data,
                batch_size=self.val_batch_size,
                imgsz=640,
                model=self.model,
                dataloader=self.val_loader,
                compute_loss=self.compute_loss,
                conf_thres=0.25,
                iou_thres=0.45,
                save_json=False,
                verbose=False,
                plots=True,
                save_dir=round_save_dir,
                device=self.device  # CPU for INT8, GPU for FP32/FP16
            )
            
            # Extract metrics
            if len(results) >= 4:
                precision, recall, map50, map = results[:4]
                
                if len(results) >= 8:
                    box_loss, obj_loss, cls_loss = results[4:7]
                    total_loss = results[7]
                    self.last_val_loss_items = [box_loss, obj_loss, cls_loss]
                else:
                    total_loss = results[4] if len(results) > 4 else 0.0
            else:
                precision = recall = map50 = map = total_loss = 0.0
            
            metrics = {
                "precision": float(precision),
                "recall": float(recall),
                "mAP@0.5": float(map50),
                "mAP@0.5:0.95": float(map),
                "loss": float(total_loss)
            }
            
            f1_score = 2 * precision * recall / (precision + recall + 1e-6)
            self.update_metrics_csv(
                round_num=current_round,
                precision=precision,
                recall=recall,
                f1=f1_score,
                tp=0,
                fp=0,
                fn=0
            )
            
            logger.info(f"Evaluation metrics: {metrics}")
            
            # Benchmark inference performance
            if self.metrics_tracker:
                try:
                    logger.info("Benchmarking inference performance...")
                    inference_metrics = self.metrics_tracker.benchmark_inference(
                        model=self.model,
                        dataloader=self.val_loader,
                        warmup_runs=10,
                        test_runs=min(100, len(self.val_loader)),
                        device=self.device  # Will use CPU for INT8
                    )
                    
                    memory_metrics = self.metrics_tracker.measure_memory_usage(stage='inference')
                    inference_metrics.update(memory_metrics)
                    
                    gpu_util = self.metrics_tracker.measure_gpu_utilization()
                    cpu_util = self.metrics_tracker.measure_cpu_utilization()
                    inference_metrics.update(gpu_util)
                    inference_metrics.update(cpu_util)
                    
                    efficiency_metrics = self.metrics_tracker.calculate_efficiency_metrics(inference_metrics)
                    inference_metrics.update(efficiency_metrics)
                    
                    inference_metrics['precision_type'] = self.precision
                    inference_metrics['evaluation_device'] = str(self.device)
                    
                    self.metrics_tracker.record_inference_metrics(current_round, inference_metrics)
                    self.metrics_tracker.record_memory_metrics(current_round, memory_metrics)
                    
                    logger.info(f"Inference: {inference_metrics.get('inference_time_mean_ms', 0):.2f}ms, "
                               f"{inference_metrics.get('throughput_fps', 0):.1f} FPS")
                    
                    self.metrics_tracker.save_all_metrics()
                    
                except Exception as e:
                    logger.warning(f"Could not benchmark inference: {e}")
                    import traceback as tb
                    logger.warning(tb.format_exc())
            
            return float(total_loss), len(self.val_loader.dataset), metrics
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            logger.error(traceback.format_exc())
            
            return 0.0, len(self.val_loader.dataset), {
                "precision": 0.0,
                "recall": 0.0,
                "mAP@0.5": 0.0,
                "mAP@0.5:0.95": 0.0,
                "loss": 0.0,
                "error": str(e)
            }
    

def main():
    parser = argparse.ArgumentParser(description='Optimized Federated Learning Client with Quantization')
    parser.add_argument('--client_id', type=int, required=True, help='Client ID')
    parser.add_argument('--server_address', type=str, default='[::]:8080', help='Server address')
    parser.add_argument('--precision', type=str, default='fp32',
                       choices=['fp32', 'fp16', 'int8'],
                       help='Model precision: fp32 (default), fp16, or int8')
    
    args = parser.parse_args()
    
    logger.info(f"Starting optimized client {args.client_id} with {args.precision.upper()} precision")
    
    try:
        # Create optimized client
        client = FederatedClient(
            client_id=args.client_id,
            precision=args.precision
        )
        
        # Start federated learning
        fl.client.start_numpy_client(server_address=args.server_address, client=client)
        
        logger.info(f"Client {args.client_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in client {args.client_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()