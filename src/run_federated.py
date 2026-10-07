import os
import subprocess
import argparse
import glob
import time
import signal
import sys
import socket
from multiprocessing import Process
import threading
from pathlib import Path

def is_port_in_use(port):
    """Return True if the port is already bound."""
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
        return s.connect_ex(('::1', port)) == 0

def log_stream(stream, prefix, log_file=None):
    """Stream a subprocess pipe to stdout and optionally to a log file."""
    for line in stream:
        line = line.strip()
        print(f"{prefix}: {line}")
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"{prefix}: {line}\n")

def run_server(rounds, epochs, min_clients, port, images_per_client=9000):
    """Start the server process, checking the port and tailing its output."""
    # Bail out early if the port is taken
    if is_port_in_use(port):
        print(f"ERROR: Port {port} is already in use. Choose a different port.")
        return None
    
    server_cmd = [
        "python", "federated_server.py",
        "--rounds", str(rounds),
        "--epochs", str(epochs),
        "--min_clients", str(min_clients),
        "--port", str(port),
        "--images_per_client", str(images_per_client) 
    ]
    
    print(f"Executing server command: {' '.join(server_cmd)}")
    
    server_process = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # Line buffered
    )
    
    # Threads to tail stdout and stderr
    stdout_log = "server_stdout.log"
    stderr_log = "server_stderr.log"
    
    # Truncate logs from a previous run
    open(stdout_log, 'w').close()
    open(stderr_log, 'w').close()
    
    t1 = threading.Thread(target=log_stream, args=(server_process.stdout, "SERVER OUT", stdout_log), daemon=True)
    t2 = threading.Thread(target=log_stream, args=(server_process.stderr, "SERVER ERR", stderr_log), daemon=True)
    
    t1.start()
    t2.start()
    
    return server_process

def run_client(client_id, server_address, precision='fp32'):
    """Start a client process and tail its output."""
    # Copy the environment and allow in-place updates of inference tensors
    env = os.environ.copy()
    env["PYTORCH_ALLOW_INFERENCE_TENSOR_UPDATES"] = "1"
    
    client_cmd = [
        "python", "federated_client.py",
        "--client_id", str(client_id),
        "--server_address", server_address,
        "--precision", precision
    ]
    
    print(f"Executing client command: {' '.join(client_cmd)}")
    
    client_process = subprocess.Popen(
        client_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffered
        env=env  # Use the modified environment
    )
    
    # Threads to tail stdout and stderr
    stdout_log = f"client_{client_id}_stdout.log"
    stderr_log = f"client_{client_id}_stderr.log"
    
    # Truncate logs from a previous run
    open(stdout_log, 'w').close()
    open(stderr_log, 'w').close()
    
    t1 = threading.Thread(target=log_stream, args=(client_process.stdout, f"CLIENT {client_id} OUT", stdout_log), daemon=True)
    t2 = threading.Thread(target=log_stream, args=(client_process.stderr, f"CLIENT {client_id} ERR", stderr_log), daemon=True)
    
    t1.start()
    t2.start()
    
    return client_process

def client_process_wrapper(client_id, server_address, precision='fp32'):
    """Run a client in its own process and wait for it to finish."""
    process = run_client(client_id, server_address, precision)
    
    # Wait for the client to exit
    process.wait()
    
    print(f"Client {client_id} ({precision.upper()}) process exited with code {process.returncode}")

def visualize_training_evolution(results_csv, output_dir):
    """Plot training evolution from the results CSV."""
    vis_cmd = [
        "python", "-c",
        """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

# Load the data
df = pd.read_csv('%s')
output_dir = Path('%s')
output_dir.mkdir(parents=True, exist_ok=True)

# Set up the subplot grid
fig, axs = plt.subplots(2, 5, figsize=(20, 10))

# Metrics to plot
metrics = [
    'train/box_loss', 'train/obj_loss', 'train/cls_loss', 
    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5',
    'metrics/mAP_0.5:0.95', 'val/box_loss', 'val/obj_loss', 'val/cls_loss'
]

for i, metric in enumerate(metrics):
    if metric not in df.columns:
        continue
    
    row, col = i // 5, i %% 5
    ax = axs[row, col]
    
    ax.plot(df['epoch'], df[metric], 'b-', label='results')
    # Overlay a smoothed line
    if len(df) > 5:
        smooth = gaussian_filter1d(df[metric], sigma=2)
        ax.plot(df['epoch'], smooth, 'r--', label='smooth')
    
    ax.set_title(metric)
    ax.set_xlabel('epoch')
    ax.set_ylabel(metric.split('/')[-1])
    ax.grid(True, alpha=0.3)
    
    if i == 0:
        ax.legend()

plt.tight_layout()
plt.savefig(output_dir / 'training_evolution.png', dpi=200)
plt.close()
print('Training evolution visualization saved to', output_dir / 'training_evolution.png')
        """ % (results_csv, output_dir)
    ]
    
    process = subprocess.Popen(
        vis_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = process.communicate()
    if stdout:
        print(stdout)
    if stderr:
        print("Errors during training evolution visualization:")
        print(stderr)

def visualize_detections(model_path, image_folder, output_folder, round_num, conf_thres=0.25, show_confidence=False):
    """Draw detections on sample images, optionally with confidence scores."""
    # Create the output directory
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Visualisation runs in a subprocess so torch.hub stays out of this process
    vis_cmd = [
        "python", "-c",
        """
import torch
import cv2
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys

try:
    # Load the model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='%s')
    model.conf = %f  # Confidence threshold
    
    show_confidence = %s
    round_num = %d
    
    # Images to process
    image_folder = Path('%s')
    output_folder = Path('%s')
    
    # Subdirectory for the current round
    round_folder = output_folder / f"round_{round_num}"
    round_folder.mkdir(parents=True, exist_ok=True)
    
    # Find images
    image_paths = list(image_folder.glob('*.jpg')) + list(image_folder.glob('*.png'))
    print(f"Found {len(image_paths)} images in folder")
    
    # Sample up to 16 images per round
    if len(image_paths) > 16:
        import random
        image_paths = random.sample(image_paths, 16)
    
    # Build the image grid
    grid_size = min(4, int(np.ceil(np.sqrt(len(image_paths)))))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    axes = axes.flatten() if len(image_paths) > 1 else [axes]
    
    # Process and draw each image
    for i, img_path in enumerate(image_paths):
        if i >= len(axes):
            break
            
        # Run detection
        results = model(str(img_path))
        
        # Read the image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Show it in the matching subplot
        axes[i].imshow(img)
        
        # Overlay the detections
        detections = results.pandas().xyxy[0]
        
        for _, detection in detections.iterrows():
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            class_name = detection['name']
            conf = detection['confidence']
            
            # Colour per class
            color_dict = {
                'person': 'red', 'bicycle': 'blue', 'car': 'green', 'motorcycle': 'orange',
                'airplane': 'yellow', 'bus': 'purple', 'train': 'cyan', 'truck': 'magenta',
                'boat': 'lime', 'traffic light': 'pink', 'fire hydrant': 'gray', 'stop sign': 'brown',
                'parking meter': 'olive', 'bench': 'teal', 'bird': 'navy', 'cat': 'darkred',
                'dog': 'darkgreen', 'horse': 'darkblue', 'sheep': 'darkgray', 'cow': 'darkorange',
                'elephant': 'darkviolet', 'bear': 'darkslategray', 'zebra': 'darkcyan', 'giraffe': 'darkmagenta'
            }
            
            color = color_dict.get(class_name, 'white')
            
            # Draw the bounding box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=2)
            axes[i].add_patch(rect)
            
            # Label, with or without the confidence score
            if show_confidence:
                label = f"{class_name} {conf:.2f}"
            else:
                label = class_name
                
            axes[i].text(x1, y1-5, label, color=color, backgroundcolor='black', fontsize=8)
        
        # Hide the axes
        axes[i].axis('off')
        
        # Use the file name as the title
        axes[i].set_title(img_path.name, fontsize=10)
    
    # Hide unused subplots
    for i in range(len(image_paths), len(axes)):
        axes[i].axis('off')
    
    # Save the grid
    plt.tight_layout()
    if show_confidence:
        plt.savefig(round_folder / f"detections_with_conf.png", dpi=200)
    else:
        plt.savefig(round_folder / f"detections.png", dpi=200)
    plt.close()
    
    # A combined grid across all images is assembled later
    print(f"Detection visualization for round {round_num} complete")
except Exception as e:
    print(f"Error in detection visualization: {e}")
    import traceback
    traceback.print_exc()
        """ % (model_path, conf_thres, str(show_confidence).lower(), round_num, image_folder, output_folder)
    ]
    
    # Render once without confidence scores, then again with them
    process = subprocess.Popen(
        vis_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = process.communicate()
    if stdout:
        print(stdout)
    if stderr:
        print("Errors during detection visualization:")
        print(stderr)
    
    # Second pass, this time showing confidence scores
    if not show_confidence:
        visualize_detections(model_path, image_folder, output_folder, round_num, conf_thres, True)

def run_visualization(results_dir, output_dir, trained_model_path=None, test_images_dir=None):
    """Generate all result visualisations."""
    # Base plots
    vis_cmd = [
        "python", "visualize_results.py",
        "--results_dir", results_dir,
        "--output_dir", output_dir
    ]
    
    print(f"Executing visualization command: {' '.join(vis_cmd)}")
    
    process = subprocess.Popen(
        vis_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = process.communicate()
    print("Visualization output:")
    print(stdout)
    
    if stderr:
        print("Visualization errors:")
        print(stderr)
    
    # Detection visualisations
    if trained_model_path and test_images_dir:
        print("Generating detection visualizations...")
        # Collect every detection image into one final grid
        output_vis_dir = f"{output_dir}/detections"
        os.makedirs(output_vis_dir, exist_ok=True)
        
        # Find detection images across all rounds
        all_detection_images = []
        for client_dir in glob.glob(f"{results_dir}/client_*"):
            for round_dir in glob.glob(f"{client_dir}/detections/round_*"):
                round_num = int(os.path.basename(round_dir).split('_')[1])
                
                # Images with labels only
                detection_img = glob.glob(f"{round_dir}/detections.png")
                if detection_img:
                    all_detection_images.append((round_num, detection_img[0], False))
                
                # Images with confidence scores
                detection_conf_img = glob.glob(f"{round_dir}/detections_with_conf.png")
                if detection_conf_img:
                    all_detection_images.append((round_num, detection_conf_img[0], True))
        
        # Sort by round
        all_detection_images.sort()
        
        # Two grids: one without confidence scores, one with
        for with_conf in [False, True]:
            relevant_images = [img for (_, img, conf) in all_detection_images if conf == with_conf]
            
            if not relevant_images:
                continue
                
            # Assemble the grid
            grid_cmd = [
                "python", "-c",
                f"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import cv2
from pathlib import Path

# Detection images to combine
image_paths = {relevant_images}
output_file = "{output_vis_dir}/{'detections_with_conf' if with_conf else 'detections'}_grid.png"

# Grid size
n_images = len(image_paths)
grid_size = min(5, int(np.ceil(np.sqrt(n_images))))

# Create the figure
fig = plt.figure(figsize=(20, 20))

for i, img_path in enumerate(image_paths[:grid_size*grid_size]):
    if i >= grid_size*grid_size:
        break
        
    # Load the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Add it to the grid
    plt.subplot(grid_size, grid_size, i+1)
    plt.imshow(img)
    
    # Recover the round number from the path
    round_num = "Unknown"
    try:
        # The parent directory is named "round_X"
        parts = Path(img_path).parent.name.split('_')
        if len(parts) >= 2 and parts[0] == "round":
            round_num = parts[1]
    except:
        pass
    
    plt.title(f"Round {round_num}")
    plt.axis('off')

plt.tight_layout()
plt.savefig(output_file, dpi=200, bbox_inches='tight')
plt.close()

print(f"Created detection grid at {{output_file}}")
                """
            ]
            
            subprocess.run(grid_cmd, check=False)
    
    print("All visualizations completed. Check the results directory for output.")

def signal_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) and shut everything down cleanly."""
    print("\nStopping execution...")
    
    # Terminate all client processes
    for process in processes:
        if process.is_alive():
            process.terminate()
    
    # Wait for them to exit
    for process in processes:
        process.join(timeout=1)
    
    if server_process and server_process.poll() is None:
        print("Terminating server process...")
        server_process.terminate()
        server_process.wait(timeout=5)
    
    print("All processes terminated. Exiting.")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='Run Federated Learning with YOLOv5 and Quantization Support')
    parser.add_argument('--num_clients', type=int, default=1, help='Number of clients to run')
    parser.add_argument('--rounds', type=int, default=10, help='Number of federated learning rounds')
    parser.add_argument('--epochs', type=int, default=2, help='Number of local epochs for each round')
    parser.add_argument('--min_clients', type=int, default=1, help='Minimum number of clients to start a round')
    parser.add_argument('--port', type=int, default=8080, help='Port to use for the server')
    parser.add_argument('--images_per_client', type=int, default=9000, help='Number of images per client per round')
    
    parser.add_argument('--precision', type=str, default='fp32',
                       choices=['fp32', 'fp16', 'int8','int8_onnx'],
                       help='Model precision: fp32 (default), fp16, or int8')
    
    parser.add_argument('--visualize_detections', action='store_true', help='Visualize object detections on test images')
    parser.add_argument('--model_path', type=str, help='Path to the trained model for visualization')
    parser.add_argument('--test_images', type=str, help='Path to test images for detection visualization')
    args = parser.parse_args()
    
    print("="*70)
    print(f"Starting Federated Learning with {args.precision.upper()} precision")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Precision: {args.precision.upper()}")
    print(f"  - Clients: {args.num_clients}")
    print(f"  - Rounds: {args.rounds}")
    print(f"  - Epochs per round: {args.epochs}")
    print(f"  - Images per client: {args.images_per_client}")
    print("="*70)
    
    # Server address
    server_address = f"[::]:{args.port}"
    
    # Start the server
    print(f"Starting server with {args.rounds} rounds, {args.epochs} epochs per round")
    global server_process
    server_process = run_server(args.rounds, args.epochs, args.min_clients, args.port, args.images_per_client)
    
    if not server_process:
        print("Failed to start server. Exiting.")
        return
    
    # Give the server time to come up
    print("Waiting for server to start...")
    time.sleep(5)
    
    # Check the server is still alive
    if server_process.poll() is not None:
        print(f"Server process terminated unexpectedly with code {server_process.returncode}")
        return
    
    # Start the clients
    global processes
    processes = []
    
    print(f"Starting {args.num_clients} clients with {args.precision.upper()} precision")
    for client_id in range(args.num_clients):
        print(f"Starting client {client_id} ({args.precision.upper()})")
        client_process = Process(target=client_process_wrapper, args=(client_id, server_address, args.precision))
        client_process.start()
        processes.append(client_process)
    
    # Install the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Wait for all clients to finish
    for process in processes:
        process.join()
    
    # Wait for the server to finish
    print("All clients completed. Waiting for server to complete...")
    
    if server_process.poll() is None:
        server_process.wait()
    
    print(f"Server process exited with code {server_process.returncode}")
    
    print(f"Federated learning with {args.precision.upper()} completed")
    
    # Produce the result visualisations
    print("Generating visualizations...")
    model_path = args.model_path if args.visualize_detections else None
    test_images = args.test_images if args.visualize_detections else None
    
    results_dir = f"results_{args.precision}" if args.precision != 'fp32' else "results"
    run_visualization(results_dir, f"{results_dir}/combined", model_path, test_images)
    
    print(f"All done! Check the '{results_dir}' directory for output.")

if __name__ == "__main__":
    main()