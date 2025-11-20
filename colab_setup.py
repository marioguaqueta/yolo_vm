"""
Google Colab Setup and Training Script
Copy and paste this entire script into a Colab notebook cell

Usage in Colab:
1. Upload this notebook to Google Colab
2. Run all cells
3. Training will start automatically with wandb integration
"""

# ============================================================================
# CELL 1: Mount Google Drive and Setup
# ============================================================================

from google.colab import drive
import os
import sys

# Mount Google Drive
print("Mounting Google Drive...")
drive.mount('/content/drive')

# Set your project path in Google Drive
# Note: This should point to your CODE directory (yolo_vm)
# The dataset should be in a sibling directory: ../general_dataset
PROJECT_PATH = "/content/drive/MyDrive/MAIA_Final_Project_2025/yolo_vm"  # UPDATE THIS

# Change to project directory
os.chdir(PROJECT_PATH)
print(f"Changed directory to: {os.getcwd()}")

# Check GPU
import torch
print(f"\n{'='*70}")
print("System Information:")
print(f"{'='*70}")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"{'='*70}\n")


# ============================================================================
# CELL 2: Install Dependencies
# ============================================================================

print("Installing dependencies...")
!pip install -q ultralytics>=8.0.0 wandb pandas pillow pyyaml tqdm

# Verify installation
from ultralytics import YOLO
print("✓ Ultralytics YOLO installed")

import wandb
print("✓ Wandb installed")


# ============================================================================
# CELL 3: Login to Wandb (Optional but Recommended)
# ============================================================================

# Login to Weights & Biases
# Get your API key from: https://wandb.ai/authorize
wandb.login()


# ============================================================================
# CELL 4: Download/Verify Dataset
# ============================================================================

# Verify dataset structure
import os
from pathlib import Path

BASE_DIR = Path(PROJECT_PATH)  # yolo_vm directory
DATASET_ROOT = BASE_DIR.parent / "general_dataset"  # sibling directory

print(f"Checking dataset at: {DATASET_ROOT}")
print(f"\nDataset structure:")

required_paths = [
    DATASET_ROOT / "train",
    DATASET_ROOT / "val", 
    DATASET_ROOT / "test",
    DATASET_ROOT / "groundtruth/csv/train_big_size_A_B_E_K_WH_WB.csv",
    DATASET_ROOT / "groundtruth/csv/val_big_size_A_B_E_K_WH_WB.csv",
    DATASET_ROOT / "groundtruth/csv/test_big_size_A_B_E_K_WH_WB.csv",
]

all_exist = True
for path in required_paths:
    exists = path.exists()
    status = "✓" if exists else "✗"
    print(f"{status} {path}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n⚠ WARNING: Some dataset files are missing!")
    print("Please upload your dataset to Google Drive at:")
    print(f"{DATASET_ROOT}")
else:
    print("\n✓ All dataset files found!")
    
    # Count images
    train_images = len(list((DATASET_ROOT / "train").glob("*.JPG")))
    val_images = len(list((DATASET_ROOT / "val").glob("*.JPG")))
    test_images = len(list((DATASET_ROOT / "test").glob("*.JPG")))
    
    print(f"\nDataset statistics:")
    print(f"  Train images: {train_images}")
    print(f"  Val images: {val_images}")
    print(f"  Test images: {test_images}")
    print(f"  Total: {train_images + val_images + test_images}")


# ============================================================================
# CELL 5: Start Training
# ============================================================================

# Update the BASE_DIR in train_vm.py to point to your Colab path
print("Starting training...")
print(f"{'='*70}\n")

# Run the training script
!python3 train_vm.py --epochs 50 --batch 8 --imgsz 2048

print(f"\n{'='*70}")
print("Training complete! Check the results in:")
print(f"{BASE_DIR}/runs/yolov11_wildlife_vm/")
print(f"{'='*70}")


# ============================================================================
# CELL 6 (Optional): Download Trained Model
# ============================================================================

# Download the best model to your local machine
from google.colab import files

best_model = f"{PROJECT_PATH}/runs/yolov11_wildlife_vm/weights/best.pt"

if os.path.exists(best_model):
    print(f"Downloading best model: {best_model}")
    files.download(best_model)
    print("✓ Model downloaded!")
else:
    print(f"⚠ Model not found at: {best_model}")


# ============================================================================
# CELL 7 (Optional): Test Inference
# ============================================================================

from ultralytics import YOLO
from IPython.display import Image, display

# Load the trained model
model = YOLO(f"{PROJECT_PATH}/runs/yolov11_wildlife_vm/weights/best.pt")

# Run inference on a test image
test_image = list(Path(f"{PROJECT_PATH}/general_dataset/test").glob("*.JPG"))[0]

print(f"Running inference on: {test_image.name}")
results = model.predict(
    source=str(test_image),
    imgsz=2048,
    conf=0.25,
    save=True,
    project=f"{PROJECT_PATH}/predictions",
    name="test_inference"
)

# Display result
result_image = f"{PROJECT_PATH}/predictions/test_inference/{test_image.name}"
if os.path.exists(result_image):
    display(Image(filename=result_image))
    print(f"\n✓ Inference complete! See predictions at: {result_image}")

