#!/usr/bin/env python3
"""
YOLOv11 Hyperparameter Evolution (Genetic Algorithm) Script
Project: Guacamaya - Microsoft AI for Good Lab

STRATEGY: GENETIC EVOLUTION
Instead of guessing values, this script uses a Genetic Algorithm (GA) to 
"evolve" the best hyperparameters over multiple generations.

It will:
1. Train the model for a few epochs (e.g., 10-30) with random hyperparameters.
2. Measure fitness (mAP50-95).
3. "Mutate" the best values (tweak them slightly).
4. Repeat for N iterations.
5. Output the mathematically optimal set of parameters for your specific dataset.

This is computationally expensive but is the GOLD STANDARD for maximizing performance.
"""

import sys
from pathlib import Path
from ultralytics import YOLO
import torch

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
BASE_DIR = Path(__file__).parent.absolute()
DATASET_ROOT = BASE_DIR.parent / "general_dataset"
DATA_YAML = DATASET_ROOT / "data.yaml"

# Tuning Settings
MODEL_NAME = "yolo11l.pt"   # Use 'l' or 'x'. 'x' is slower to tune. 'l' is a good proxy.
EPOCHS_PER_GEN = 15         # Short runs to quickly evaluate fitness
ITERATIONS = 30             # Number of generations (Total epochs = EPOCHS * ITERATIONS)
OPTIMIZER = "AdamW"         # Base optimizer to tune around

# ============================================================================
# CUSTOM SEARCH SPACE
# ============================================================================
# This defines the min/max ranges for the values you wanted to iterate on.

def custom_search_space(trial=None):
    """
    Defines the hyperparameter search space.
    Returns a dictionary of parameter distributions.
    """
    return {
        # Optimizer parameters
        "lr0": (1e-5, 1e-2),             # Initial learning rate
        "lrf": (0.01, 1.0),              # Final learning rate (lr0 * lrf)
        "momentum": (0.6, 0.98),         # SGD momentum/Adam beta1
        "weight_decay": (0.0, 0.001),    # Optimizer weight decay
        "warmup_epochs": (0.0, 5.0),     # Warmup epochs
        
        # Augmentation parameters (The ones you requested)
        "degrees": (0.0, 45.0),          # Rotation (+/- deg)
        "scale": (0.0, 0.9),             # Image scale (+/- gain)
        "translate": (0.0, 0.9),         # Image translation (+/- fraction)
        "hsv_h": (0.0, 0.1),             # Image HSV-Hue augmentation
        "hsv_s": (0.0, 0.9),             # Image HSV-Saturation augmentation
        "hsv_v": (0.0, 0.9),             # Image HSV-Value augmentation
        "fliplr": (0.0, 1.0),            # Image flip left-right (probability)
        "mosaic": (0.0, 1.0),            # Image mosaic (probability)
        "mixup": (0.0, 1.0),             # Image mixup (probability)
        "copy_paste": (0.0, 1.0),        # Segment copy-paste (probability)
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("STARTING HYPERPARAMETER EVOLUTION")
    print(f"Model: {MODEL_NAME}")
    print(f"Generations: {ITERATIONS}")
    print(f"Epochs per Gen: {EPOCHS_PER_GEN}")
    print("=" * 70)

    # Check GPU
    if not torch.cuda.is_available():
        print("❌ ERROR: GPU required for hyperparameter tuning.")
        return

    # Ensure data.yaml exists
    if not DATA_YAML.exists():
        print(f"❌ ERROR: {DATA_YAML} not found. Run train_vm_v3.py first to generate it.")
        return

    # Initialize Model
    model = YOLO(MODEL_NAME)

    # Start Tuning
    # The 'tune' method will automatically save the best hyperparameters to runs/detect/tune/best_hyperparameters.yaml
    print("\nStarting evolution... This will take time.")
    print("Check the 'runs/detect/tune' folder for progress.")
    
    model.tune(
        data=str(DATA_YAML),
        epochs=EPOCHS_PER_GEN,
        iterations=ITERATIONS,
        optimizer=OPTIMIZER,
        plots=True,
        save=False,
        val=True,
        imgsz=1024,      # Slightly lower res for faster tuning, then train final model at 1536
        batch=16,        # Maximize batch for speed
        space=custom_search_space, # Inject our custom search space
        name="wildlife_evolution"
    )

    print("\n" + "=" * 70)
    print("EVOLUTION COMPLETE")
    print("=" * 70)
    print("1. Go to 'runs/detect/wildlife_evolution/tune'")
    print("2. Look for 'best_hyperparameters.yaml'")
    print("3. Use those values in 'train_vm_v3.py' for your final long training run.")

if __name__ == "__main__":
    main()
