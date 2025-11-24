#!/usr/bin/env python3
"""
YOLOv11 Wildlife Detection Training Script - High-Performance VM Version (v3)
Project: Guacamaya - Microsoft AI for Good Lab

OPTIMIZED FOR: 
- 25GB+ GPU VRAM (A10G, A100, or similar)
- Aerial wildlife detection (small objects)
- Maximum accuracy (mAP)

Key changes vs v2:
- Model: yolo11x.pt (Extra Large) for maximum capacity
- Batch Size: Increased to 8 (or 16 if memory permits) for stable gradients
- Epochs: Increased to 150 for full convergence
- Augmentations: Enabled flips and increased mosaic for better generalization
"""

import os
import sys
import yaml
from pathlib import Path
import argparse
import shutil

import torch
from ultralytics import YOLO
import wandb


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration for aerial wildlife detection on High-End GPU"""

    # Paths
    BASE_DIR = Path(__file__).parent.absolute()
    DATASET_ROOT = BASE_DIR.parent / "general_dataset"  # sibling directory

    # Dataset structure (YOLO standard)
    IMAGES_TRAIN = DATASET_ROOT / "images" / "train"
    IMAGES_VAL = DATASET_ROOT / "images" / "val"
    IMAGES_TEST = DATASET_ROOT / "images" / "test"

    LABELS_TRAIN = DATASET_ROOT / "labels" / "train"
    LABELS_VAL = DATASET_ROOT / "labels" / "val"
    LABELS_TEST = DATASET_ROOT / "labels" / "test"

    # Classes (0-indexed)
    CLASS_NAMES = {
        0: "Buffalo",
        1: "Elephant",
        2: "Kudu",
        3: "Topi",
        4: "Warthog",
        5: "Waterbuck",
    }

    # Model & training hyperparameters
    MODEL = "yolo11x.pt"        # UPGRADE: Using Extra Large model
    EPOCHS = 150                # UPGRADE: More epochs for convergence
    BATCH_SIZE = 8              # UPGRADE: Increased from 2 to 8 (Try 16 if VRAM allows)
    IMG_SIZE = 1536             # High resolution for small objects
    PATIENCE = 25               # Increased patience
    WORKERS = 8
    SAVE_PERIOD = 5

    # Memory / speed
    CACHE = True                # UPGRADE: Enable RAM caching if you have >32GB System RAM
    RECT = False

    # Optimizer & Learning
    OPTIMIZER = "AdamW"         
    LR0 = 0.0015                # Slightly lower initial LR for the larger model
    IOU_TRAIN = 0.5             # Standard IoU often works better with larger batch sizes

    # Aerial-wildlife-specific augmentation
    # Adjusted for better generalization
    DEGREES = 15.0              # UPGRADE: Increased rotation range
    SCALE = 0.25                # Slight increase
    TRANSLATE = 0.10            
    HSV_H = 0.015               
    HSV_S = 0.6                 
    HSV_V = 0.4                 
    FLIPLR = 0.5                # UPGRADE: Enabled horizontal flip (animals can face left/right)
    FLIPUD = 0.0                # Keep vertical flip off (unless purely top-down 90deg view)
    MOSAIC = 0.50               # UPGRADE: Increased mosaic for better context learning
    MIXUP = 0.1                 # UPGRADE: Slight mixup enabled
    COPY_PASTE = 0.15           

    # Device
    if torch.cuda.is_available():
        DEVICE = 0
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = "cpu"
        print("⚠ WARNING: No GPU detected. Training will be very slow!")

    # WandB
    WANDB_PROJECT = "yolov11-wildlife-detection"
    WANDB_ENTITY = None         
    WANDB_RUN_NAME = None       


# ============================================================================
# UTILS
# ============================================================================

def print_system_info():
    print("\n" + "=" * 70)
    print("SYSTEM INFORMATION")
    print("=" * 70)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA: {torch.version.cuda}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU Memory: {mem:.2f} GB")
        
        if mem < 20:
            print("  ⚠ WARNING: Less than 20GB VRAM detected.")
            print("  Consider reducing BATCH_SIZE to 4 or 2 if OOM errors occur.")
    else:
        print("⚠ GPU: Not available (CPU only)")

    total, used, free = shutil.disk_usage("/")
    print(f"Disk: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")
    print("=" * 70)


# ============================================================================
# DATA CONFIGURATION
# ============================================================================

def create_data_yaml(config: Config) -> Path:
    """Create data.yaml configuration pointing to the YOLO-style dataset."""
    print("\n" + "=" * 70)
    print("CONFIGURING DATASET")
    print("=" * 70)

    yaml_path = config.DATASET_ROOT / "data.yaml"

    data_yaml = {
        "path": str(config.DATASET_ROOT.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(config.CLASS_NAMES),
        "names": list(config.CLASS_NAMES.values()),
    }

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"✓ data.yaml created at: {yaml_path}")
    
    # Quick stats
    train_images = list(config.IMAGES_TRAIN.glob("*.jpg")) + list(config.IMAGES_TRAIN.glob("*.JPG"))
    print(f"  Training images : {len(train_images)}")
    
    return yaml_path


# ============================================================================
# WANDB
# ============================================================================

def initialize_wandb(config: Config, enabled: bool = True):
    if not enabled:
        print("\n⚠ W&B tracking disabled")
        return None

    print("\n" + "=" * 70)
    print("INITIALIZING WEIGHTS & BIASES")
    print("=" * 70)

    try:
        if wandb.login(relogin=False):
            # Generate run name based on model and image size
            model_name = Path(config.MODEL).stem  # e.g., "yolo11x"
            run_name = f"{model_name}-HIGH-RES-{config.IMG_SIZE}px-v3"
            
            run = wandb.init(
                project=config.WANDB_PROJECT,
                entity=config.WANDB_ENTITY,
                name=run_name,
                config={
                    "model": config.MODEL,
                    "epochs": config.EPOCHS,
                    "batch_size": config.BATCH_SIZE,
                    "img_size": config.IMG_SIZE,
                    "patience": config.PATIENCE,
                    "optimizer": config.OPTIMIZER,
                    "lr0": config.LR0,
                    "aug_mosaic": config.MOSAIC,
                    "aug_mixup": config.MIXUP,
                    "aug_fliplr": config.FLIPLR,
                },
            )
            print(f"✓ WandB initialized: {run_name}")
            return run
    except Exception as e:
        print(f"⚠ Could not initialize WandB: {e}")
        print("  Training will continue without logging.")

    return None


# ============================================================================
# TRAINING
# ============================================================================

def train_model(config: Config, yaml_path: Path, use_wandb: bool = True):
    print("\n" + "=" * 70)
    print(f"STARTING YOLOv11 TRAINING (High-Performance Mode)")
    print(f"Model: {config.MODEL} | Batch: {config.BATCH_SIZE} | Img: {config.IMG_SIZE}")
    print("=" * 70)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = YOLO(config.MODEL)

    train_args = {
        "data": str(yaml_path),
        "epochs": config.EPOCHS,
        "imgsz": config.IMG_SIZE,
        "batch": config.BATCH_SIZE,
        "device": config.DEVICE,
        "patience": config.PATIENCE,
        "project": str(config.BASE_DIR / "runs"),
        "name": "yolov11_wildlife_v3",
        "exist_ok": True,
        "pretrained": True,
        "optimizer": config.OPTIMIZER,
        "lr0": config.LR0,
        "iou": config.IOU_TRAIN,
        "verbose": True,
        "seed": 42,
        "plots": True,
        "save": True,
        "save_period": config.SAVE_PERIOD,
        "workers": config.WORKERS,
        "amp": True,
        "cache": config.CACHE,
        "multi_scale": True,

        # Enhanced Augmentations
        "hsv_h": config.HSV_H,
        "hsv_s": config.HSV_S,
        "hsv_v": config.HSV_V,
        "degrees": config.DEGREES,
        "translate": config.TRANSLATE,
        "scale": config.SCALE,
        "flipud": config.FLIPUD,
        "fliplr": config.FLIPLR,
        "mosaic": config.MOSAIC,
        "mixup": config.MIXUP,
        "copy_paste": config.COPY_PASTE,
    }

    results = model.train(**train_args)
    return model, results



# ============================================================================
# TUNING (GENETIC EVOLUTION)
# ============================================================================

def custom_search_space(trial=None):
    """
    Defines the hyperparameter search space for Genetic Evolution.
    Returns a dictionary of parameter distributions.
    """
    return {
        # Optimizer parameters
        "lr0": (1e-5, 1e-2),             # Initial learning rate
        "lrf": (0.01, 1.0),              # Final learning rate (lr0 * lrf)
        "momentum": (0.6, 0.98),         # SGD momentum/Adam beta1
        "weight_decay": (0.0, 0.001),    # Optimizer weight decay
        "warmup_epochs": (0.0, 5.0),     # Warmup epochs
        
        # Augmentation parameters
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

def tune_model(config: Config, yaml_path: Path):
    print("\n" + "=" * 70)
    print("STARTING HYPERPARAMETER EVOLUTION (GENETIC ALGORITHM)")
    print("=" * 70)
    
    # Tuning settings (hardcoded for reasonable defaults, can be exposed if needed)
    EPOCHS_PER_GEN = 15
    ITERATIONS = 30
    
    print(f"Model: {config.MODEL}")
    print(f"Generations: {ITERATIONS}")
    print(f"Epochs per Gen: {EPOCHS_PER_GEN}")
    
    model = YOLO(config.MODEL)
    
    print("\nStarting evolution... This will take time.")
    print("Check 'runs/detect/wildlife_evolution/tune' for progress.")
    
    model.tune(
        data=str(yaml_path),
        epochs=EPOCHS_PER_GEN,
        iterations=ITERATIONS,
        optimizer=config.OPTIMIZER,
        plots=True,
        save=False,
        val=True,
        imgsz=1024,      # Faster tuning at slightly lower res
        batch=16,        # Maximize batch for speed
        space=custom_search_space(), # Pass the dictionary, not the function
        name="wildlife_evolution"
    )
    
    print("\n" + "=" * 70)
    print("EVOLUTION COMPLETE")
    print("=" * 70)
    print("1. Go to 'runs/detect/wildlife_evolution/tune'")
    print("2. Look for 'best_hyperparameters.yaml'")
    print("3. Update Config class in this script with the new values.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train or Tune YOLOv11x High-Performance")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter evolution instead of training")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--batch", type=int, help="Override batch size")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    
    parser.add_argument("--weights", type=str, help="Custom weights path (e.g. runs/.../best.pt)")
    
    args = parser.parse_args()

    config = Config()
    
    if args.weights:
        config.MODEL = args.weights
        print(f"✓ Using custom weights: {config.MODEL}")

    if args.batch:
        config.BATCH_SIZE = args.batch
    if args.epochs:
        config.EPOCHS = args.epochs

    print_system_info()
    
    # Basic checks
    if not config.DATASET_ROOT.exists():
        print(f"❌ ERROR: Dataset directory not found at {config.DATASET_ROOT}")
        return 1

    yaml_path = create_data_yaml(config)

    # BRANCH: TUNING OR TRAINING
    if args.tune:
        tune_model(config, yaml_path)
        return 0

    wandb_run = initialize_wandb(config, enabled=not args.no_wandb)

    try:
        model, _ = train_model(config, yaml_path, use_wandb=(wandb_run is not None))
        
        # Validation
        print("\nValidating best model...")
        metrics = model.val()
        print(f"Final mAP50-95: {metrics.box.map}")
        print(f"Final mAP50:    {metrics.box.map50}")

        if wandb_run:
            wandb.finish()
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        if wandb_run:
            wandb.finish(exit_code=1)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
