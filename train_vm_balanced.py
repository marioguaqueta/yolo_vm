#!/usr/bin/env python3
"""
YOLOv11 Balanced Training Script
Project: Guacamaya - Microsoft AI for Good Lab

OPTIMIZED FOR: 
- Training on the 'general_dataset_balanced' (Generated Crops)
- Validating if the balanced/augmented data improves performance.
- Saving results to 'runs/balanced' for easy comparison.
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
    """Training configuration for Balanced Dataset"""

    # Paths
    BASE_DIR = Path(__file__).parent.absolute()
    # POINTING TO THE NEW BALANCED DATASET
    DATASET_ROOT = BASE_DIR.parent / "general_dataset_balanced" / "general_dataset_balanced"

    # Dataset structure (YOLO standard)
    IMAGES_TRAIN = DATASET_ROOT / "images" / "train"
    IMAGES_VAL = DATASET_ROOT / "images" / "val"
    # Note: We might not have a separate 'test' folder in the generated data, 
    # so we can use 'val' for testing or point to the original test set if needed.
    # For now, let's assume standard structure.
    IMAGES_TEST = DATASET_ROOT / "images" / "val" 

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
    MODEL = "yolo11l.pt"        # Start with Large (or use 'x' if you have resources)
    EPOCHS = 100                # Good baseline for comparison
    BATCH_SIZE = 8              
    IMG_SIZE = 1024             # Matches the crop size we generated (1024x1024)
    PATIENCE = 20               
    WORKERS = 8
    SAVE_PERIOD = 5

    # Memory / speed
    CACHE = True                
    RECT = False

    # Optimizer & Learning
    OPTIMIZER = "AdamW"         
    LR0 = 0.001                 
    IOU_TRAIN = 0.5             

    # Augmentation (We already did heavy augmentation offline, so keep online aug moderate)
    DEGREES = 0.0               # Already rotated offline
    SCALE = 0.1                 # Slight scaling
    TRANSLATE = 0.1             
    HSV_H = 0.010               
    HSV_S = 0.3                 
    HSV_V = 0.2                 
    FLIPLR = 0.5                # Still good to have
    FLIPUD = 0.0                
    MOSAIC = 0.2                # Lower mosaic since crops are already focused
    MIXUP = 0.0                 
    COPY_PASTE = 0.0           

    # Device
    if torch.cuda.is_available():
        DEVICE = 0
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = "cpu"
        print("⚠ WARNING: No GPU detected. Training will be very slow!")

    # WandB
    WANDB_PROJECT = "yolov11-wildlife-balanced"
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
        "test": "images/val", # Use val as test for now
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
            model_name = Path(config.MODEL).stem 
            run_name = f"{model_name}-BALANCED-{config.IMG_SIZE}px"
            
            run = wandb.init(
                project=config.WANDB_PROJECT,
                entity=config.WANDB_ENTITY,
                name=run_name,
                config={
                    "model": config.MODEL,
                    "epochs": config.EPOCHS,
                    "batch_size": config.BATCH_SIZE,
                    "img_size": config.IMG_SIZE,
                    "dataset": "balanced_crops",
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
    print(f"STARTING YOLOv11 TRAINING (Balanced Dataset)")
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
        "project": str(config.BASE_DIR / "runs" / "balanced"), # Save to 'runs/balanced'
        "name": "yolov11_balanced",
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
        "multi_scale": False, # Disable multi-scale since crops are fixed size

        # Reduced Online Augmentation (Since we did it offline)
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
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 on Balanced Dataset")
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
        print("   Run 'generate_balanced_data.py' first!")
        return 1

    yaml_path = create_data_yaml(config)
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
