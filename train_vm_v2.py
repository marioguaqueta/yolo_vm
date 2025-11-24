#!/usr/bin/env python3
"""
YOLOv11 Wildlife Detection Training Script - Aerial Small-Object Optimized
Project: Guacamaya - Microsoft AI for Good Lab

OPTIMIZED FOR: Aerial wildlife detection with small objects (animals in savanna)

Key optimizations vs standard YOLO training:
- Default model: yolo11x.pt (best capacity for small objects)
- Aerial-friendly augmentations:
  * Low mosaic (0.2): Strong mosaic hurts small object localization
  * No flips (0.0): Wildlife orientation matters (top-down aerial view)
  * Gentle rotation (5°): Too much rotation breaks small animal features
  * Small scale (0.2): Large scale changes lose tiny animals
- Lower IoU threshold (0.4): Better recall on small/overlapping boxes
- AdamW optimizer + lower LR (0.002): Better stability for fine-tuning
- Multi-scale training: Critical for varying animal sizes
- Smoothed F1-score (EMA): Cleaner monitoring in Weights & Biases

Usage:
  # Basic training (recommended defaults for aerial wildlife)
  python train_vm_v2.py

  # Continue training from checkpoint
  python train_vm_v2.py --weights runs/yolov11_wildlife/weights/best.pt --epochs 100

  # Use smaller model (faster, less GPU memory)
  python train_vm_v2.py --model yolo11m.pt --batch 4 --imgsz 2048

  # Custom augmentation (experiment carefully - defaults are optimized!)
  python train_vm_v2.py --rotate 10.0 --scale 0.3 --mosaic 0.3
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
    """Training configuration for aerial wildlife detection"""

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
    MODEL = "yolo11l.pt"        # default; override with --model or --weights
    EPOCHS = 80
    BATCH_SIZE = 2              # small because images are large (yolo11x + 1536px)
    IMG_SIZE = 1536             # good trade-off for 2048px originals
    PATIENCE = 15               # early stopping patience
    WORKERS = 8
    SAVE_PERIOD = 5

    # Memory / speed
    CACHE = False
    RECT = False

    # Optimizer & Learning
    OPTIMIZER = "AdamW"         # Better for fine-tuning than SGD
    LR0 = 0.002                 # Lower learning rate for stability
    IOU_TRAIN = 0.4             # Lower IoU threshold helps small object detection

    # Aerial-wildlife-specific augmentation
    # (small objects, top-down, little orientation change)
    DEGREES = 5.0               # gentle rotation (too much hurts small objects)
    SCALE = 0.20                # smaller scale = better small-object detection
    TRANSLATE = 0.10            # moderate translation
    HSV_H = 0.01                # minimal hue shift (wildlife colors matter)
    HSV_S = 0.5                 # moderate saturation
    HSV_V = 0.3                 # moderate brightness
    FLIPLR = 0.0                # do NOT flip wildlife horizontally (orientation matters)
    FLIPUD = 0.0                # never flip vertically (top-down view)
    MOSAIC = 0.20               # low mosaic: too much mosaic hurts small-object location
    MIXUP = 0.0                 # disabled for aerial
    COPY_PASTE = 0.10           # helps animals slightly

    # Device
    if torch.cuda.is_available():
        DEVICE = 0
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = "cpu"
        print("⚠ WARNING: No GPU detected. Training will be very slow!")

    # WandB
    WANDB_PROJECT = "yolov11-wildlife-detection"
    WANDB_ENTITY = None         # put your username if needed
    WANDB_RUN_NAME = None       # Will be set dynamically based on MODEL


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
        "test": "images/test",
        "nc": len(config.CLASS_NAMES),
        "names": list(config.CLASS_NAMES.values()),
    }

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"✓ data.yaml created at: {yaml_path}")
    print(f"  Path:   {data_yaml['path']}")
    print(f"  Train:  {data_yaml['train']}")
    print(f"  Val:    {data_yaml['val']}")
    print(f"  Test:   {data_yaml['test']}")
    print(f"  Classes ({len(config.CLASS_NAMES)}): {data_yaml['names']}")

    # Quick stats
    train_images = list(config.IMAGES_TRAIN.glob("*.jpg")) + list(
        config.IMAGES_TRAIN.glob("*.JPG")
    )
    val_images = list(config.IMAGES_VAL.glob("*.jpg")) + list(
        config.IMAGES_VAL.glob("*.JPG")
    )
    train_labels = list(config.LABELS_TRAIN.glob("*.txt"))
    val_labels = list(config.LABELS_VAL.glob("*.txt"))

    print("\nDataset Statistics:")
    print(f"  Training images : {len(train_images)}")
    print(f"  Training labels : {len(train_labels)}")
    print(f"  Validation images: {len(val_images)}")
    print(f"  Validation labels: {len(val_labels)}")

    if len(train_images) != len(train_labels):
        print("  ⚠ WARNING: Mismatch between training images and labels")
    if len(val_images) != len(val_labels):
        print("  ⚠ WARNING: Mismatch between validation images and labels")

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
            model_name = Path(config.MODEL).stem  # e.g., "yolo11x" or "best"
            run_name = f"{model_name}-wildlife-{config.IMG_SIZE}px"
            
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
                    "device": config.DEVICE,
                    "optimizer": config.OPTIMIZER,
                    "lr0": config.LR0,
                    "iou_train": config.IOU_TRAIN,
                    "classes": config.CLASS_NAMES,
                    # Augmentation
                    "degrees": config.DEGREES,
                    "scale": config.SCALE,
                    "mosaic": config.MOSAIC,
                    "copy_paste": config.COPY_PASTE,
                },
            )
            print(f"✓ WandB initialized")
            print(f"  Project: {config.WANDB_PROJECT}")
            print(f"  Run name: {run_name}")
            print(f"  Dashboard: {run.get_url()}")
            return run
    except Exception as e:
        print(f"⚠ Could not initialize WandB: {e}")
        print("  Training will continue without logging.")

    return None


# ============================================================================
# CALLBACKS
# ============================================================================

def on_fit_epoch_end(trainer):
    """
    Callback to compute F1 from precision/recall and log both raw and smoothed curves.
    Uses an exponential moving average (EMA) to smooth the graph.
    """
    if wandb.run is None:
        return

    metrics = trainer.metrics
    p_key = "metrics/precision(B)"
    r_key = "metrics/recall(B)"

    if p_key in metrics and r_key in metrics:
        precision = metrics[p_key]
        recall = metrics[r_key]
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        # EMA smoothing
        alpha = 0.2  # smoothing factor (0.2 → quite smooth)
        if not hasattr(trainer, "f1_ema"):
            trainer.f1_ema = f1
        else:
            trainer.f1_ema = alpha * f1 + (1 - alpha) * trainer.f1_ema

        wandb.log(
            {
                "metrics/f1_score": f1,
                "metrics/f1_score_ema": trainer.f1_ema,
                "epoch": trainer.epoch,
            }
        )


# ============================================================================
# TRAINING
# ============================================================================

def train_model(config: Config, yaml_path: Path, use_wandb: bool = True):
    print("\n" + "=" * 70)
    print("STARTING YOLOv11 TRAINING (Wildlife / Aerial Optimized)")
    print("=" * 70)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ GPU cache cleared")

    print(f"\nLoading model: {config.MODEL}")
    model = YOLO(config.MODEL)

    if use_wandb:
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        print("✓ Added custom F1/EMA logging callback")

    train_args = {
        "data": str(yaml_path),
        "epochs": config.EPOCHS,
        "imgsz": config.IMG_SIZE,
        "batch": config.BATCH_SIZE,
        "device": config.DEVICE,
        "patience": config.PATIENCE,
        "project": str(config.BASE_DIR / "runs"),
        "name": "yolov11_wildlife",
        "exist_ok": True,
        "pretrained": True,
        "optimizer": config.OPTIMIZER,  # AdamW for aerial wildlife
        "lr0": config.LR0,              # lower LR for stability
        "iou": config.IOU_TRAIN,        # lower IoU better for small boxes
        "verbose": True,
        "seed": 42,
        "deterministic": True,
        "plots": True,
        "save": True,
        "save_period": config.SAVE_PERIOD,
        "workers": config.WORKERS,
        "amp": True,                    # Automatic Mixed Precision
        "cache": config.CACHE,
        "rect": config.RECT,
        "multi_scale": True,            # CRITICAL for aerial small objects

        # Aerial-optimized augmentation (gentle transforms for small objects)
        "hsv_h": config.HSV_H,
        "hsv_s": config.HSV_S,
        "hsv_v": config.HSV_V,
        "degrees": config.DEGREES,
        "translate": config.TRANSLATE,
        "scale": config.SCALE,
        "shear": 0.0,                   # disabled for aerial
        "perspective": 0.0,             # disabled for aerial
        "flipud": config.FLIPUD,
        "fliplr": config.FLIPLR,
        "mosaic": config.MOSAIC,
        "mixup": config.MIXUP,
        "copy_paste": config.COPY_PASTE,
    }

    print("\nTraining Configuration:")
    print(f"  Model       : {config.MODEL}")
    print(f"  Epochs      : {config.EPOCHS}")
    print(f"  Batch size  : {config.BATCH_SIZE}")
    print(f"  Image size  : {config.IMG_SIZE}")
    print(f"  Device      : {config.DEVICE}")
    print(f"  Workers     : {config.WORKERS}")
    print(f"  Save period : Every {config.SAVE_PERIOD} epochs")
    
    print(f"\nOptimizer & Learning:")
    print(f"  Optimizer   : {config.OPTIMIZER}")
    print(f"  Learning rate: {config.LR0}")
    print(f"  IoU (train) : {config.IOU_TRAIN}")
    print(f"  Multi-scale : {train_args['multi_scale']}")
    print(f"  Mixed Precision: Enabled")
    
    print(f"\nAerial-Optimized Augmentation (Small Objects):")
    print(f"  Rotation    : ±{config.DEGREES}° (gentle)")
    print(f"  Scale/Zoom  : {config.SCALE * 100:.0f}% (small-object friendly)")
    print(f"  Translation : {config.TRANSLATE * 100:.0f}%")
    print(f"  HSV (H/S/V) : {config.HSV_H:.3f} / {config.HSV_S:.2f} / {config.HSV_V:.2f}")
    print(f"  Flips (LR/UD): {config.FLIPLR} / {config.FLIPUD} (disabled for aerial)")
    print(f"  Mosaic      : {config.MOSAIC * 100:.0f}% (low to preserve small objects)")
    print(f"  Copy-paste  : {config.COPY_PASTE * 100:.0f}%")
    print(f"  Mixup       : {config.MIXUP} (disabled)")
    
    print(f"\nMemory & Speed:")
    print(f"  Cache       : {config.CACHE}")
    print(f"  Rect        : {config.RECT}")
    print(f"  WandB       : {'Enabled' if use_wandb else 'Disabled'}")

    print("\n" + "=" * 70)
    print("TRAINING IN PROGRESS...")
    print("=" * 70 + "\n")

    results = model.train(**train_args)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)

    return model, results


# ============================================================================
# VALIDATION
# ============================================================================

def validate_model(model: YOLO, yaml_path: Path, config: Config, log_to_wandb: bool = False):
    print("\n" + "=" * 70)
    print("VALIDATING MODEL (TEST SPLIT)")
    print("=" * 70)

    results = model.val(
        data=str(yaml_path),
        split="test",
        imgsz=config.IMG_SIZE,
        batch=config.BATCH_SIZE,
        device=config.DEVICE,
        plots=True,
    )

    precision = results.box.mp
    recall = results.box.mr
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print("\nValidation Results:")
    print(f"  mAP50    : {results.box.map50:.4f}")
    print(f"  mAP50-95 : {results.box.map:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1 Score : {f1_score:.4f}")

    # Log custom metrics to Wandb
    if log_to_wandb and wandb.run is not None:
        wandb.log(
            {
                "test/mAP50": results.box.map50,
                "test/mAP50-95": results.box.map,
                "test/precision": precision,
                "test/recall": recall,
                "test/f1_score": f1_score,
            }
        )
        print("✓ Test metrics logged to WandB")

    # Per-class metrics
    if hasattr(results.box, "ap_class_index"):
        print(f"\nPer-class mAP50:")
        per_class_metrics = {}
        for idx, class_idx in enumerate(results.box.ap_class_index):
            class_name = config.CLASS_NAMES[int(class_idx)]
            map50 = results.box.ap50[idx]
            print(f"  {class_name:12s}: {map50:.4f}")
            
            # Log per-class metrics to Wandb
            if log_to_wandb and wandb.run is not None:
                per_class_metrics[f"test/{class_name}_mAP50"] = map50
        
        if per_class_metrics and wandb.run is not None:
            wandb.log(per_class_metrics)
            print("✓ Per-class mAP50 logged to WandB")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv11x for Wildlife Aerial Detection (Small-Object Optimized)"
    )

    # Basic training parameters
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--epochs", type=int, help="Number of epochs (default: 80)")
    parser.add_argument("--batch", type=int, help="Batch size (default: 2 for yolo11x)")
    parser.add_argument("--imgsz", type=int, help="Image size (default: 1536)")
    parser.add_argument("--patience", type=int, help="Early stopping patience (default: 15)")
    parser.add_argument("--weights", type=str, help="Custom .pt weights to fine-tune")
    parser.add_argument("--model", type=str, help="Base model (yolo11s/m/l/x.pt, default: yolo11x.pt)")
    parser.add_argument("--wandb-key", type=str, help="WandB API key")
    
    # Augmentation parameters (aerial-optimized defaults, but customizable)
    parser.add_argument("--rotate", type=float, help="Rotation degrees (default: 5.0)")
    parser.add_argument("--scale", type=float, help="Scale/zoom (default: 0.20)")
    parser.add_argument("--translate", type=float, help="Translation (default: 0.10)")
    parser.add_argument("--hsv-h", type=float, help="HSV Hue (default: 0.01)")
    parser.add_argument("--hsv-s", type=float, help="HSV Saturation (default: 0.5)")
    parser.add_argument("--hsv-v", type=float, help="HSV Value (default: 0.3)")
    parser.add_argument("--mosaic", type=float, help="Mosaic probability (default: 0.20)")
    parser.add_argument("--copy-paste", type=float, help="Copy-paste probability (default: 0.10)")

    args = parser.parse_args()

    if args.wandb_key:
        wandb.login(key=args.wandb_key)

    config = Config()

    # CLI overrides - Basic training parameters
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch:
        config.BATCH_SIZE = args.batch
    if args.imgsz:
        config.IMG_SIZE = args.imgsz
    if args.patience:
        config.PATIENCE = args.patience
    if args.model:
        config.MODEL = args.model
    
    # CLI overrides - Augmentation parameters
    if args.rotate is not None:
        config.DEGREES = args.rotate
    if args.scale is not None:
        config.SCALE = args.scale
    if args.translate is not None:
        config.TRANSLATE = args.translate
    if args.hsv_h is not None:
        config.HSV_H = args.hsv_h
    if args.hsv_s is not None:
        config.HSV_S = args.hsv_s
    if args.hsv_v is not None:
        config.HSV_V = args.hsv_v
    if args.mosaic is not None:
        config.MOSAIC = args.mosaic
    if args.copy_paste is not None:
        config.COPY_PASTE = args.copy_paste

    # Custom weights override everything
    if args.weights:
        weights_path = Path(args.weights)
        if not weights_path.exists():
            print(f"\n❌ ERROR: Weights file not found: {weights_path}")
            return 1
        config.MODEL = str(weights_path.absolute())
        print(f"\n✓ Using custom pretrained weights: {config.MODEL}")

    print_system_info()

    print("\n" + "=" * 70)
    print("YOLOv11 WILDLIFE DETECTION TRAINING")
    print("=" * 70)
    print(f"\nCode dir   : {config.BASE_DIR}")
    print(f"Dataset dir: {config.DATASET_ROOT}")
    print(f"Model      : {config.MODEL}")
    print(f"Classes    : {list(config.CLASS_NAMES.values())}")

    # Basic checks
    if not config.DATASET_ROOT.exists():
        print(f"\n❌ ERROR: Dataset directory not found at {config.DATASET_ROOT}")
        return 1
    if not config.IMAGES_TRAIN.exists() or not config.LABELS_TRAIN.exists():
        print("\n❌ ERROR: Train images/labels not found.")
        print("   Expecting YOLO structure under general_dataset/:")
        print("     images/train, images/val, images/test")
        print("     labels/train, labels/val, labels/test")
        return 1

    try:
        yaml_path = create_data_yaml(config)
    except Exception as e:
        print(f"\n❌ ERROR creating data.yaml: {e}")
        return 1

    wandb_run = initialize_wandb(config, enabled=not args.no_wandb)

    try:
        model, _ = train_model(config, yaml_path, use_wandb=(wandb_run is not None))
        _ = validate_model(model, yaml_path, config, log_to_wandb=(wandb_run is not None))

        best_model_path = config.BASE_DIR / "runs" / "yolov11_wildlife" / "weights" / "best.pt"
        last_model_path = config.BASE_DIR / "runs" / "yolov11_wildlife" / "weights" / "last.pt"

        print("\n" + "=" * 70)
        print("TRAINING PIPELINE COMPLETE ✓")
        print("=" * 70)
        print(f"\nBest model : {best_model_path}")
        print(f"Last model : {last_model_path}")
        print(f"Results dir: {config.BASE_DIR / 'runs' / 'yolov11_wildlife'}")

        if wandb_run:
            print(f"\nWandB dashboard: {wandb_run.get_url()}")
            wandb.finish()
        print("\n" + "=" * 70)
        return 0

    except Exception as e:
        print(f"\n❌ ERROR during training/validation: {e}")
        import traceback
        traceback.print_exc()
        if wandb_run:
            wandb.finish(exit_code=1)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
