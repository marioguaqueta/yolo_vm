#!/usr/bin/env python3
"""
YOLOv11 Wildlife Detection Training Script - Aerial Small-Object Optimized
Project: Guacamaya - Microsoft AI for Good Lab

Key changes vs original:
- Default model: yolo11x.pt (better for small objects; use --model to change)
- Aerial-friendly augmentations (low mosaic, no flips, gentle rotation & scale)
- Lower IoU threshold for training (0.4) to boost recall on small boxes
- AdamW optimizer, lower LR, multi-scale training
- Smoothed F1-score logged to Weights & Biases
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
    MODEL = "yolo11x.pt"        # default; override with --model or --weights
    EPOCHS = 80
    BATCH_SIZE = 2              # small because images are large
    IMG_SIZE = 1536             # good trade-off for 2048px originals
    PATIENCE = 15
    WORKERS = 8
    SAVE_PERIOD = 5

    # Memory / speed
    CACHE = False
    RECT = False

    # Aerial-wildlife-specific augmentation
    # (small objects, top-down, little orientation change)
    DEGREES = 5.0       # gentle rotation
    SCALE = 0.20        # moderate zoom; >0.3 often hurts small objects
    TRANSLATE = 0.10
    HSV_H = 0.010
    HSV_S = 0.50
    HSV_V = 0.30
    FLIPLR = 0.0        # no flips → wildlife has consistent orientation
    FLIPUD = 0.0
    MOSAIC = 0.20       # low mosaic: strong mosaic kills tiny animals
    MIXUP = 0.0
    COPY_PASTE = 0.10

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
    WANDB_RUN_NAME = f"yolo11x-wildlife-{IMG_SIZE}px"


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
            run = wandb.init(
                project=config.WANDB_PROJECT,
                entity=config.WANDB_ENTITY,
                name=config.WANDB_RUN_NAME,
                config={
                    "model": config.MODEL,
                    "epochs": config.EPOCHS,
                    "batch_size": config.BATCH_SIZE,
                    "img_size": config.IMG_SIZE,
                    "patience": config.PATIENCE,
                    "device": config.DEVICE,
                    "classes": config.CLASS_NAMES,
                },
            )
            print(f"✓ WandB initialized: {run.get_url()}")
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
        "optimizer": "AdamW",
        "lr0": 0.002,          # lower LR than default
        "iou": 0.40,           # better for small boxes
        "verbose": True,
        "seed": 42,
        "deterministic": True,
        "plots": True,
        "save": True,
        "save_period": config.SAVE_PERIOD,
        "workers": config.WORKERS,
        "amp": True,
        "cache": config.CACHE,
        "rect": config.RECT,
        "multi_scale": True,   # important for aerial small objects

        # Augmentation
        "hsv_h": config.HSV_H,
        "hsv_s": config.HSV_S,
        "hsv_v": config.HSV_V,
        "degrees": config.DEGREES,
        "translate": config.TRANSLATE,
        "scale": config.SCALE,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": config.FLIPUD,
        "fliplr": config.FLIPLR,
        "mosaic": config.MOSAIC,
        "mixup": config.MIXUP,
        "copy_paste": config.COPY_PASTE,
    }

    print("\nTraining Configuration:")
    print(f"  Epochs      : {config.EPOCHS}")
    print(f"  Batch size  : {config.BATCH_SIZE}")
    print(f"  Image size  : {config.IMG_SIZE}")
    print(f"  Device      : {config.DEVICE}")
    print(f"  Workers     : {config.WORKERS}")
    print(f"  Optimizer   : AdamW, lr0={train_args['lr0']}")
    print(f"  IoU (train) : {train_args['iou']}")
    print(f"  Multi-scale : {train_args['multi_scale']}")
    print(f"  Mosaic      : {config.MOSAIC}")
    print(f"  Scale       : {config.SCALE}")
    print(f"  Degrees     : {config.DEGREES}")
    print(f"  Flips (LR/UD): {config.FLIPLR} / {config.FLIPUD}")
    print(f"  Copy-paste  : {config.COPY_PASTE}")
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

        if hasattr(results.box, "ap_class_index"):
            per_class_metrics = {}
            for idx, class_idx in enumerate(results.box.ap_class_index):
                name = config.CLASS_NAMES[int(class_idx)]
                per_class_metrics[f"test/{name}_mAP50"] = results.box.ap50[idx]
            wandb.log(per_class_metrics)
            print("✓ Per-class mAP50 logged to WandB.")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 for Wildlife Aerial Detection"
    )

    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--imgsz", type=int, help="Image size")
    parser.add_argument("--weights", type=str, help="Custom .pt weights to fine-tune")
    parser.add_argument("--model", type=str, help="Base model (yolo11s/m/l/x.pt)")
    parser.add_argument("--wandb-key", type=str, help="WandB API key")

    args = parser.parse_args()

    if args.wandb_key:
        wandb.login(key=args.wandb_key)

    config = Config()

    # CLI overrides
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch:
        config.BATCH_SIZE = args.batch
    if args.imgsz:
        config.IMG_SIZE = args.imgsz
    if args.model:
        config.MODEL = args.model

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
