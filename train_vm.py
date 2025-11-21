#!/usr/bin/env python3
"""
YOLOv11 Wildlife Detection Training Script - Simplified VM Version
Project: Guacamaya - Microsoft AI for Good Lab
Description: Train YOLOv11 model for aerial wildlife detection with wandb integration
"""

import os
import yaml
from pathlib import Path
import wandb
from ultralytics import YOLO
import torch
import argparse
import sys


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration"""
    
    # Paths - Using relative paths
    # Code structure: parent_dir/yolo_vm/ (code) and parent_dir/general_dataset/ (data)
    BASE_DIR = Path(__file__).parent.absolute()  # yolo_vm directory
    print(f"BASE_DIR: {BASE_DIR}")
    DATASET_ROOT = BASE_DIR.parent / "general_dataset"  # sibling directory
    print(f"DATASET_ROOT: {DATASET_ROOT}")
    
    # For Universidad de los Andes VM:
    # /home/estudiantes/grupo_12/sahariandataset/
    #   ├── yolo_vm/          <- Code here (this script)
    #   └── general_dataset/  <- Data here (YOLO standard structure)
    #       ├── images/
    #       │   ├── train/    <- Training images (.JPG)
    #       │   ├── val/      <- Validation images
    #       │   └── test/     <- Test images
    #       └── labels/
    #           ├── train/    <- Training labels (.txt)
    #           ├── val/      <- Validation labels
    #           └── test/     <- Test labels
    
    # Dataset paths - Standard YOLO structure
    IMAGES_TRAIN = DATASET_ROOT / "images" / "train"
    print(f"IMAGES_TRAIN: {IMAGES_TRAIN}")
    IMAGES_VAL = DATASET_ROOT / "images" / "val"
    print(f"IMAGES_VAL: {IMAGES_VAL}")
    IMAGES_TEST = DATASET_ROOT / "images" / "test"
    print(f"IMAGES_TEST: {IMAGES_TEST}")
    
    # Labels paths
    LABELS_TRAIN = DATASET_ROOT / "labels" / "train"
    LABELS_VAL = DATASET_ROOT / "labels" / "val"
    LABELS_TEST = DATASET_ROOT / "labels" / "test"
    
    # Class mapping (MUST be 0-indexed for YOLO)
    CLASS_NAMES = {
        0: "Buffalo",
        1: "Elephant", 
        2: "Kudu",
        3: "Topi",
        4: "Warthog",
        5: "Waterbuck"
    }
    
    # Training hyperparameters
    MODEL = "yolo11m.pt"  # Starting model (use yolo11s/m/l/x.pt for regular bounding boxes)
    EPOCHS = 50
    BATCH_SIZE = 4 if torch.cuda.is_available() else 2  # Reduced for 2048px images
    IMG_SIZE = 2048  # High resolution for aerial images
    PATIENCE = 10  # Early stopping patience
    WORKERS = 8  # Data loading workers
    SAVE_PERIOD = 5  # Save checkpoint every 5 epochs
    
    # Memory optimization
    CACHE = False  # Don't cache images in RAM
    RECT = False  # Don't use rectangular training (saves memory)
    
    # Data augmentation parameters
    DEGREES = 0.0      # Rotation (0.0 = disabled, try 10.0 for ±10°)
    SCALE = 0.5        # Scale/zoom (0.0-1.0, 0.5 = up to 50% zoom)
    TRANSLATE = 0.1    # Translation (0.0-1.0)
    HSV_H = 0.015      # Hue augmentation (0.0-1.0)
    HSV_S = 0.7        # Saturation augmentation (0.0-1.0)
    HSV_V = 0.4        # Value/brightness augmentation (0.0-1.0)
    FLIPLR = 0.5       # Horizontal flip probability (0.0-1.0)
    FLIPUD = 0.0       # Vertical flip (0.0 for aerial images)
    MOSAIC = 1.0       # Mosaic augmentation (0.0-1.0)
    
    # Device configuration
    if torch.cuda.is_available():
        DEVICE = 0
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = "cpu"
        print("⚠ WARNING: No GPU detected. Training will be very slow!")
    
    # Wandb configuration
    WANDB_PROJECT = "yolov11-wildlife-detection"
    WANDB_ENTITY = None  # Set to your wandb username if needed
    WANDB_RUN_NAME = f"yolo11s-wildlife-{IMG_SIZE}px"


def print_system_info():
    """Print system information"""
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ GPU: Not available (CPU only)")
    
    import shutil
    total, used, free = shutil.disk_usage("/")
    print(f"Disk: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")
    
    print("="*70)


# ============================================================================
# DATA CONFIGURATION
# ============================================================================

def create_data_yaml(config):
    """Create data.yaml configuration for YOLO standard structure"""
    print("\n" + "="*70)
    print("CONFIGURING DATASET")
    print("="*70)
    
    # Create data.yaml in dataset root
    yaml_path = config.DATASET_ROOT / 'data.yaml'
    
    # YOLO standard structure
    print("✓ Using standard YOLO structure")
    data_yaml = {
        'path': str(config.DATASET_ROOT.absolute()),
        'train': 'images/train',  # Relative to 'path'
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(config.CLASS_NAMES),
        'names': list(config.CLASS_NAMES.values())
    }
    
    # Write data.yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ data.yaml created at: {yaml_path}")
    print(f"\nConfiguration:")
    print(f"  Path: {data_yaml['path']}")
    print(f"  Train: {data_yaml['train']}")
    print(f"  Val: {data_yaml['val']}")
    print(f"  Test: {data_yaml['test']}")
    print(f"  Classes: {len(config.CLASS_NAMES)}")
    print(f"  Names: {list(config.CLASS_NAMES.values())}")
    
    # Count files for verification
    train_images = list(config.IMAGES_TRAIN.glob('*.JPG')) + list(config.IMAGES_TRAIN.glob('*.jpg'))
    val_images = list(config.IMAGES_VAL.glob('*.JPG')) + list(config.IMAGES_VAL.glob('*.jpg'))
    train_labels = list(config.LABELS_TRAIN.glob('*.txt'))
    val_labels = list(config.LABELS_VAL.glob('*.txt'))
    
    print(f"\nDataset Statistics:")
    print(f"  Training images: {len(train_images)}")
    print(f"  Training labels: {len(train_labels)}")
    print(f"  Validation images: {len(val_images)}")
    print(f"  Validation labels: {len(val_labels)}")
    
    if len(train_images) != len(train_labels):
        print(f"  ⚠ WARNING: Mismatch between images and labels in training set")
    if len(val_images) != len(val_labels):
        print(f"  ⚠ WARNING: Mismatch between images and labels in validation set")
    
    return yaml_path


# ============================================================================
# WANDB INTEGRATION
# ============================================================================

def initialize_wandb(config, enabled=True):
    """Initialize Weights & Biases for experiment tracking"""
    if not enabled:
        print("\n⚠ Wandb tracking disabled")
        return None
    
    print("\n" + "="*70)
    print("INITIALIZING WEIGHTS & BIASES")
    print("="*70)
    
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
                }
            )
            
            print(f"✓ Wandb initialized")
            print(f"  Project: {config.WANDB_PROJECT}")
            print(f"  Run name: {config.WANDB_RUN_NAME}")
            print(f"  Dashboard: {run.get_url()}")
            
            return run
    except Exception as e:
        print(f"⚠ WARNING: Could not initialize wandb: {e}")
        print("  Training will continue without wandb logging")
    
    return None


# ============================================================================
# TRAINING
# ============================================================================

def on_fit_epoch_end(trainer):
    """Callback to log custom metrics (F1 score) after each epoch"""
    if wandb.run is not None:
        # Get metrics from trainer
        metrics = trainer.metrics
        
        # Calculate F1 score from precision and recall
        if 'metrics/precision(B)' in metrics and 'metrics/recall(B)' in metrics:
            precision = metrics['metrics/precision(B)']
            recall = metrics['metrics/recall(B)']
            
            if (precision + recall) > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
                
                # Log F1 score to Wandb
                wandb.log({
                    'metrics/f1_score': f1_score,
                    'epoch': trainer.epoch
                })


def train_model(config, yaml_path, use_wandb=True):
    """Train YOLOv11 model"""
    print("\n" + "="*70)
    print("STARTING YOLOV11 TRAINING")
    print("="*70)
    
    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ GPU cache cleared")
    
    # Initialize model
    print(f"\nLoading model: {config.MODEL}")
    if str(config.MODEL).endswith('best.pt') or str(config.MODEL).endswith('last.pt'):
        print(f"  → Fine-tuning from custom checkpoint")
    model = YOLO(config.MODEL)
    
    # Add custom callback for F1 score logging
    if use_wandb:
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        print("✓ Custom F1 score logging callback added")
    
    # Training parameters
    train_args = {
        'data': str(yaml_path),
        'epochs': config.EPOCHS,
        'imgsz': config.IMG_SIZE,
        'batch': config.BATCH_SIZE,
        'device': config.DEVICE,
        'patience': config.PATIENCE,
        'project': str(config.BASE_DIR / 'runs'),
        'name': 'yolov11_wildlife',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'plots': True,
        'save': True,
        'save_period': config.SAVE_PERIOD,
        'workers': config.WORKERS,
        'amp': True,  # Automatic Mixed Precision
        'cache': config.CACHE,  # Memory optimization
        'rect': config.RECT,  # Memory optimization
        
        # Data augmentation (from config, customizable via CLI)
        'hsv_h': config.HSV_H,
        'hsv_s': config.HSV_S,
        'hsv_v': config.HSV_V,
        'degrees': config.DEGREES,
        'translate': config.TRANSLATE,
        'scale': config.SCALE,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': config.FLIPUD,
        'fliplr': config.FLIPLR,
        'mosaic': config.MOSAIC,
        'mixup': 0.0,
        'copy_paste': 0.0,
    }
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Image size: {config.IMG_SIZE}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Workers: {config.WORKERS}")
    print(f"  Save period: Every {config.SAVE_PERIOD} epochs")
    print(f"  Mixed Precision: Enabled")
    print(f"  Memory optimization: Cache={config.CACHE}, Rect={config.RECT}")
    print(f"  Wandb: {'Enabled' if use_wandb else 'Disabled'}")
    
    print(f"\nData Augmentation:")
    print(f"  Rotation: ±{config.DEGREES}° {'(disabled)' if config.DEGREES == 0 else ''}")
    print(f"  Scale/Zoom: {config.SCALE * 100:.0f}%")
    print(f"  Translation: {config.TRANSLATE * 100:.0f}%")
    print(f"  HSV (H/S/V): {config.HSV_H:.3f} / {config.HSV_S:.3f} / {config.HSV_V:.3f}")
    print(f"  Horizontal Flip: {config.FLIPLR * 100:.0f}%")
    print(f"  Mosaic: {config.MOSAIC * 100:.0f}%")
    
    # Memory warning
    if config.IMG_SIZE >= 2048 and config.BATCH_SIZE > 4:
        print(f"\n⚠️  WARNING: Large image size ({config.IMG_SIZE}px) with batch size {config.BATCH_SIZE}")
        print(f"   This may cause GPU memory issues. Consider:")
        print(f"   - Reducing batch size: --batch 2 or --batch 4")
        print(f"   - Reducing image size: --imgsz 1024")
    
    print(f"\n{'='*70}")
    print("TRAINING IN PROGRESS...")
    print(f"{'='*70}\n")
    
    # Train the model
    results = model.train(**train_args)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    
    return model, results


def validate_model(model, yaml_path, config, log_to_wandb=False):
    """Validate the trained model"""
    print("\n" + "="*70)
    print("VALIDATING MODEL")
    print("="*70)
    
    print("\nRunning validation on test set...")
    results = model.val(
        data=str(yaml_path),
        split='test',
        imgsz=config.IMG_SIZE,
        batch=config.BATCH_SIZE,
        device=config.DEVICE,
        plots=True
    )
    
    # Calculate F1 score
    precision = results.box.mp
    recall = results.box.mr
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nValidation Results:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1_score:.4f}")  # Now displayed!
    
    # Log custom metrics to Wandb
    if log_to_wandb and wandb.run is not None:
        wandb.log({
            'test/mAP50': results.box.map50,
            'test/mAP50-95': results.box.map,
            'test/precision': precision,
            'test/recall': recall,
            'test/f1_score': f1_score,  # Custom F1 score!
        })
        print(f"\n✓ Test metrics logged to Wandb")
    
    if hasattr(results.box, 'ap_class_index'):
        print(f"\nPer-class mAP50:")
        per_class_metrics = {}
        for idx, class_idx in enumerate(results.box.ap_class_index):
            class_name = config.CLASS_NAMES[int(class_idx)]
            map50 = results.box.ap50[idx]
            print(f"  {class_name:12s}: {map50:.4f}")
            
            # Log per-class metrics to Wandb
            if log_to_wandb and wandb.run is not None:
                per_class_metrics[f'test/{class_name}_mAP50'] = map50
        
        if per_class_metrics and wandb.run is not None:
            wandb.log(per_class_metrics)
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train YOLOv11 for Wildlife Detection')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=None, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=None, help='Image size')
    parser.add_argument('--weights', type=str, default=None, 
                        help='Path to pretrained model weights (e.g., best.pt from previous training)')
    parser.add_argument('--wandb-key', type=str, default=None, help='Wandb API key')
    
    # Data augmentation options
    parser.add_argument('--rotate', type=float, default=None, 
                        help='Image rotation augmentation in degrees (e.g., 10.0 for ±10°)')
    parser.add_argument('--scale', type=float, default=None, 
                        help='Image scale/zoom augmentation (0.0-1.0, default: 0.5)')
    parser.add_argument('--hsv-h', type=float, default=None, 
                        help='HSV Hue augmentation (0.0-1.0, default: 0.015)')
    parser.add_argument('--hsv-s', type=float, default=None, 
                        help='HSV Saturation augmentation (0.0-1.0, default: 0.7)')
    parser.add_argument('--hsv-v', type=float, default=None, 
                        help='HSV Value augmentation (0.0-1.0, default: 0.4)')
    parser.add_argument('--translate', type=float, default=None, 
                        help='Translation augmentation (0.0-1.0, default: 0.1)')
    parser.add_argument('--fliplr', type=float, default=None, 
                        help='Horizontal flip probability (0.0-1.0, default: 0.5)')
    parser.add_argument('--mosaic', type=float, default=None, 
                        help='Mosaic augmentation probability (0.0-1.0, default: 1.0)')
    
    args = parser.parse_args()
    
    # Login to wandb if key provided
    if args.wandb_key:
        wandb.login(key=args.wandb_key)
    
    # Initialize configuration
    config = Config()
    
    # Override with CLI args
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch:
        config.BATCH_SIZE = args.batch
    if args.imgsz:
        config.IMG_SIZE = args.imgsz
    
    # Override augmentation parameters
    if args.rotate is not None:
        config.DEGREES = args.rotate
    if args.scale is not None:
        config.SCALE = args.scale
    if args.hsv_h is not None:
        config.HSV_H = args.hsv_h
    if args.hsv_s is not None:
        config.HSV_S = args.hsv_s
    if args.hsv_v is not None:
        config.HSV_V = args.hsv_v
    if args.translate is not None:
        config.TRANSLATE = args.translate
    if args.fliplr is not None:
        config.FLIPLR = args.fliplr
    if args.mosaic is not None:
        config.MOSAIC = args.mosaic
    
    # Handle custom weights path
    if args.weights:
        weights_path = Path(args.weights)
        if not weights_path.exists():
            print(f"\n❌ ERROR: Weights file not found at {weights_path}")
            print(f"   Please provide a valid path to a .pt file")
            print(f"\n   Example:")
            print(f"   python train_vm.py --weights runs/yolov11_wildlife/weights/best.pt")
            return 1
        config.MODEL = str(weights_path.absolute())
        print(f"\n✓ Using custom pretrained weights: {weights_path}")
    
    # Print system info
    print_system_info()
    
    print("\n" + "="*70)
    print("YOLOv11 WILDLIFE DETECTION TRAINING")
    print("Project: Guacamaya - Microsoft AI for Good Lab")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Code directory: {config.BASE_DIR}")
    print(f"  Dataset directory: {config.DATASET_ROOT}")
    print(f"  Model: {config.MODEL}")
    if args.weights:
        print(f"  Training mode: Fine-tuning from custom weights")
    else:
        print(f"  Training mode: Starting from base model")
    print(f"  Classes: {list(config.CLASS_NAMES.values())}")
    
    # Verify dataset exists
    if not config.DATASET_ROOT.exists():
        print(f"\n❌ ERROR: Dataset not found at {config.DATASET_ROOT}")
        print(f"   Expected structure:")
        print(f"   {config.DATASET_ROOT.parent}/")
        print(f"   ├── yolo_vm/          (code)")
        print(f"   └── general_dataset/  (data)")
        return 1
    
    # Check for YOLO standard structure
    if not config.IMAGES_TRAIN.exists():
        print(f"\n❌ ERROR: Training images not found at {config.IMAGES_TRAIN}")
        print(f"   This script expects YOLO standard structure:")
        print(f"   {config.DATASET_ROOT}/")
        print(f"   ├── images/")
        print(f"   │   ├── train/")
        print(f"   │   ├── val/")
        print(f"   │   └── test/")
        print(f"   └── labels/")
        print(f"       ├── train/")
        print(f"       ├── val/")
        print(f"       └── test/")
        print(f"\n   SOLUTION:")
        print(f"   1. First run: python convert_csv_to_yolo.py")
        print(f"   2. Then run:  python reorganize_to_yolo_structure.py")
        print(f"   3. Then run:  python train_vm.py")
        return 1
    
    if not config.LABELS_TRAIN.exists():
        print(f"\n❌ ERROR: Training labels not found at {config.LABELS_TRAIN}")
        print(f"   SOLUTION:")
        print(f"   1. First run: python convert_csv_to_yolo.py")
        print(f"   2. Then run:  python reorganize_to_yolo_structure.py")
        print(f"   3. Then run:  python train_vm.py")
        return 1
    
    # Create data.yaml
    try:
        yaml_path = create_data_yaml(config)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to configure dataset: {e}")
        return 1
    
    # Initialize wandb
    wandb_run = initialize_wandb(config, enabled=not args.no_wandb)
    
    # Train model
    try:
        model, train_results = train_model(config, yaml_path, use_wandb=(wandb_run is not None))
        
        # Validate model
        val_results = validate_model(model, yaml_path, config, log_to_wandb=(wandb_run is not None))
        
        # Print final information
        best_model_path = config.BASE_DIR / 'runs' / 'yolov11_wildlife' / 'weights' / 'best.pt'
        last_model_path = config.BASE_DIR / 'runs' / 'yolov11_wildlife' / 'weights' / 'last.pt'
        
        print("\n" + "="*70)
        print("TRAINING PIPELINE COMPLETE! ✓")
        print("="*70)
        print(f"\nModel Checkpoints:")
        print(f"  Best model: {best_model_path}")
        print(f"  Last model: {last_model_path}")
        print(f"\nResults directory: {config.BASE_DIR / 'runs' / 'yolov11_wildlife'}")
        
        if wandb_run:
            print(f"\nWandb Dashboard: {wandb_run.get_url()}")
            wandb.finish()
        
        print("\n" + "="*70)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        if wandb_run:
            wandb.finish(exit_code=1)
        
        return 1


if __name__ == "__main__":
    exit(main())
