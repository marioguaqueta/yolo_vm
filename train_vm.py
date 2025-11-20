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
    DATASET_ROOT = BASE_DIR.parent / "general_dataset"  # sibling directory
    
    # For Universidad de los Andes VM:
    # /home/estudiantes/grupo_12/sahariandataset/
    #   ├── yolo_vm/          <- Code here (this script)
    #   └── general_dataset/  <- Data here
    #       ├── train/        <- Training images
    #       ├── val/          <- Validation images
    #       └── test/         <- Test images
    
    # Dataset paths
    IMAGES_TRAIN = DATASET_ROOT / "train"
    IMAGES_VAL = DATASET_ROOT / "val"
    IMAGES_TEST = DATASET_ROOT / "test"
    
    # Class mapping
    CLASS_NAMES = {
        0: "Buffalo",
        1: "Elephant", 
        2: "Kudu",
        3: "Topi",
        4: "Warthog",
        5: "Waterbuck"
    }
    
    # Training hyperparameters
    MODEL = "yolo11s.pt"  # Starting model
    EPOCHS = 50
    BATCH_SIZE = 4 if torch.cuda.is_available() else 2  # Reduced for 2048px images
    IMG_SIZE = 2048  # High resolution for aerial images
    PATIENCE = 10  # Early stopping patience
    WORKERS = 8  # Data loading workers
    SAVE_PERIOD = 5  # Save checkpoint every 5 epochs
    
    # Memory optimization
    CACHE = False  # Don't cache images in RAM
    RECT = False  # Don't use rectangular training (saves memory)
    
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
    """Create or verify data.yaml configuration"""
    print("\n" + "="*70)
    print("CONFIGURING DATASET")
    print("="*70)
    
    # Create data.yaml in dataset root
    yaml_path = config.DATASET_ROOT / 'data.yaml'
    
    # Check if directories exist
    if not config.IMAGES_TRAIN.exists():
        raise FileNotFoundError(f"Training images not found at: {config.IMAGES_TRAIN}")
    if not config.IMAGES_VAL.exists():
        raise FileNotFoundError(f"Validation images not found at: {config.IMAGES_VAL}")
    
    # Create data.yaml
    data_yaml = {
        'path': str(config.DATASET_ROOT.absolute()),
        'train': 'train',  # Relative to 'path'
        'val': 'val',
        'test': 'test',
        'nc': len(config.CLASS_NAMES),
        'names': list(config.CLASS_NAMES.values())
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Dataset configured at: {config.DATASET_ROOT}")
    print(f"✓ data.yaml created at: {yaml_path}")
    print(f"\nDataset structure:")
    print(f"  Training images: {config.IMAGES_TRAIN}")
    print(f"  Validation images: {config.IMAGES_VAL}")
    print(f"  Test images: {config.IMAGES_TEST}")
    print(f"\nClasses ({len(config.CLASS_NAMES)}): {list(config.CLASS_NAMES.values())}")
    
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
    model = YOLO(config.MODEL)
    
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
        
        # Data augmentation
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
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


def validate_model(model, yaml_path, config):
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
    
    print(f"\nValidation Results:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")
    
    if hasattr(results.box, 'ap_class_index'):
        print(f"\nPer-class mAP50:")
        for idx, class_idx in enumerate(results.box.ap_class_index):
            class_name = config.CLASS_NAMES[int(class_idx)]
            map50 = results.box.ap50[idx]
            print(f"  {class_name:12s}: {map50:.4f}")
    
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
    parser.add_argument('--wandb-key', type=str, default=None, help='Wandb API key')
    
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
    print(f"  Classes: {list(config.CLASS_NAMES.values())}")
    
    # Verify dataset exists
    if not config.DATASET_ROOT.exists():
        print(f"\n❌ ERROR: Dataset not found at {config.DATASET_ROOT}")
        print(f"   Expected structure:")
        print(f"   {config.DATASET_ROOT.parent}/")
        print(f"   ├── yolo_vm/          (code)")
        print(f"   └── general_dataset/  (data)")
        return 1
    
    if not config.IMAGES_TRAIN.exists():
        print(f"\n❌ ERROR: Training images not found at {config.IMAGES_TRAIN}")
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
        val_results = validate_model(model, yaml_path, config)
        
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
