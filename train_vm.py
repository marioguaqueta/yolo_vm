#!/usr/bin/env python3
"""
YOLOv11 Wildlife Detection Training Script - VM/Cloud Optimized Version
Project: Guacamaya - Microsoft AI for Good Lab
Description: Train YOLOv11 model for aerial wildlife detection with wandb integration
             Optimized for Google Colab, AWS, Azure, and other cloud environments
"""

import os
import yaml
import pandas as pd
import shutil
from pathlib import Path
from PIL import Image
import wandb
from ultralytics import YOLO
import torch
import argparse
import sys


# ============================================================================
# CONFIGURATION - CLOUD ENVIRONMENT
# ============================================================================

class VMConfig:
    """Training configuration for VM/Cloud environments"""
    
    # Detect environment
    IS_COLAB = 'google.colab' in sys.modules
    IS_KAGGLE = 'kaggle_secrets' in sys.modules
    
    # ===== MODIFY THESE PATHS FOR YOUR ENVIRONMENT =====
    if IS_COLAB:
        # Google Colab with Google Drive
        BASE_DIR = Path("/content/drive/MyDrive/MAIA_Final_Project_2025/yolo_vm")
        DATASET_ROOT = BASE_DIR.parent / "general_dataset"
    else:
        # Generic cloud VM or local path
        # Using relative paths: code in yolo_vm/, data in sibling general_dataset/
        BASE_DIR = Path(__file__).parent.absolute()  # yolo_vm directory
        DATASET_ROOT = BASE_DIR.parent / "general_dataset"  # sibling directory
        
        # For Universidad de los Andes VM, the structure is:
        # /home/estudiantes/grupo_12/sahariandataset/
        #   ├── yolo_vm/          <- Code here (this script)
        #   └── general_dataset/  <- Data here
    
    # Dataset paths (relative to DATASET_ROOT)
    IMAGES_TRAIN = DATASET_ROOT / "train"
    IMAGES_VAL = DATASET_ROOT / "val"
    IMAGES_TEST = DATASET_ROOT / "test"
    
    CSV_TRAIN = DATASET_ROOT / "groundtruth/csv/train_big_size_A_B_E_K_WH_WB.csv"
    CSV_VAL = DATASET_ROOT / "groundtruth/csv/val_big_size_A_B_E_K_WH_WB.csv"
    CSV_TEST = DATASET_ROOT / "groundtruth/csv/test_big_size_A_B_E_K_WH_WB.csv"
    
    # YOLO dataset output
    YOLO_DATASET = BASE_DIR / "yolo_wildlife_dataset"
    
    # Class mapping (from CSV labels to class names)
    CLASS_NAMES = {
        0: "Buffalo",
        1: "Elephant", 
        2: "Kudu",
        3: "Topi",
        4: "Warthog",
        5: "Waterbuck"
    }
    
    # Training hyperparameters - optimized for cloud GPU
    MODEL = "yolo11s.pt"  # Starting model
    EPOCHS = 50
    BATCH_SIZE = 8 if torch.cuda.is_available() else 2  # Auto-adjust for GPU/CPU
    IMG_SIZE = 2048  # High resolution for aerial images
    PATIENCE = 10  # Early stopping patience
    WORKERS = 8  # Data loading workers
    
    # Device configuration
    if torch.cuda.is_available():
        DEVICE = 0
        # Optimize for GPU
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = "cpu"
        print("⚠ WARNING: No GPU detected. Training will be very slow!")
    
    # Wandb configuration
    WANDB_PROJECT = "yolov11-wildlife-detection"
    WANDB_ENTITY = None  # Set to your wandb username if needed
    WANDB_RUN_NAME = f"yolo11s-vm-{IMG_SIZE}px"
    
    # Save checkpoints more frequently on cloud
    SAVE_PERIOD = 5  # Save every 5 epochs


def print_system_info():
    """Print system information for debugging"""
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    
    # Python
    print(f"Python: {sys.version}")
    
    # PyTorch
    print(f"PyTorch: {torch.__version__}")
    
    # GPU
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  GPU Count: {torch.cuda.device_count()}")
    else:
        print("⚠ GPU: Not available (CPU only)")
    
    # Environment
    if VMConfig.IS_COLAB:
        print("Environment: Google Colab")
    elif VMConfig.IS_KAGGLE:
        print("Environment: Kaggle")
    else:
        print("Environment: Generic VM/Server")
    
    # Disk space
    import shutil as sh
    total, used, free = sh.disk_usage("/")
    print(f"Disk: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")
    
    print("="*70)


# ============================================================================
# DATA PREPARATION
# ============================================================================

def create_yolo_directories(config):
    """Create YOLO dataset directory structure"""
    print("\n" + "="*70)
    print("CREATING YOLO DATASET STRUCTURE")
    print("="*70)
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        images_dir = config.YOLO_DATASET / split / 'images'
        labels_dir = config.YOLO_DATASET / split / 'labels'
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Created: {images_dir}")
        print(f"✓ Created: {labels_dir}")
    
    print(f"\n✓ YOLO dataset structure created at: {config.YOLO_DATASET}")


def csv_to_yolo_format(csv_path, images_dir, output_images_dir, output_labels_dir, config):
    """
    Convert CSV annotations to YOLO format
    Includes progress bar for long conversions
    """
    print(f"\nProcessing: {csv_path.name}")
    
    if not csv_path.exists():
        print(f"⚠ WARNING: CSV file not found: {csv_path}")
        return 0, 0
    
    df = pd.read_csv(csv_path)
    print(f"  Total annotations: {len(df)}")
    
    # Group by image
    grouped = df.groupby('Image')
    images_processed = 0
    annotations_written = 0
    
    from tqdm import tqdm
    
    for image_name, group in tqdm(grouped, desc="Converting images", unit="img"):
        image_path = images_dir / image_name
        
        if not image_path.exists():
            # Try with different extensions
            base_name = image_path.stem
            found = False
            for ext in ['.JPG', '.jpg', '.png', '.jpeg']:
                alt_path = images_dir / f"{base_name}{ext}"
                if alt_path.exists():
                    image_path = alt_path
                    image_name = alt_path.name
                    found = True
                    break
            
            if not found:
                continue
        
        # Get image dimensions
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception:
            continue
        
        # Copy image to YOLO dataset
        dest_image_path = output_images_dir / image_name
        if not dest_image_path.exists():
            shutil.copy(image_path, dest_image_path)
        
        # Create YOLO label file
        label_file = output_labels_dir / f"{Path(image_name).stem}.txt"
        
        with open(label_file, 'w') as f:
            for _, row in group.iterrows():
                class_id = int(row['Label'])
                x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
                
                # Convert to YOLO format
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # Clamp values
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                annotations_written += 1
        
        images_processed += 1
    
    print(f"  ✓ Images processed: {images_processed}")
    print(f"  ✓ Annotations written: {annotations_written}")
    
    return images_processed, annotations_written


def prepare_dataset(config):
    """Prepare complete YOLO dataset from CSV annotations"""
    print("\n" + "="*70)
    print("CONVERTING CSV ANNOTATIONS TO YOLO FORMAT")
    print("="*70)
    
    create_yolo_directories(config)
    
    total_images = 0
    total_annotations = 0
    
    splits = [
        ('train', config.CSV_TRAIN, config.IMAGES_TRAIN),
        ('val', config.CSV_VAL, config.IMAGES_VAL),
        ('test', config.CSV_TEST, config.IMAGES_TEST)
    ]
    
    for split_name, csv_path, images_dir in splits:
        output_images_dir = config.YOLO_DATASET / split_name / 'images'
        output_labels_dir = config.YOLO_DATASET / split_name / 'labels'
        
        images, annotations = csv_to_yolo_format(
            csv_path, images_dir, output_images_dir, output_labels_dir, config
        )
        
        total_images += images
        total_annotations += annotations
    
    print(f"\n" + "="*70)
    print(f"DATASET CONVERSION COMPLETE")
    print(f"  Total images: {total_images}")
    print(f"  Total annotations: {total_annotations}")
    print("="*70)
    
    return total_images > 0


def create_data_yaml(config):
    """Create YAML configuration file for YOLO training"""
    print("\n" + "="*70)
    print("CREATING DATA.YAML CONFIGURATION")
    print("="*70)
    
    data_yaml = {
        'path': str(config.YOLO_DATASET.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(config.CLASS_NAMES),
        'names': list(config.CLASS_NAMES.values())
    }
    
    yaml_path = config.YOLO_DATASET / 'data.yaml'
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ data.yaml created at: {yaml_path}")
    print(f"\nConfiguration:")
    print(f"  Path: {data_yaml['path']}")
    print(f"  Classes: {data_yaml['nc']}")
    print(f"  Names: {data_yaml['names']}")
    
    return yaml_path


# ============================================================================
# TRAINING
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
        # Try to login (will use existing credentials or prompt)
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
                    "environment": "Colab" if config.IS_COLAB else "VM",
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


def train_model(config, yaml_path, use_wandb=True):
    """Train YOLOv11 model on VM/Cloud"""
    print("\n" + "="*70)
    print("STARTING YOLOV11 TRAINING")
    print("="*70)
    
    # Initialize model
    print(f"\nLoading model: {config.MODEL}")
    model = YOLO(config.MODEL)
    
    # Training parameters optimized for cloud
    train_args = {
        'data': str(yaml_path),
        'epochs': config.EPOCHS,
        'imgsz': config.IMG_SIZE,
        'batch': config.BATCH_SIZE,
        'device': config.DEVICE,
        'patience': config.PATIENCE,
        'project': str(config.BASE_DIR / 'runs'),
        'name': 'yolov11_wildlife_vm',
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
        'amp': True,  # Automatic Mixed Precision for faster training
        
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
    print(f"  Mixed Precision: Enabled")
    print(f"  Wandb: {'Enabled' if use_wandb else 'Disabled'}")
    
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
    """Main training pipeline for VM/Cloud"""
    parser = argparse.ArgumentParser(description='Train YOLOv11 for Wildlife Detection (VM/Cloud)')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=None, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=None, help='Image size')
    parser.add_argument('--skip-conversion', action='store_true', 
                       help='Skip dataset conversion')
    parser.add_argument('--wandb-key', type=str, default=None,
                       help='Wandb API key (for automated setup)')
    
    args = parser.parse_args()
    
    # Login to wandb if key provided
    if args.wandb_key:
        wandb.login(key=args.wandb_key)
    
    # Initialize configuration
    config = VMConfig()
    
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
    print("YOLOv11 WILDLIFE DETECTION TRAINING (VM/Cloud)")
    print("Project: Guacamaya - Microsoft AI for Good Lab")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Dataset: {config.DATASET_ROOT}")
    print(f"  Output: {config.YOLO_DATASET}")
    print(f"  Model: {config.MODEL}")
    print(f"  Classes: {list(config.CLASS_NAMES.values())}")
    
    # Verify dataset exists
    if not config.DATASET_ROOT.exists():
        print(f"\n❌ ERROR: Dataset not found at {config.DATASET_ROOT}")
        print("   Please update BASE_DIR in VMConfig class to point to your dataset")
        return 1
    
    # Step 1: Prepare dataset
    if not args.skip_conversion:
        success = prepare_dataset(config)
        if not success:
            print("\n❌ ERROR: Dataset preparation failed!")
            return 1
    else:
        print("\n⏭  Skipping dataset conversion")
    
    # Step 2: Create data.yaml
    yaml_path = create_data_yaml(config)
    
    # Step 3: Initialize wandb
    wandb_run = initialize_wandb(config, enabled=not args.no_wandb)
    
    # Step 4: Train model
    try:
        model, train_results = train_model(config, yaml_path, use_wandb=(wandb_run is not None))
        
        # Step 5: Validate model
        val_results = validate_model(model, yaml_path, config)
        
        # Step 6: Save final information
        best_model_path = config.BASE_DIR / 'runs' / 'yolov11_wildlife_vm' / 'weights' / 'best.pt'
        
        print("\n" + "="*70)
        print("TRAINING PIPELINE COMPLETE! ✓")
        print("="*70)
        print(f"\nBest model: {best_model_path}")
        print(f"Results: {config.BASE_DIR / 'runs' / 'yolov11_wildlife_vm'}")
        
        if wandb_run:
            print(f"Wandb: {wandb_run.get_url()}")
            wandb.finish()
        
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

