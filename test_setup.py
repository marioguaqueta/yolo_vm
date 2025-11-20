#!/usr/bin/env python3
"""
Test Setup Script for YOLOv11 Wildlife Detection
Run this before training to verify everything is configured correctly
"""

import sys
from pathlib import Path
import importlib.util

def print_header(text):
    """Print a formatted header"""
    print(f"\n{'='*70}")
    print(f"{text}")
    print(f"{'='*70}")

def check_mark(condition):
    """Return checkmark or X based on condition"""
    return "✓" if condition else "✗"

def test_python_version():
    """Test Python version"""
    print_header("1. Python Version")
    version = sys.version_info
    is_ok = version.major == 3 and version.minor >= 8
    print(f"{check_mark(is_ok)} Python {version.major}.{version.minor}.{version.micro}")
    if not is_ok:
        print("  ⚠ WARNING: Python 3.8+ is recommended")
    return is_ok

def test_dependencies():
    """Test required dependencies"""
    print_header("2. Dependencies")
    
    required_packages = {
        'torch': 'PyTorch',
        'ultralytics': 'Ultralytics YOLO',
        'pandas': 'Pandas',
        'PIL': 'Pillow',
        'yaml': 'PyYAML',
        'wandb': 'Weights & Biases',
        'tqdm': 'tqdm'
    }
    
    all_ok = True
    for package, name in required_packages.items():
        try:
            spec = importlib.util.find_spec(package)
            if spec is not None:
                print(f"✓ {name}")
            else:
                print(f"✗ {name} - Not found")
                all_ok = False
        except Exception:
            print(f"✗ {name} - Not found")
            all_ok = False
    
    if not all_ok:
        print("\n  Install missing packages with:")
        print("  pip install -r requirements.txt")
    
    return all_ok

def test_gpu():
    """Test GPU availability"""
    print_header("3. GPU Configuration")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        print(f"{check_mark(cuda_available)} CUDA Available: {cuda_available}")
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            print(f"✓ GPU Count: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            print(f"✓ CUDA Version: {torch.version.cuda}")
            print(f"✓ cuDNN Version: {torch.backends.cudnn.version()}")
            
            return True
        else:
            print("  ⚠ WARNING: No GPU detected. Training will be very slow on CPU.")
            print("  Consider using Google Colab or a cloud GPU instance.")
            return False
            
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False

def test_dataset_structure():
    """Test dataset structure"""
    print_header("4. Dataset Structure")
    
    # Try to import the config
    try:
        # Add current directory to path
        sys.path.insert(0, str(Path(__file__).parent))
        from train_yolov11_wildlife import Config
        config = Config()
        
        print(f"Base Directory: {config.BASE_DIR}")
        print(f"Dataset Root: {config.DATASET_ROOT}")
        
        # Check directories
        required_dirs = [
            ("Train Images", config.IMAGES_TRAIN),
            ("Val Images", config.IMAGES_VAL),
            ("Test Images", config.IMAGES_TEST),
        ]
        
        required_files = [
            ("Train CSV", config.CSV_TRAIN),
            ("Val CSV", config.CSV_VAL),
            ("Test CSV", config.CSV_TEST),
        ]
        
        all_ok = True
        
        print("\nDirectories:")
        for name, path in required_dirs:
            exists = path.exists()
            print(f"{check_mark(exists)} {name}: {path}")
            if not exists:
                all_ok = False
            else:
                # Count images
                image_count = len(list(path.glob("*.JPG"))) + len(list(path.glob("*.jpg")))
                print(f"    {image_count} images found")
        
        print("\nAnnotation Files:")
        for name, path in required_files:
            exists = path.exists()
            print(f"{check_mark(exists)} {name}: {path}")
            if not exists:
                all_ok = False
            else:
                # Read CSV and count annotations
                try:
                    import pandas as pd
                    df = pd.read_csv(path)
                    print(f"    {len(df)} annotations")
                except Exception as e:
                    print(f"    ⚠ Could not read CSV: {e}")
        
        print(f"\nClasses: {list(config.CLASS_NAMES.values())}")
        
        if not all_ok:
            print("\n  ⚠ WARNING: Some dataset files/folders are missing!")
            print(f"  Please ensure your dataset is at: {config.DATASET_ROOT}")
        
        return all_ok
        
    except Exception as e:
        print(f"✗ Error checking dataset: {e}")
        print(f"  Make sure train_yolov11_wildlife.py is in the same directory")
        return False

def test_wandb():
    """Test Wandb configuration"""
    print_header("5. Weights & Biases")
    
    try:
        import wandb
        
        # Check if logged in
        try:
            api = wandb.Api()
            user = api.viewer
            print(f"✓ Logged in as: {user.username if hasattr(user, 'username') else 'unknown'}")
            return True
        except Exception:
            print("✗ Not logged in to wandb")
            print("  Run 'wandb login' to setup")
            print("  Or use --no-wandb flag to train without logging")
            return False
            
    except Exception as e:
        print(f"✗ Error checking wandb: {e}")
        return False

def test_disk_space():
    """Test available disk space"""
    print_header("6. Disk Space")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        
        total_gb = total / 1e9
        used_gb = used / 1e9
        free_gb = free / 1e9
        
        print(f"Total: {total_gb:.1f} GB")
        print(f"Used: {used_gb:.1f} GB")
        print(f"Free: {free_gb:.1f} GB")
        
        # Need at least 20GB free for training
        is_ok = free_gb > 20
        print(f"\n{check_mark(is_ok)} Sufficient space: {is_ok}")
        
        if not is_ok:
            print("  ⚠ WARNING: Low disk space. Training may fail.")
            print("  Recommended: 20GB+ free space")
        
        return is_ok
        
    except Exception as e:
        print(f"✗ Error checking disk space: {e}")
        return False

def test_write_permissions():
    """Test write permissions"""
    print_header("7. Write Permissions")
    
    try:
        # Try to create a test file
        test_file = Path("test_write_permissions.tmp")
        test_file.write_text("test")
        test_file.unlink()
        
        print("✓ Write permissions OK")
        return True
        
    except Exception as e:
        print(f"✗ Cannot write to current directory: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("YOLOv11 Wildlife Detection - Setup Verification")
    print("="*70)
    
    results = {
        "Python Version": test_python_version(),
        "Dependencies": test_dependencies(),
        "GPU": test_gpu(),
        "Dataset": test_dataset_structure(),
        "Wandb": test_wandb(),
        "Disk Space": test_disk_space(),
        "Write Permissions": test_write_permissions()
    }
    
    # Summary
    print_header("Test Summary")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nYou're ready to start training!")
        print("\nRun:")
        print("  python train_yolov11_wildlife.py")
        print("\nOr:")
        print("  ./setup_and_train.sh")
    else:
        print("⚠ SOME TESTS FAILED")
        print("\nPlease fix the issues above before training.")
        print("Some issues (like wandb or GPU) are warnings and won't prevent training.")
        print("\nCritical issues to fix:")
        if not results["Dependencies"]:
            print("  - Install dependencies: pip install -r requirements.txt")
        if not results["Dataset"]:
            print("  - Verify dataset paths in train_yolov11_wildlife.py")
        if not results["Write Permissions"]:
            print("  - Check folder permissions")
    
    print("="*70 + "\n")
    
    return all_passed

if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

