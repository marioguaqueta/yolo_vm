# 📁 Directory Structure Guide

## New Project Structure

The training code and dataset are now in **sibling directories** for better organization:

```
sahariandataset/                    # Parent directory
├── yolo_vm/                        # Code directory (THIS FOLDER)
│   ├── train_yolov11_wildlife.py
│   ├── train_vm.py
│   ├── test_setup.py
│   ├── setup_and_train.sh
│   ├── setup_conda_and_train.sh
│   ├── environment.yml
│   ├── requirements.txt
│   └── ... (all other scripts and docs)
│
└── general_dataset/                # Data directory (SIBLING)
    ├── train/                      # 928 training images
    ├── val/                        # 111 validation images
    ├── test/                       # 258 test images
    └── groundtruth/
        └── csv/
            ├── train_big_size_A_B_E_K_WH_WB.csv
            ├── val_big_size_A_B_E_K_WH_WB.csv
            └── test_big_size_A_B_E_K_WH_WB.csv
```

## Why This Structure?

✅ **Separation of code and data** - Cleaner organization  
✅ **Easy data sharing** - Dataset can be shared across projects  
✅ **Version control** - Code (yolo_vm) can be in git, data separate  
✅ **Portable** - Uses relative paths automatically  

## Path Configuration

### Automatic Relative Paths

All scripts now use **relative paths** automatically:

```python
BASE_DIR = Path(__file__).parent.absolute()  # yolo_vm/
DATASET_ROOT = BASE_DIR.parent / "general_dataset"  # ../general_dataset/
```

This works automatically on:
- ✅ Universidad de los Andes VM
- ✅ Local machines
- ✅ AWS/Azure/GCP VMs
- ✅ Google Colab (with adjusted paths)

### Your VM Paths

For the Universidad de los Andes VM:

```
/home/estudiantes/grupo_12/sahariandataset/
├── yolo_vm/          <- Your code here
└── general_dataset/  <- Your data here
```

**No path changes needed!** The scripts detect this automatically.

## Setup Instructions

### On VM (Universidad de los Andes)

1. **Navigate to code directory:**
   ```bash
   cd /home/estudiantes/grupo_12/sahariandataset/yolo_vm
   ```

2. **Create conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate yolov11-wildlife
   ```

3. **Verify paths are detected correctly:**
   ```bash
   python test_setup.py
   ```

4. **Start training:**
   ```bash
   python train_vm.py --epochs 50 --batch 8
   ```

### On Google Colab

Update `PROJECT_PATH` in `colab_setup.py`:

```python
# Point to your code directory (yolo_vm)
PROJECT_PATH = "/content/drive/MyDrive/MAIA_Final_Project_2025/yolo_vm"

# Dataset will be automatically found at:
# /content/drive/MyDrive/MAIA_Final_Project_2025/general_dataset
```

Google Drive structure:
```
MyDrive/
└── MAIA_Final_Project_2025/
    ├── yolo_vm/          <- Code
    └── general_dataset/  <- Data
```

### On Local Machine

If you're testing locally, maintain the same structure:

```bash
# Your working directory
/path/to/project/
├── yolo_vm/          <- Put code here
└── general_dataset/  <- Put data here

# Navigate to code directory
cd /path/to/project/yolo_vm

# Run training
python train_yolov11_wildlife.py
```

## Output Structure

Training outputs are created **inside the code directory** (yolo_vm):

```
yolo_vm/
├── runs/                          # Training results
│   └── yolov11_wildlife/
│       ├── weights/
│       │   ├── best.pt           # Best model
│       │   └── last.pt           # Last checkpoint
│       └── ... (plots, metrics)
│
└── yolo_wildlife_dataset/        # Converted YOLO format
    ├── train/
    ├── val/
    ├── test/
    └── data.yaml
```

## Verification

### Check Paths Are Correct

```bash
cd /home/estudiantes/grupo_12/sahariandataset/yolo_vm
python test_setup.py
```

You should see:
```
Detected paths:
  Code directory (BASE_DIR): /home/estudiantes/grupo_12/sahariandataset/yolo_vm
  Dataset directory: /home/estudiantes/grupo_12/sahariandataset/general_dataset

✓ Train Images: /home/estudiantes/grupo_12/sahariandataset/general_dataset/train
  928 images found
✓ Val Images: /home/estudiantes/grupo_12/sahariandataset/general_dataset/val
  111 images found
...
```

### Python Quick Test

```python
from pathlib import Path

# This script's location
script_dir = Path(__file__).parent.absolute()
print(f"Code directory: {script_dir}")

# Dataset location (sibling directory)
dataset_dir = script_dir.parent / "general_dataset"
print(f"Dataset directory: {dataset_dir}")
print(f"Dataset exists: {dataset_dir.exists()}")
```

## Common Issues

### Issue: "Dataset not found"

**Check 1:** Verify directory structure
```bash
ls -la /home/estudiantes/grupo_12/sahariandataset/
# Should show: yolo_vm/ and general_dataset/
```

**Check 2:** Verify you're in the code directory
```bash
pwd
# Should be: /home/estudiantes/grupo_12/sahariandataset/yolo_vm
```

**Check 3:** Run test_setup.py
```bash
python test_setup.py
```

### Issue: "Path not found" in scripts

**Solution:** Make sure you're running scripts from the `yolo_vm` directory:
```bash
cd /home/estudiantes/grupo_12/sahariandataset/yolo_vm
python train_vm.py
```

### Issue: Colab paths not working

**Solution:** Update PROJECT_PATH to point to `yolo_vm`:
```python
PROJECT_PATH = "/content/drive/MyDrive/MAIA_Final_Project_2025/yolo_vm"
```

## Migration from Old Structure

If you had the old structure (code and data in same folder):

**Old:**
```
Yolo/
├── general_dataset/
├── train_yolov11_wildlife.py
└── ...
```

**New:**
```
sahariandataset/
├── yolo_vm/
│   ├── train_yolov11_wildlife.py
│   └── ...
└── general_dataset/
```

**Migration steps:**
1. Create parent directory: `sahariandataset/`
2. Move code to: `sahariandataset/yolo_vm/`
3. Move data to: `sahariandataset/general_dataset/`
4. No code changes needed - paths are automatic!

## Summary

✅ **Code location:** `/home/estudiantes/grupo_12/sahariandataset/yolo_vm/`  
✅ **Data location:** `/home/estudiantes/grupo_12/sahariandataset/general_dataset/`  
✅ **Paths:** Automatic relative paths - no manual configuration needed  
✅ **Training:** Run from yolo_vm directory  
✅ **Outputs:** Saved in yolo_vm/runs/  

**Ready to train!** 🚀

```bash
cd /home/estudiantes/grupo_12/sahariandataset/yolo_vm
./setup_conda_and_train.sh
```

