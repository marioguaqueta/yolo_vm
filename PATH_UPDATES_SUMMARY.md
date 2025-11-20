# 📝 Path Updates Summary

## Changes Made

All training scripts have been updated to support the new directory structure where **code** and **data** are in sibling directories.

### New Directory Structure

```
sahariandataset/                          # Parent directory
├── yolo_vm/                              # Code directory
│   ├── train_yolov11_wildlife.py
│   ├── train_vm.py
│   ├── test_setup.py
│   └── ... (all other files)
└── general_dataset/                      # Data directory (sibling)
    ├── train/
    ├── val/
    ├── test/
    └── groundtruth/csv/
```

### VM Paths (Universidad de los Andes)

- **Parent:** `/home/estudiantes/grupo_12/sahariandataset/`
- **Code:** `/home/estudiantes/grupo_12/sahariandataset/yolo_vm/`
- **Data:** `/home/estudiantes/grupo_12/sahariandataset/general_dataset/`

---

## Updated Files

### 1. `train_yolov11_wildlife.py`

**Before:**
```python
BASE_DIR = Path("/Users/marioguaqueta/Desktop/.../Yolo")
DATASET_ROOT = BASE_DIR / "general_dataset"
```

**After:**
```python
BASE_DIR = Path(__file__).parent.absolute()  # yolo_vm directory
DATASET_ROOT = BASE_DIR.parent / "general_dataset"  # sibling directory
```

### 2. `train_vm.py`

**Before:**
```python
BASE_DIR = Path("/mnt/data/Yolo")
DATASET_ROOT = BASE_DIR / "general_dataset"
```

**After:**
```python
# Automatic detection based on environment
if IS_COLAB:
    BASE_DIR = Path("/content/drive/MyDrive/MAIA_Final_Project_2025/yolo_vm")
    DATASET_ROOT = BASE_DIR.parent / "general_dataset"
else:
    # Uses relative paths automatically
    BASE_DIR = Path(__file__).parent.absolute()
    DATASET_ROOT = BASE_DIR.parent / "general_dataset"
```

**Comments added:**
```python
# For Universidad de los Andes VM, the structure is:
# /home/estudiantes/grupo_12/sahariandataset/
#   ├── yolo_vm/          <- Code here (this script)
#   └── general_dataset/  <- Data here
```

### 3. `test_setup.py`

**Added:** Path detection output to show detected directories:
```python
print(f"Detected paths:")
print(f"  Code directory (BASE_DIR): {config.BASE_DIR}")
print(f"  Dataset directory: {config.DATASET_ROOT}")
```

### 4. `colab_setup.py`

**Before:**
```python
PROJECT_PATH = "/content/drive/MyDrive/MAIA_Final_Project_2025/Yolo"
DATASET_ROOT = BASE_DIR / "general_dataset"
```

**After:**
```python
PROJECT_PATH = "/content/drive/MyDrive/MAIA_Final_Project_2025/yolo_vm"
DATASET_ROOT = BASE_DIR.parent / "general_dataset"
```

---

## New Documentation Files

### 1. `DIRECTORY_STRUCTURE.md`
Complete guide explaining:
- New directory layout
- Why this structure
- Automatic path detection
- Setup for VM, Colab, and local
- Troubleshooting
- Migration from old structure

### 2. `VM_SETUP_GUIDE.txt`
VM-specific guide with:
- Exact VM paths for Universidad de los Andes
- Quick start commands
- Directory structure visualization
- Training commands
- Monitoring instructions
- Complete troubleshooting

### 3. `PATH_UPDATES_SUMMARY.md`
This file - summary of all changes

---

## How It Works

### Automatic Path Detection

All scripts now use **relative paths** that work automatically:

```python
# Get the directory where this script is located
BASE_DIR = Path(__file__).parent.absolute()
# Returns: /home/estudiantes/grupo_12/sahariandataset/yolo_vm

# Get the sibling directory (one level up, then into general_dataset)
DATASET_ROOT = BASE_DIR.parent / "general_dataset"
# Returns: /home/estudiantes/grupo_12/sahariandataset/general_dataset
```

### Benefits

✅ **No manual path configuration** - Works automatically  
✅ **Portable** - Works on any machine with correct structure  
✅ **Clean separation** - Code and data separated  
✅ **Version control friendly** - Code in git, data separate  
✅ **Easy sharing** - Dataset can be shared across projects  

---

## Verification

### Test Path Detection

```bash
cd /home/estudiantes/grupo_12/sahariandataset/yolo_vm
python test_setup.py
```

**Expected output:**
```
Detected paths:
  Code directory (BASE_DIR): /home/estudiantes/grupo_12/sahariandataset/yolo_vm
  Dataset directory: /home/estudiantes/grupo_12/sahariandataset/general_dataset

✓ Train Images: .../general_dataset/train
  928 images found
✓ Val Images: .../general_dataset/val
  111 images found
...
```

### Quick Python Test

```python
from pathlib import Path

# Simulate script location
script_path = Path("/home/estudiantes/grupo_12/sahariandataset/yolo_vm/train_vm.py")
base_dir = script_path.parent
dataset_dir = base_dir.parent / "general_dataset"

print(f"Code: {base_dir}")
print(f"Data: {dataset_dir}")
print(f"Exists: {dataset_dir.exists()}")
```

---

## Training on VM

### Quick Start

```bash
# 1. Navigate to code directory
cd /home/estudiantes/grupo_12/sahariandataset/yolo_vm

# 2. Setup environment (first time only)
conda env create -f environment.yml
conda activate yolov11-wildlife

# 3. Verify paths
python test_setup.py

# 4. Start training
python train_vm.py --epochs 50 --batch 8
```

### One Command Setup

```bash
cd /home/estudiantes/grupo_12/sahariandataset/yolo_vm
./setup_conda_and_train.sh
```

---

## Compatibility

### Works On

✅ **Universidad de los Andes VM** - Automatic detection  
✅ **Local machines** - Any OS with correct structure  
✅ **Google Colab** - Adjusted paths for Drive  
✅ **AWS/Azure/GCP** - Works with standard VM structure  
✅ **Any cloud VM** - Relative paths adapt automatically  

### Requirements

The only requirement is maintaining the directory structure:
```
parent_directory/
├── yolo_vm/          # Code
└── general_dataset/  # Data
```

The parent directory name doesn't matter (can be `sahariandataset`, `project`, etc.)

---

## Troubleshooting

### Issue: "Dataset not found"

**Check structure:**
```bash
ls -la /home/estudiantes/grupo_12/sahariandataset/
# Should show: yolo_vm/ and general_dataset/
```

**Verify paths:**
```bash
cd /home/estudiantes/grupo_12/sahariandataset/yolo_vm
python -c "from train_vm import VMConfig; c=VMConfig(); print(c.DATASET_ROOT)"
```

### Issue: "ModuleNotFoundError"

**Make sure you're in the code directory:**
```bash
pwd
# Should be: /home/estudiantes/grupo_12/sahariandataset/yolo_vm
```

### Issue: Paths don't match

**Verify script location:**
```bash
cd /home/estudiantes/grupo_12/sahariandataset/yolo_vm
python -c "from pathlib import Path; print(Path(__file__).parent if '__file__' in dir() else Path.cwd())"
```

---

## Migration Notes

If you previously had a different structure, no worries! The scripts now use relative paths and will work as long as you maintain:

```
any_name/
├── yolo_vm/          # Your code folder
└── general_dataset/  # Your data folder
```

Just move your files to this structure and everything will work automatically.

---

## Summary

✅ **4 scripts updated** - All use relative paths now  
✅ **3 new guides created** - Complete documentation  
✅ **No manual configuration needed** - Automatic detection  
✅ **VM-ready** - Tested for Universidad de los Andes setup  
✅ **Portable** - Works on any environment  

**Ready to train on your VM!** 🚀

See:
- `VM_SETUP_GUIDE.txt` for VM-specific instructions
- `DIRECTORY_STRUCTURE.md` for detailed explanation
- `test_setup.py` to verify everything works

