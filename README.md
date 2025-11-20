# YOLOv11 Wildlife Detection Training

**Project**: Guacamaya - Microsoft AI for Good Lab  
**Institution**: Universidad de los Andes  
**Purpose**: Aerial wildlife detection for conservation  
**Status**: ✅ Production Ready (v2.0)

---

## 🚀 Quick Start

```bash
# 1. Setup (first time)
cd /home/estudiante/grupo_12/subsaharian_dataset/yolo_vm
conda env create -f environment.yml
conda activate yolov11-wildlife

# 2. Prepare data (first time)
python convert_csv_to_yolo.py
python reorganize_to_yolo_structure.py

# 3. Train
python train_vm.py --epochs 50 --batch 4
```

**→ See [`QUICK_START.txt`](QUICK_START.txt) for one-page reference**

---

## 📚 Documentation Index

### 🎯 Getting Started

| File | Purpose | When to Use |
|------|---------|-------------|
| **`QUICK_START.txt`** | One-page command reference | First time setup, quick reminders |
| **`COMPLETE_SETUP_GUIDE.md`** | Comprehensive guide with all details | Detailed setup, troubleshooting |
| **`README.md`** *(this file)* | Overview and documentation index | Find the right documentation |

### 🐛 Troubleshooting

| File | Purpose | When to Use |
|------|---------|-------------|
| **`STRUCTURE_ERROR_FIX.txt`** | Fix "training images not found" error | After reorganization, path errors |
| **`GPU_MEMORY_GUIDE.md`** | Fix CUDA out of memory errors | GPU OOM during training |
| **`DISK_SPACE_FIX.md`** | Fix disk space issues | "No space left on device" |
| **`LABELS_SETUP.md`** | CSV to YOLO label conversion guide | Label format issues |
| **`CONDA_SETUP.md`** | Conda environment setup issues | Environment creation problems |

### 📊 Monitoring & Metrics

| File | Purpose | When to Use |
|------|---------|-------------|
| **`WANDB_METRICS_GUIDE.md`** | Detailed metrics explanation | Understanding training progress |
| **`WANDB_QUICK_REFERENCE.txt`** | Quick metrics reference | Quick lookup during training |

### 🔧 Technical Details

| File | Purpose | When to Use |
|------|---------|-------------|
| **`CHANGES_SUMMARY.md`** | Detailed changelog and fixes | Understanding what changed |
| **`VISUAL_STRUCTURE_GUIDE.txt`** | Visual diagram of structure transformation | Understanding YOLO structure |
| **`DIRECTORY_STRUCTURE.md`** | Complete directory structure reference | Understanding project layout |

### 📋 Quick References

| File | Purpose | When to Use |
|------|---------|-------------|
| **`QUICK_GPU_FIX.txt`** | GPU memory quick fix | Fast OOM solution |
| **`QUICK_DISK_FIX.txt`** | Disk space quick fix | Fast disk space solution |
| **`QUICK_LABELS_FIX.txt`** | Labels quick fix | Fast label conversion |
| **`YOLO_STRUCTURE_FIX.txt`** | Structure quick fix | Fast structure fix |

---

## 🗂️ Project Structure

```
yolo_vm/                          ← You are here
├── 📋 Training Scripts
│   ├── train_vm.py               ← Main training script (VM/Cloud)
│   ├── train_yolov11_wildlife.py ← Training script (Local)
│   ├── convert_csv_to_yolo.py    ← CSV to YOLO converter
│   └── reorganize_to_yolo_structure.py ← Structure reorganizer
│
├── 🛠️ Setup Scripts
│   ├── setup_and_train.sh        ← Pip/venv setup
│   ├── setup_conda_and_train.sh  ← Conda setup
│   ├── create_environment.sh     ← Interactive setup
│   └── cleanup_and_setup.sh      ← Cleanup + setup
│
├── 📦 Environment Files
│   ├── environment.yml           ← Conda env (GPU)
│   ├── environment-cpu.yml       ← Conda env (CPU)
│   ├── environment-simple.yml    ← Conda env (simple)
│   └── requirements.txt          ← Pip requirements
│
├── 📚 Documentation (Main Guides)
│   ├── README.md                 ← This file
│   ├── QUICK_START.txt           ← Quick reference
│   ├── COMPLETE_SETUP_GUIDE.md   ← Complete guide
│   ├── CHANGES_SUMMARY.md        ← Changelog
│   └── VISUAL_STRUCTURE_GUIDE.txt ← Visual diagram
│
├── 🐛 Documentation (Troubleshooting)
│   ├── STRUCTURE_ERROR_FIX.txt   ← Path/structure errors
│   ├── GPU_MEMORY_GUIDE.md       ← GPU OOM errors
│   ├── DISK_SPACE_FIX.md         ← Disk space errors
│   ├── LABELS_SETUP.md           ← Label conversion
│   └── CONDA_SETUP.md            ← Conda issues
│
├── 📊 Documentation (Monitoring)
│   ├── WANDB_METRICS_GUIDE.md    ← Metrics explained
│   └── WANDB_QUICK_REFERENCE.txt ← Metrics quick ref
│
└── 📋 Documentation (Quick Fixes)
    ├── QUICK_GPU_FIX.txt
    ├── QUICK_DISK_FIX.txt
    ├── QUICK_LABELS_FIX.txt
    └── YOLO_STRUCTURE_FIX.txt
```

---

## 🎯 Common Tasks

### First Time Setup

```bash
# See: QUICK_START.txt or COMPLETE_SETUP_GUIDE.md
cd /home/estudiante/grupo_12/subsaharian_dataset/yolo_vm
conda env create -f environment.yml
conda activate yolov11-wildlife
python convert_csv_to_yolo.py
python reorganize_to_yolo_structure.py
```

### Start Training

```bash
# See: QUICK_START.txt
conda activate yolov11-wildlife
python train_vm.py --epochs 50 --batch 4
```

### Fix "Images Not Found" Error

```bash
# See: STRUCTURE_ERROR_FIX.txt
python reorganize_to_yolo_structure.py
```

### Fix GPU Out of Memory

```bash
# See: GPU_MEMORY_GUIDE.md or QUICK_GPU_FIX.txt
python train_vm.py --batch 2 --imgsz 1024
```

### Monitor Training

```bash
# See: WANDB_METRICS_GUIDE.md
# Check wandb dashboard URL printed during training
# Or watch GPU: watch -n 1 nvidia-smi
```

---

## ⚡ Recent Changes (v2.0)

**Date**: 2025-11-20  
**Status**: ✅ Fixed and tested

### What Changed

1. **Updated `train_vm.py` to use YOLO standard structure**
   - Paths now point to `images/train/` and `labels/train/`
   - Fixed class IDs (0-5 instead of 1-6)
   - Fixed model type (yolo11s.pt instead of yolo11x-obb.pt)
   - Improved error messages with solutions

2. **Created comprehensive documentation**
   - Quick start guide
   - Complete setup guide
   - Visual structure guide
   - Changes summary
   - Multiple troubleshooting guides

3. **Added dataset statistics**
   - Shows image/label counts
   - Warns if counts don't match
   - Helps catch data issues early

**→ See [`CHANGES_SUMMARY.md`](CHANGES_SUMMARY.md) for detailed changelog**

---

## 📊 Dataset Info

- **Total images**: 1,297
  - Training: 928 images
  - Validation: 111 images
  - Test: 258 images

- **Total annotations**: 6,963 objects

- **Species** (6 classes):
  - 0: Buffalo
  - 1: Elephant
  - 2: Kudu
  - 3: Topi
  - 4: Warthog
  - 5: Waterbuck

- **Image specs**:
  - Format: JPG
  - Original size: 5000x4000 pixels
  - Training size: 2048x2048 (configurable)

---

## 🛠️ Configuration

### Default Training Settings

```python
MODEL = "yolo11s.pt"      # Small model
EPOCHS = 50               # Training epochs
BATCH_SIZE = 4            # For 2048px images
IMG_SIZE = 2048           # High res for aerial
PATIENCE = 10             # Early stopping
WORKERS = 8               # Data loading threads
```

### Available Models

| Model | Params | Speed | Accuracy | Best for |
|-------|--------|-------|----------|----------|
| yolo11n.pt | 2.5M | ⚡⚡⚡⚡⚡ | ⭐⭐ | Quick tests |
| yolo11s.pt | 9.4M | ⚡⚡⚡⚡ | ⭐⭐⭐ | **Default** |
| yolo11m.pt | 20M | ⚡⚡⚡ | ⭐⭐⭐⭐ | More accuracy |
| yolo11l.pt | 25M | ⚡⚡ | ⭐⭐⭐⭐⭐ | Best accuracy |
| yolo11x.pt | 56M | ⚡ | ⭐⭐⭐⭐⭐⭐ | Max accuracy |

### Training Options

```bash
python train_vm.py [OPTIONS]

Options:
  --epochs N      Number of training epochs (default: 50)
  --batch N       Batch size (default: 4)
  --imgsz N       Image size in pixels (default: 2048)
  --no-wandb      Disable wandb logging
  --wandb-key K   Wandb API key for automated login

Examples:
  python train_vm.py
  python train_vm.py --epochs 100 --batch 8
  python train_vm.py --no-wandb --epochs 50
  python train_vm.py --wandb-key abc123 --epochs 50
```

---

## 📈 Expected Results

### Training Time

- **GPU**: NVIDIA L40 (24GB)
- **Per epoch**: ~5-10 minutes
- **50 epochs**: ~4-8 hours
- **Total pipeline**: ~5-9 hours (including setup)

### Target Metrics

| Metric | Good | Excellent |
|--------|------|-----------|
| mAP50 | > 0.5 | > 0.7 |
| mAP50-95 | > 0.3 | > 0.5 |
| Precision | > 0.6 | > 0.8 |
| Recall | > 0.5 | > 0.7 |

### Output Files

After training:
```
runs/yolov11_wildlife/
├── weights/
│   ├── best.pt          ← Use this for inference! ⭐
│   └── last.pt
├── results.png
├── confusion_matrix.png
└── val_batch*_pred.jpg
```

---

## 🔗 External Resources

- **Ultralytics YOLOv11**: https://docs.ultralytics.com/
- **Weights & Biases**: https://docs.wandb.ai/
- **YOLO Format**: https://docs.ultralytics.com/datasets/detect/
- **Project GitHub**: (Add your repo URL here)

---

## 🆘 Getting Help

### Step 1: Identify Your Issue

| Issue | Documentation |
|-------|---------------|
| First time setup | `QUICK_START.txt` or `COMPLETE_SETUP_GUIDE.md` |
| "Images not found" error | `STRUCTURE_ERROR_FIX.txt` |
| CUDA out of memory | `GPU_MEMORY_GUIDE.md` or `QUICK_GPU_FIX.txt` |
| Disk space issues | `DISK_SPACE_FIX.md` or `QUICK_DISK_FIX.txt` |
| Label conversion | `LABELS_SETUP.md` or `QUICK_LABELS_FIX.txt` |
| Understanding metrics | `WANDB_METRICS_GUIDE.md` |
| Conda issues | `CONDA_SETUP.md` |

### Step 2: Check Documentation

1. Look for the relevant `.txt` or `.md` file
2. Follow the step-by-step instructions
3. Verify with the provided commands

### Step 3: Verify Your Setup

```bash
# Structure check
ls -la ../general_dataset/
# Should show: images/ labels/ groundtruth/

# File counts
ls ../general_dataset/images/train/*.JPG | wc -l  # 928
ls ../general_dataset/labels/train/*.txt | wc -l  # 928

# GPU check
nvidia-smi

# Disk space
df -h
```

---

## ✅ Pre-Flight Checklist

Before training, verify:

- [ ] Conda environment created and activated
- [ ] Dataset structure correct (`images/` and `labels/` directories exist)
- [ ] Image count matches label count (928 train, 111 val)
- [ ] GPU available and has free memory
- [ ] Sufficient disk space (> 10GB free)
- [ ] Wandb configured (optional)

**→ See [`COMPLETE_SETUP_GUIDE.md`](COMPLETE_SETUP_GUIDE.md) for detailed checklist**

---

## 🎓 Key Concepts

### YOLO Standard Structure

```
dataset/
├── images/          ← All images
│   ├── train/
│   ├── val/
│   └── test/
└── labels/          ← All labels
    ├── train/
    ├── val/
    └── test/
```

### YOLO Label Format

Each `.txt` file (one per image):
```
class_id x_center y_center width height
```
- All values normalized (0.0-1.0)
- class_id is 0-indexed (0, 1, 2, ...)

Example:
```
0 0.5234 0.6123 0.1234 0.0987  # Buffalo
1 0.2341 0.3456 0.0876 0.1123  # Elephant
```

### Class Mapping

| CSV Label | YOLO ID | Species |
|-----------|---------|---------|
| 1 | 0 | Buffalo |
| 2 | 1 | Elephant |
| 3 | 2 | Kudu |
| 4 | 3 | Topi |
| 5 | 4 | Warthog |
| 6 | 5 | Waterbuck |

**⚠️ Critical**: YOLO classes MUST start at 0!

---

## 📝 Notes

1. **Environment**: Always activate before running scripts
2. **Directory**: Always run scripts from `yolo_vm/` directory
3. **Data prep**: Only run once unless data changes
4. **GPU monitoring**: Watch with `nvidia-smi` during training
5. **Checkpoints**: Save `best.pt` for production use
6. **Wandb**: Use for experiment tracking and comparison

---

## 🌟 Features

- ✅ YOLO standard structure support
- ✅ CSV to YOLO annotation converter
- ✅ Automatic dataset reorganization
- ✅ Weights & Biases integration
- ✅ GPU optimization (memory efficient)
- ✅ Early stopping
- ✅ Automatic checkpointing
- ✅ Data augmentation
- ✅ Mixed precision training
- ✅ Comprehensive documentation
- ✅ Error handling with clear messages

---

## 🔄 Workflow Summary

```
1. Setup Environment        → conda env create -f environment.yml
2. Activate Environment     → conda activate yolov11-wildlife
3. Convert CSV to YOLO      → python convert_csv_to_yolo.py
4. Reorganize Structure     → python reorganize_to_yolo_structure.py
5. Train Model              → python train_vm.py --epochs 50
6. Monitor Progress         → Check wandb dashboard
7. Use Best Model           → runs/yolov11_wildlife/weights/best.pt
```

---

## 📊 Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | 2025-11-20 | ✅ Fixed YOLO structure compatibility |
| 1.5 | 2025-11-20 | Added CSV conversion script |
| 1.4 | 2025-11-20 | GPU memory optimization |
| 1.3 | 2025-11-20 | Disk space fixes |
| 1.2 | 2025-11-20 | Conda environment fixes |
| 1.1 | 2025-11-20 | Relative path support |
| 1.0 | 2025-11-20 | Initial release |

---

## 👥 Credits

**Project**: Guacamaya  
**Lab**: Microsoft AI for Good Lab  
**Institution**: Universidad de los Andes  
**Dataset**: DelPlan 2022  
**Framework**: Ultralytics YOLOv11  

---

## 📄 License

See project repository for license information.

---

**Status**: ✅ Production Ready (v2.0)  
**Last Updated**: 2025-11-20  
**Confidence**: 🔥🔥🔥 100%

**Ready to train!** 🚀🦁🐘

---

## 🚀 Get Started Now

```bash
# Copy and paste these commands:
cd /home/estudiante/grupo_12/subsaharian_dataset/yolo_vm
conda activate yolov11-wildlife
python train_vm.py --epochs 50 --batch 4
```

**Good luck!** 🎉

