# Complete YOLOv11 Wildlife Detection Setup Guide

## 📋 Overview

This guide provides a complete, step-by-step process to set up and train YOLOv11 for wildlife detection on a VM with GPU.

**Project**: Guacamaya - Microsoft AI for Good Lab  
**Goal**: Aerial wildlife detection for conservation  
**Species**: Buffalo, Elephant, Kudu, Topi, Warthog, Waterbuck  

---

## 🗂️ Directory Structure

### Required Structure

```
/home/estudiante/grupo_12/subsaharian_dataset/
├── yolo_vm/                          ← Code directory
│   ├── train_vm.py                   ← Main training script
│   ├── convert_csv_to_yolo.py        ← CSV to YOLO converter
│   ├── reorganize_to_yolo_structure.py  ← Structure reorganizer
│   ├── environment.yml               ← Conda environment
│   ├── requirements.txt              ← Pip dependencies
│   └── [other scripts and docs]
│
└── general_dataset/                  ← Data directory (YOLO standard)
    ├── images/                       ← All images
    │   ├── train/                    (928 .JPG files)
    │   ├── val/                      (111 .JPG files)
    │   └── test/                     (258 .JPG files)
    ├── labels/                       ← All labels (YOLO format)
    │   ├── train/                    (928 .txt files)
    │   ├── val/                      (111 .txt files)
    │   └── test/                     (258 .txt files)
    └── groundtruth/                  ← Original annotations
        └── csv/
            ├── train_big_size_A_B_E_K_WH_WB.csv
            ├── val_big_size_A_B_E_K_WH_WB.csv
            └── test_big_size_A_B_E_K_WH_WB.csv
```

---

## 🚀 Complete Setup Process

### Step 1: Environment Setup

#### Option A: Conda (Recommended for GPU)

```bash
# Navigate to code directory
cd /home/estudiante/grupo_12/subsaharian_dataset/yolo_vm

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate yolov11-wildlife
```

#### Option B: Pip/Venv (Lighter, faster)

```bash
# Navigate to code directory
cd /home/estudiante/grupo_12/subsaharian_dataset/yolo_vm

# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 2: Convert CSV Annotations to YOLO Format

```bash
# Make sure you're in the yolo_vm directory
cd /home/estudiante/grupo_12/subsaharian_dataset/yolo_vm

# Run conversion script
python convert_csv_to_yolo.py
```

**What it does:**
- Reads CSV annotations from `general_dataset/groundtruth/csv/`
- Converts bounding boxes to YOLO format (normalized x_center, y_center, width, height)
- Maps class IDs (1-6 in CSV) to YOLO format (0-5)
- Creates `.txt` label files for each image
- Initially places labels alongside images

**Expected Output:**
```
✓ Processing train set...
✓ Processing val set...
✓ Processing test set...
✓ Total images processed: 1297
✓ Total annotations written: 6963
```

---

### Step 3: Reorganize to YOLO Standard Structure

```bash
# Run reorganization script
python reorganize_to_yolo_structure.py
```

**What it does:**
- Creates `images/` and `labels/` directories
- Moves all `.JPG` files to `images/train/`, `images/val/`, `images/test/`
- Moves all `.txt` files to `labels/train/`, `labels/val/`, `labels/test/`
- Cleans up empty old directories

**Expected Output:**
```
✓ Created: .../general_dataset/images/train
✓ Created: .../general_dataset/labels/train
✓ Moving train images...
✓ Moving train labels...
✓ Finished!
```

**Verify structure:**
```bash
# Check directories
ls -la ../general_dataset/
# Should show: images/ labels/ groundtruth/

# Count files (should match)
ls ../general_dataset/images/train/*.JPG | wc -l  # 928
ls ../general_dataset/labels/train/*.txt | wc -l  # 928
```

---

### Step 4: Configure Wandb (Optional but Recommended)

#### First time setup:
```bash
# Login to wandb
wandb login

# Paste your API key when prompted
# Get key from: https://wandb.ai/authorize
```

#### Or pass key in command:
```bash
python train_vm.py --wandb-key YOUR_API_KEY_HERE
```

#### Or disable wandb:
```bash
python train_vm.py --no-wandb
```

---

### Step 5: Train!

```bash
# Basic training (default settings)
python train_vm.py

# With custom settings
python train_vm.py --epochs 50 --batch 4 --imgsz 2048

# Without wandb
python train_vm.py --no-wandb --epochs 50

# Full example
python train_vm.py \
  --epochs 50 \
  --batch 4 \
  --imgsz 2048 \
  --wandb-key YOUR_KEY
```

**Training Parameters:**
- **epochs**: Number of training epochs (default: 50)
- **batch**: Batch size (default: 4 for 2048px, auto-reduces if GPU OOM)
- **imgsz**: Image size in pixels (default: 2048)
- **no-wandb**: Disable Weights & Biases logging
- **wandb-key**: Wandb API key for automated login

---

## 📊 Monitoring Training

### Wandb Dashboard

If wandb is enabled, you'll see:
```
✓ Wandb initialized
  Project: yolov11-wildlife-detection
  Dashboard: https://wandb.ai/your-username/yolov11-wildlife-detection/runs/...
```

**Key Metrics to Watch:**

| Metric | What it means | Good value |
|--------|---------------|------------|
| `mAP50` | Mean Average Precision @ 0.5 IoU | > 0.5 |
| `mAP50-95` | Mean Average Precision @ 0.5:0.95 IoU | > 0.3 |
| `Precision` | Correct predictions / Total predictions | > 0.7 |
| `Recall` | Correct predictions / Total ground truth | > 0.6 |
| `loss/box` | Bounding box regression loss | ↓ decreasing |
| `loss/cls` | Classification loss | ↓ decreasing |
| `loss/dfl` | Distribution focal loss | ↓ decreasing |

**Per-Class Metrics:**
- `metrics/Buffalo_mAP50`
- `metrics/Elephant_mAP50`
- `metrics/Kudu_mAP50`
- `metrics/Topi_mAP50`
- `metrics/Warthog_mAP50`
- `metrics/Waterbuck_mAP50`

### Local Monitoring

Training progress is also saved locally:
```bash
# Results directory
cd runs/yolov11_wildlife/

# View training curves
open results.png

# View confusion matrix
open confusion_matrix.png

# View predictions
open val_batch0_pred.jpg
```

---

## ✅ Verification Checklist

Before training, verify:

- [ ] Environment activated (`conda activate yolov11-wildlife` or `source venv/bin/activate`)
- [ ] Directory structure correct (run `ls -la ../general_dataset/`)
- [ ] Images exist (`ls ../general_dataset/images/train/*.JPG | wc -l` → 928)
- [ ] Labels exist (`ls ../general_dataset/labels/train/*.txt | wc -l` → 928)
- [ ] GPU available (`nvidia-smi` shows your GPU)
- [ ] Wandb configured (optional, `wandb login`)

---

## 🔧 Troubleshooting

### Error: "Training images not found"

**Cause**: Images not in standard YOLO structure  
**Solution**:
```bash
python reorganize_to_yolo_structure.py
```

### Error: "Training labels not found"

**Cause**: Labels not created or not in correct structure  
**Solution**:
```bash
python convert_csv_to_yolo.py
python reorganize_to_yolo_structure.py
```

### Error: "CUDA out of memory"

**Cause**: GPU memory exhausted  
**Solution 1** - Reduce batch size:
```bash
python train_vm.py --batch 2  # or --batch 1
```

**Solution 2** - Reduce image size:
```bash
python train_vm.py --imgsz 1024
```

**Solution 3** - Set memory environment variable:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_vm.py
```

### Error: "No space left on device"

**Cause**: Disk full  
**Solution**:
```bash
# Clean conda cache
conda clean --all -y

# Clean pip cache
pip cache purge

# Or use the automated cleanup script
bash cleanup_and_setup.sh
```

### Error: "Mismatch between images and labels"

**Cause**: Some images don't have corresponding label files  
**Solution**: Check CSV files for completeness, re-run conversion
```bash
python convert_csv_to_yolo.py
```

---

## 🎯 Training Configuration

### Default Settings (train_vm.py)

```python
MODEL = "yolo11s.pt"      # Starting model (small)
EPOCHS = 50               # Training epochs
BATCH_SIZE = 4            # For 2048px images
IMG_SIZE = 2048           # High res for aerial images
PATIENCE = 10             # Early stopping patience
WORKERS = 8               # Data loading workers
SAVE_PERIOD = 5           # Save checkpoint every 5 epochs

# Data augmentation
hsv_h = 0.015             # Hue augmentation
hsv_s = 0.7               # Saturation augmentation
hsv_v = 0.4               # Value augmentation
degrees = 0.0             # Rotation (disabled for aerial)
translate = 0.1           # Translation
scale = 0.5               # Scale augmentation
fliplr = 0.5              # Horizontal flip
mosaic = 1.0              # Mosaic augmentation
```

### Model Options

| Model | Size | Speed | Accuracy | Use case |
|-------|------|-------|----------|----------|
| `yolo11n.pt` | 2.5M | Fastest | Lower | Quick tests |
| `yolo11s.pt` | 9.4M | Fast | Good | **Default** |
| `yolo11m.pt` | 20M | Medium | Better | More accuracy |
| `yolo11l.pt` | 25M | Slow | High | Best accuracy |
| `yolo11x.pt` | 56M | Slowest | Highest | Maximum accuracy |

**⚠️ Important**: Use regular models (`.pt`), NOT oriented bounding box models (`-obb.pt`)!

---

## 📈 Expected Results

### Training Time (L40 GPU, batch=4, imgsz=2048)

- **Per epoch**: ~5-10 minutes
- **50 epochs**: ~4-8 hours
- **Early stopping**: May finish earlier if no improvement

### Target Metrics (End of Training)

- **mAP50**: 0.5-0.7 (higher is better)
- **mAP50-95**: 0.3-0.5 (higher is better)
- **Precision**: 0.6-0.8 (higher is better)
- **Recall**: 0.5-0.7 (higher is better)

### Output Files

After training, you'll have:

```
runs/yolov11_wildlife/
├── weights/
│   ├── best.pt          ← Best model checkpoint (use this!)
│   └── last.pt          ← Last epoch checkpoint
├── results.png          ← Training curves
├── confusion_matrix.png ← Class confusion matrix
├── F1_curve.png         ← F1 score curve
├── PR_curve.png         ← Precision-Recall curve
├── P_curve.png          ← Precision curve
├── R_curve.png          ← Recall curve
└── val_batch*_pred.jpg  ← Validation predictions
```

---

## 🎓 Key Concepts

### YOLO Label Format

Each `.txt` file contains one line per object:
```
class_id x_center y_center width height
```

Where:
- `class_id`: 0-5 (Buffalo, Elephant, Kudu, Topi, Warthog, Waterbuck)
- `x_center`, `y_center`: Normalized center coordinates (0.0-1.0)
- `width`, `height`: Normalized box dimensions (0.0-1.0)

Example:
```
0 0.5234 0.6123 0.1234 0.0987  # Buffalo at center-right
1 0.2341 0.3456 0.0876 0.1123  # Elephant at left
```

### Class Mapping

| ID | Species | CSV Label |
|----|---------|-----------|
| 0 | Buffalo | 1 |
| 1 | Elephant | 2 |
| 2 | Kudu | 3 |
| 3 | Topi | 4 |
| 4 | Warthog | 5 |
| 5 | Waterbuck | 6 |

**⚠️ Critical**: YOLO classes MUST start at 0, not 1!

---

## 🔗 Useful Commands

### Environment Management

```bash
# Activate conda environment
conda activate yolov11-wildlife

# Deactivate
conda deactivate

# Activate venv
source venv/bin/activate

# Deactivate
deactivate
```

### GPU Monitoring

```bash
# Check GPU status
nvidia-smi

# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Check CUDA version
nvcc --version
```

### File Management

```bash
# Count files
ls ../general_dataset/images/train/*.JPG | wc -l
ls ../general_dataset/labels/train/*.txt | wc -l

# Check disk space
df -h

# Check directory sizes
du -sh ../general_dataset/*
```

### Process Management

```bash
# Run training in background
nohup python train_vm.py --epochs 50 > training.log 2>&1 &

# Check background jobs
jobs

# View training log
tail -f training.log

# Kill training
pkill -f train_vm.py
```

---

## 📚 Additional Resources

### Documentation Files

- `STRUCTURE_ERROR_FIX.txt` - Fix for "images not found" error
- `GPU_MEMORY_GUIDE.md` - GPU memory optimization
- `WANDB_METRICS_GUIDE.md` - Detailed metrics explanation
- `LABELS_SETUP.md` - Label conversion guide
- `DISK_SPACE_FIX.md` - Disk space issues

### Scripts

- `train_vm.py` - Main training script
- `convert_csv_to_yolo.py` - CSV to YOLO converter
- `reorganize_to_yolo_structure.py` - Structure reorganizer
- `test_setup.py` - Setup verification
- `cleanup_and_setup.sh` - Automated cleanup and setup

### External Links

- [Ultralytics YOLOv11 Docs](https://docs.ultralytics.com/)
- [Wandb Documentation](https://docs.wandb.ai/)
- [YOLO Format Explained](https://docs.ultralytics.com/datasets/detect/)

---

## ✨ Quick Start Summary

**First time setup (5 minutes):**
```bash
cd /home/estudiante/grupo_12/subsaharian_dataset/yolo_vm
conda env create -f environment.yml
conda activate yolov11-wildlife
wandb login
```

**Data preparation (5-10 minutes):**
```bash
python convert_csv_to_yolo.py
python reorganize_to_yolo_structure.py
```

**Training (4-8 hours):**
```bash
python train_vm.py --epochs 50 --batch 4
```

**That's it!** 🎉

---

## 📝 Notes

1. **Always activate environment** before running scripts
2. **Run scripts from yolo_vm directory** (not from general_dataset)
3. **Data preparation is one-time** - only run once unless data changes
4. **Monitor GPU usage** during training to optimize batch size
5. **Check wandb dashboard** for real-time training progress
6. **Save best.pt model** for inference/deployment
7. **Backup checkpoints** regularly during long training runs

---

## 🆘 Getting Help

If you encounter issues:

1. Check error message carefully
2. Look for relevant `*.txt` or `*.md` guide in this directory
3. Verify directory structure is correct
4. Check GPU memory with `nvidia-smi`
5. Check disk space with `df -h`
6. Review training logs for clues
7. Check wandb dashboard for metric anomalies

---

**Last Updated**: 2025-11-20  
**Version**: 2.0 (Fixed YOLO structure compatibility)  
**Status**: ✅ Production Ready

Good luck with your training! 🚀🦁🐘

