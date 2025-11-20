# YOLOv11 Wildlife Detection Training - Project Summary

## 📁 Project Files Created

This project includes all necessary files to train a YOLOv11 model for wildlife detection with complete wandb integration for experiment tracking.

### Core Training Scripts

1. **`train_yolov11_wildlife.py`** ⭐ Main training script (local/desktop)
   - Complete data preparation pipeline (CSV → YOLO format)
   - Full training loop with YOLOv11
   - Wandb integration for experiment tracking
   - Validation and metrics reporting
   - GPU/CPU automatic detection

2. **`train_vm.py`** ☁️ Cloud/VM optimized training script
   - Optimized for Google Colab, AWS, Azure
   - Auto-detects cloud environment
   - Progress bars for long operations
   - Memory-efficient processing
   - Higher batch sizes for GPU instances

3. **`test_setup.py`** 🔍 Setup verification script
   - Tests Python version
   - Checks all dependencies
   - Verifies GPU availability
   - Validates dataset structure
   - Tests wandb login
   - Checks disk space and permissions

### Setup & Automation

4. **`setup_and_train.sh`** 🚀 One-command setup script
   - Creates virtual environment
   - Installs all dependencies
   - Configures wandb
   - Starts training
   - Works on Mac/Linux/WSL

5. **`colab_setup.py`** 📓 Google Colab notebook code
   - Ready-to-use Colab cells
   - Google Drive integration
   - Automatic environment setup
   - Dataset verification
   - One-click training

### Configuration & Dependencies

6. **`requirements.txt`** 📦 Python dependencies
   - PyTorch and torchvision
   - Ultralytics YOLOv11
   - Wandb for experiment tracking
   - Data processing libraries
   - All pinned to stable versions

### Documentation

7. **`README_TRAINING.md`** 📖 Comprehensive guide
   - Complete project overview
   - Installation instructions
   - Training configuration details
   - Cloud deployment guides
   - Troubleshooting section
   - Advanced usage examples

8. **`QUICKSTART.md`** ⚡ Quick reference
   - Fast-start commands
   - Environment-specific guides
   - Common parameters
   - Troubleshooting quick fixes
   - Training time estimates

9. **`PROJECT_SUMMARY.md`** 📋 This file
   - Overview of all files
   - Project structure
   - Key features
   - Usage workflows

## 🎯 Key Features

### Data Processing
- ✅ Automatic CSV to YOLO format conversion
- ✅ Support for bounding box annotations
- ✅ Multi-split dataset handling (train/val/test)
- ✅ Image format flexibility (JPG, jpg, png, jpeg)
- ✅ Automatic data validation

### Training
- ✅ YOLOv11 small model (fast and accurate)
- ✅ High-resolution support (2048px for aerial images)
- ✅ GPU acceleration with auto-detection
- ✅ Mixed precision training (AMP)
- ✅ Early stopping
- ✅ Checkpoint saving every 5 epochs
- ✅ Data augmentation optimized for wildlife

### Experiment Tracking
- ✅ Full Weights & Biases integration
- ✅ Real-time metrics visualization
- ✅ Training curves and loss plots
- ✅ Per-class performance tracking
- ✅ System metrics (GPU, memory)
- ✅ Model artifacts saving
- ✅ Experiment comparison

### Deployment Support
- ✅ Local training (Mac/Linux/Windows)
- ✅ Google Colab integration
- ✅ AWS/Azure/GCP compatibility
- ✅ Docker-ready (can be containerized)
- ✅ Multiple GPU support

## 🔄 Training Workflow

### Option 1: Quick Start (Recommended)

```bash
# Test setup
./test_setup.py

# Start training
./setup_and_train.sh
```

### Option 2: Manual Control

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Login to wandb
wandb login

# Run training
python train_yolov11_wildlife.py --epochs 50 --batch 8 --imgsz 2048
```

### Option 3: Google Colab

```python
# Copy colab_setup.py cells into Colab notebook
# Update PROJECT_PATH
# Run all cells
```

### Option 4: Cloud VM

```bash
# SSH into VM
ssh user@vm-ip

# Upload files
# Update BASE_DIR in train_vm.py

# Install and train
pip install -r requirements.txt
wandb login YOUR_API_KEY
python train_vm.py --epochs 50 --batch 8 --imgsz 2048
```

## 📊 Dataset Information

### Species Classes (6 total)
1. Buffalo (Bovines)
2. Elephant
3. Kudu
4. Topi
5. Warthog
6. Waterbuck

### Dataset Splits
- **Train**: 928 images
- **Val**: 111 images
- **Test**: 258 images
- **Total**: 1,297 high-resolution aerial images (5000×4000 pixels)

### Annotation Format
- **Input**: CSV files with bounding boxes (x1, y1, x2, y2)
- **Output**: YOLO format (class x_center y_center width height)
- All coordinates normalized to [0, 1]

## 🔧 Configuration

### Default Training Parameters
```python
MODEL = "yolo11s.pt"        # YOLOv11 small
EPOCHS = 50                 # Training iterations
BATCH_SIZE = 4-8            # Auto-adjusted for GPU
IMG_SIZE = 2048             # High-res for small objects
PATIENCE = 10               # Early stopping
DEVICE = "auto"             # GPU if available, else CPU
```

### Data Augmentation
- HSV color jittering (H: 0.015, S: 0.7, V: 0.4)
- Horizontal flipping (50%)
- Translation (10%)
- Scaling (50%)
- Mosaic augmentation (100%)

## 📈 Expected Results

Based on notebook experiments:

| Metric | Target | Notes |
|--------|--------|-------|
| Overall mAP@50 | ~60% | Competitive with HerdNet baseline |
| Elephant mAP@50 | ~80% | Excellent (large, distinct) |
| Kudu mAP@50 | ~77% | Excellent |
| Buffalo mAP@50 | ~65% | Good |
| Waterbuck mAP@50 | ~40% | Fair |
| Warthog mAP@50 | ~29% | Challenging (few samples) |

## 📁 Output Structure

After training, results are saved in:

```
runs/yolov11_wildlife/
├── weights/
│   ├── best.pt               # Best model (use this)
│   └── last.pt               # Last checkpoint
├── results.csv               # Training metrics
├── confusion_matrix.png      # Per-class confusion
├── F1_curve.png             # F1 scores
├── PR_curve.png             # Precision-Recall
├── results.png              # Training curves
└── val_batch*.jpg           # Prediction samples
```

## 🌐 Wandb Dashboard

Access your training dashboard at:
```
https://wandb.ai/your-username/yolov11-wildlife-detection
```

View:
- Real-time training progress
- Loss curves (train/val)
- mAP metrics over time
- Per-class performance
- System metrics (GPU usage)
- Model comparison
- Hyperparameter tracking

## 💡 Tips for Success

1. **Start with test_setup.py** - Verify everything before training
2. **Use high resolution** - Aerial images need 2048px for small animals
3. **Monitor wandb** - Track training in real-time
4. **Save GPU memory** - Reduce batch size if OOM errors
5. **Early stopping** - Training stops automatically when no improvement
6. **Use checkpoints** - Resume training from saved models
7. **Test on Colab first** - Free GPU to validate setup

## 🐛 Common Issues & Solutions

### Issue: Out of Memory
**Solution**: Reduce batch size or image size
```bash
python train_yolov11_wildlife.py --batch 2 --imgsz 1024
```

### Issue: Slow training on CPU
**Solution**: Use cloud GPU (Colab, Kaggle, Paperspace)

### Issue: Wandb login fails
**Solution**: Train without wandb or set API key
```bash
python train_yolov11_wildlife.py --no-wandb
```

### Issue: Dataset not found
**Solution**: Update BASE_DIR in the training script

### Issue: Missing dependencies
**Solution**: Install requirements
```bash
pip install -r requirements.txt
```

## 📞 Support & Contact

**Project**: Guacamaya - Microsoft AI for Good Lab

**Team**:
- Jorge Mario Guaquetá
- Daniel Santiago Trujillo
- Inmaculada Concepción Rondón
- Daniela Alexandra Ortiz Santacruz

**Institutions**:
- Microsoft AI for Good Lab
- Centro SINFONÍA - Universidad de los Andes
- Instituto Sinchi
- Instituto Alexander von Humboldt

## 📄 License

Part of Microsoft AI for Good Lab initiative for wildlife conservation.

## 🎓 Citation

If you use this code, please acknowledge:
```
Proyecto Guacamaya - Microsoft AI for Good Lab
Dataset: DelPlan 2022
Institutions: Microsoft AI for Good Lab, Universidad de los Andes, 
              Instituto Sinchi, Instituto Alexander von Humboldt
```

---

## 🚀 Ready to Start?

1. **Verify setup**: `./test_setup.py`
2. **Start training**: `./setup_and_train.sh`
3. **Monitor progress**: Check wandb dashboard
4. **Use results**: Best model saved in `runs/yolov11_wildlife/weights/best.pt`

**Happy training! 🦒🐘🦌🐗🦌🐃**

