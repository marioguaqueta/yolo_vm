# 🚀 Quick Start Guide - YOLOv11 Wildlife Detection

## For Local Training (Mac/Linux/Windows with GPU)

### 1. One-Command Setup and Training

```bash
./setup_and_train.sh
```

This will:
- Create virtual environment
- Install all dependencies
- Setup wandb
- Start training

### 2. Custom Training Parameters

```bash
./setup_and_train.sh --epochs 100 --batch 8 --imgsz 1024
```

### 3. Training without Wandb

```bash
./setup_and_train.sh --no-wandb
```

---

## For Google Colab

### Method 1: Using the Colab Setup Script

1. Upload your dataset to Google Drive at: `MyDrive/MAIA_Final_Project_2025/Yolo/`

2. Create a new Colab notebook

3. Copy the contents of `colab_setup.py` into cells

4. Update the `PROJECT_PATH` variable:
   ```python
   PROJECT_PATH = "/content/drive/MyDrive/MAIA_Final_Project_2025/Yolo"
   ```

5. Run all cells

### Method 2: Manual Setup

```python
# Cell 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Install Dependencies
!pip install ultralytics wandb pandas pillow

# Cell 3: Login to Wandb
import wandb
wandb.login()

# Cell 4: Change Directory
import os
os.chdir('/content/drive/MyDrive/MAIA_Final_Project_2025/Yolo')

# Cell 5: Run Training
!python train_vm.py --epochs 50 --batch 8 --imgsz 2048
```

---

## For Cloud VM (AWS, Azure, GCP)

### 1. Upload Dataset

```bash
# Using SCP
scp -r general_dataset/ user@vm-ip:/mnt/data/Yolo/

# Or using cloud storage (AWS S3 example)
aws s3 sync s3://your-bucket/dataset/ /mnt/data/Yolo/general_dataset/
```

### 2. SSH into VM and Setup

```bash
ssh user@vm-ip

# Clone or upload training scripts
cd /mnt/data/Yolo

# Install dependencies
pip install -r requirements.txt

# Login to wandb (get key from https://wandb.ai/authorize)
wandb login YOUR_API_KEY
```

### 3. Update Path in train_vm.py

Edit the `BASE_DIR` in `train_vm.py`:

```python
BASE_DIR = Path("/mnt/data/Yolo")  # Your actual path
```

### 4. Start Training

```bash
# Run in background with nohup
nohup python train_vm.py --epochs 50 --batch 8 --imgsz 2048 > training.log 2>&1 &

# Monitor progress
tail -f training.log

# Or use tmux/screen for session management
tmux new -s training
python train_vm.py --epochs 50 --batch 8 --imgsz 2048
# Detach: Ctrl+B, then D
```

---

## Command Line Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--epochs` | Number of training epochs | 50 | `--epochs 100` |
| `--batch` | Batch size | 4 (local) / 8 (VM) | `--batch 16` |
| `--imgsz` | Input image size | 2048 | `--imgsz 1024` |
| `--no-wandb` | Disable wandb tracking | False | `--no-wandb` |
| `--skip-conversion` | Skip CSV to YOLO conversion | False | `--skip-conversion` |
| `--wandb-key` | Wandb API key (for automation) | None | `--wandb-key abc123` |

---

## Expected Training Times

| Environment | GPU | Batch Size | Image Size | Time (50 epochs) |
|------------|-----|------------|------------|------------------|
| Google Colab | T4 (16GB) | 8 | 2048 | ~3-4 hours |
| Google Colab | T4 (16GB) | 4 | 2048 | ~5-6 hours |
| Local RTX 3090 | 24GB | 16 | 2048 | ~2-3 hours |
| AWS p3.2xlarge | V100 (16GB) | 8 | 2048 | ~2.5-3.5 hours |
| CPU Only | - | 2 | 1024 | ~30-40 hours ⚠️ |

---

## Monitoring Training

### Weights & Biases Dashboard

View real-time metrics at: https://wandb.ai/your-username/yolov11-wildlife-detection

Metrics tracked:
- Training/Validation Loss
- mAP@0.5 and mAP@0.5:0.95
- Precision & Recall (per-class and overall)
- Learning rate
- GPU utilization

### Local Results

Results saved in `runs/yolov11_wildlife/`:

```
runs/yolov11_wildlife/
├── weights/
│   ├── best.pt              # Best model checkpoint
│   └── last.pt              # Last epoch checkpoint
├── results.csv              # Training metrics
├── confusion_matrix.png     # Confusion matrix
├── F1_curve.png            # F1 score curve
├── PR_curve.png            # Precision-Recall curve
└── val_batch*.jpg          # Validation predictions
```

---

## Troubleshooting

### Out of Memory (OOM) Error

**Solution 1**: Reduce batch size
```bash
python train_yolov11_wildlife.py --batch 2
```

**Solution 2**: Reduce image size
```bash
python train_yolov11_wildlife.py --imgsz 1024
```

**Solution 3**: Use both
```bash
python train_yolov11_wildlife.py --batch 2 --imgsz 1024
```

### Wandb Login Failed

Train without wandb:
```bash
python train_yolov11_wildlife.py --no-wandb
```

Or set API key directly:
```bash
export WANDB_API_KEY=your_key_here
python train_yolov11_wildlife.py
```

### Dataset Not Found

Check and update paths in the script:

**For local training**: Edit `train_yolov11_wildlife.py`
```python
BASE_DIR = Path("/your/actual/path/to/Yolo")
```

**For VM training**: Edit `train_vm.py`
```python
BASE_DIR = Path("/your/vm/path/to/Yolo")
```

### Slow Training on CPU

GPUs are **highly recommended**. Free options:
- Google Colab (free T4 GPU for limited hours)
- Kaggle Notebooks (free GPU)
- Paperspace Gradient (free tier available)

---

## After Training

### Validate Model

```python
from ultralytics import YOLO

model = YOLO('runs/yolov11_wildlife/weights/best.pt')
results = model.val(data='yolo_wildlife_dataset/data.yaml')
```

### Run Inference

```python
model = YOLO('runs/yolov11_wildlife/weights/best.pt')

# Single image
results = model.predict('path/to/image.jpg', imgsz=2048, conf=0.25)

# Batch prediction
results = model.predict('path/to/images/', imgsz=2048, conf=0.25, save=True)
```

### Export Model

```python
model = YOLO('runs/yolov11_wildlife/weights/best.pt')

# Export to ONNX (for deployment)
model.export(format='onnx')

# Export to TensorRT (for NVIDIA GPUs)
model.export(format='engine')
```

---

## Tips for Best Results

1. **Use high resolution**: Aerial images have small objects, use `--imgsz 2048`

2. **Monitor validation metrics**: Watch mAP50 on wandb dashboard

3. **Early stopping**: Training stops automatically if no improvement for 10 epochs

4. **Save checkpoints**: Models saved every 5 epochs

5. **Resume training**: Use `last.pt` to resume if interrupted

6. **Class imbalance**: Some species (Warthog) have fewer samples, expect lower mAP

---

## Need Help?

- 📖 Full documentation: See `README_TRAINING.md`
- 🐛 Issues: Check training logs and wandb dashboard
- 📧 Contact: Project Guacamaya team

---

**Good luck with your training! 🦒🐘🦌**

