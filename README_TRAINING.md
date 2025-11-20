# YOLOv11 Wildlife Detection Training

This repository contains the training pipeline for the **Guacamaya Project** - a Microsoft AI for Good Lab initiative to develop automated wildlife detection and counting systems for aerial survey images.

## 📋 Overview

- **Model**: YOLOv11 (small variant)
- **Task**: Object detection for wildlife species
- **Dataset**: DelPlan 2022 aerial survey images (5000x4000 pixels)
- **Species**: Buffalo, Elephant, Kudu, Topi, Warthog, Waterbuck
- **Experiment Tracking**: Weights & Biases (wandb)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 16GB+ GPU memory (for 2048px images)

### Installation

1. **Clone the repository or navigate to the project directory**

```bash
cd /Users/marioguaqueta/Desktop/MAIA/2025-4/ProyectoFinal/train_config/Yolo
```

2. **Create a virtual environment** (recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Login to Weights & Biases** (optional but recommended)

```bash
wandb login
```

You'll need to enter your wandb API key. Get it from: https://wandb.ai/authorize

### Dataset Structure

Ensure your dataset is organized as follows:

```
Yolo/
├── general_dataset/
│   ├── train/              # Training images (928 images)
│   ├── val/                # Validation images (111 images)
│   ├── test/               # Test images (258 images)
│   └── groundtruth/
│       └── csv/
│           ├── train_big_size_A_B_E_K_WH_WB.csv
│           ├── val_big_size_A_B_E_K_WH_WB.csv
│           └── test_big_size_A_B_E_K_WH_WB.csv
└── train_yolov11_wildlife.py
```

CSV format: `Image,x1,y1,x2,y2,Label`

## 🎯 Training

### Basic Training

Run with default settings (50 epochs, batch size 4, image size 2048):

```bash
python train_yolov11_wildlife.py
```

### Custom Training Parameters

```bash
python train_yolov11_wildlife.py --epochs 100 --batch 8 --imgsz 1024
```

### Training without Wandb

```bash
python train_yolov11_wildlife.py --no-wandb
```

### Skip Dataset Conversion

If you've already converted the dataset and want to re-train:

```bash
python train_yolov11_wildlife.py --skip-conversion
```

## 🖥️ Training on GPU Virtual Machine

### Local GPU Setup

The script automatically detects and uses available GPU:

```bash
# Check GPU availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Cloud GPU Setup (Google Colab / AWS / Azure)

1. **Upload the script and dataset** to your cloud storage
2. **Install dependencies** in the VM:

```bash
pip install -r requirements.txt
```

3. **Modify paths in the script** (edit `train_yolov11_wildlife.py`):

```python
# For Google Colab with Google Drive
BASE_DIR = Path("/content/drive/MyDrive/your_project_path")

# For AWS/Azure with mounted storage
BASE_DIR = Path("/mnt/storage/your_project_path")
```

4. **Run training**:

```bash
python train_yolov11_wildlife.py
```

### Example: Google Colab Setup

```python
# In Colab notebook
from google.colab import drive
drive.mount('/content/drive')

!cd /content/drive/MyDrive/your_project_path && \
 pip install -r requirements.txt && \
 python train_yolov11_wildlife.py
```

## 📊 Monitoring Training

### Weights & Biases Dashboard

1. Training metrics (loss, mAP, precision, recall)
2. Per-class performance
3. Training/validation curves
4. System metrics (GPU usage, memory)

Access your dashboard at: https://wandb.ai/your-username/yolov11-wildlife-detection

### Local Results

Results are saved in:

```
runs/yolov11_wildlife/
├── weights/
│   ├── best.pt           # Best model checkpoint
│   └── last.pt           # Last epoch checkpoint
├── results.csv           # Training metrics
├── confusion_matrix.png  # Confusion matrix
├── F1_curve.png         # F1 score curve
├── PR_curve.png         # Precision-Recall curve
└── val_batch*.jpg       # Validation predictions
```

## 🎯 Model Configuration

The training uses these optimized parameters for aerial wildlife detection:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | yolo11s.pt | Small YOLOv11 variant |
| Epochs | 50 | Training iterations |
| Batch Size | 4 | Images per batch |
| Image Size | 2048 | Input resolution (high res for small animals) |
| Patience | 10 | Early stopping patience |
| Optimizer | Auto | Automatic optimizer selection |

### Data Augmentation

- HSV color jittering
- Horizontal flipping (50%)
- Translation (10%)
- Scaling (50%)
- Mosaic augmentation

## 📈 Expected Results

Based on the Entregable Proyecto notebook, expected performance:

| Species | mAP50 | Notes |
|---------|-------|-------|
| Elephant | ~80% | Excellent detection |
| Kudu | ~77% | Excellent detection |
| Buffalo | ~65% | Good detection |
| Waterbuck | ~40% | Fair detection |
| Warthog | ~29% | Challenging (few samples) |
| **Overall** | **~60%** | Competitive with HerdNet |

## 🔧 Troubleshooting

### Out of Memory Error

Reduce batch size or image size:

```bash
python train_yolov11_wildlife.py --batch 2 --imgsz 1024
```

### No GPU Available

The script will automatically use CPU, but training will be very slow. Consider:
- Using Google Colab (free GPU)
- AWS/Azure GPU instances
- Local GPU setup

### Wandb Login Issues

Train without wandb:

```bash
python train_yolov11_wildlife.py --no-wandb
```

### Dataset Path Issues

Edit the `Config` class in `train_yolov11_wildlife.py`:

```python
class Config:
    BASE_DIR = Path("/your/actual/path/to/Yolo")
    # ... rest of config
```

## 📚 Advanced Usage

### Resume Training

```python
from ultralytics import YOLO

# Load the last checkpoint
model = YOLO('runs/yolov11_wildlife/weights/last.pt')

# Continue training
model.train(
    data='yolo_wildlife_dataset/data.yaml',
    epochs=100,  # Total epochs
    resume=True  # Resume from last
)
```

### Custom Hyperparameters

Edit the `train_args` dictionary in `train_yolov11_wildlife.py`:

```python
train_args = {
    'data': str(yaml_path),
    'epochs': 100,
    'imgsz': 1024,
    'batch': 8,
    'lr0': 0.01,      # Initial learning rate
    'lrf': 0.001,     # Final learning rate
    'momentum': 0.937,
    'weight_decay': 0.0005,
    # ... more parameters
}
```

## 🌐 Inference

After training, use the model for predictions:

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/yolov11_wildlife/weights/best.pt')

# Predict on new images
results = model.predict(
    source='path/to/image.jpg',
    imgsz=2048,
    conf=0.25,  # Confidence threshold
    save=True   # Save annotated images
)
```

## 📖 Citation

If you use this code, please cite:

```
Proyecto Guacamaya - Microsoft AI for Good Lab
Instituciones: Microsoft AI for Good Lab, Centro SINFONÍA - Universidad de los Andes, 
               Instituto Sinchi, Instituto Alexander von Humboldt
Dataset: DelPlan 2022
```

## 🤝 Contributing

For questions or contributions, contact:
- Jorge Mario Guaquetá
- Daniel Santiago Trujillo
- Inmaculada Concepción Rondón
- Daniela Alexandra Ortiz Santacruz

## 📄 License

This project is part of the Microsoft AI for Good Lab initiative.

---

**Happy Training! 🚀🦒🐘**

