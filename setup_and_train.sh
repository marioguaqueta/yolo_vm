#!/bin/bash
# Setup and Train Script for YOLOv11 Wildlife Detection
# Usage: ./setup_and_train.sh [--no-wandb] [--epochs N] [--batch N] [--imgsz N]

set -e  # Exit on error

echo "========================================================================"
echo "YOLOv11 Wildlife Detection - Setup and Training"
echo "Project: Guacamaya - Microsoft AI for Good Lab"
echo "========================================================================"

# Check Python version
echo ""
echo "Checking Python version..."
python3 --version

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check GPU availability
echo ""
echo "Checking GPU availability..."
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Setup wandb (if not using --no-wandb)
if [[ ! "$*" == *"--no-wandb"* ]]; then
    echo ""
    echo "Setting up Weights & Biases..."
    echo "If you haven't logged in to wandb yet, you'll be prompted for your API key"
    echo "Get your key from: https://wandb.ai/authorize"
    wandb login
fi

# Run training
echo ""
echo "========================================================================"
echo "Starting training..."
echo "========================================================================"
python3 train_yolov11_wildlife.py "$@"

echo ""
echo "========================================================================"
echo "Setup and training complete!"
echo "========================================================================"

