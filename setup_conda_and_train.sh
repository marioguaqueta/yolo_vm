#!/bin/bash
# Conda Setup and Train Script for YOLOv11 Wildlife Detection
# Usage: ./setup_conda_and_train.sh [--no-wandb] [--epochs N] [--batch N] [--imgsz N]

set -e  # Exit on error

echo "========================================================================"
echo "YOLOv11 Wildlife Detection - Conda Setup and Training"
echo "Project: Guacamaya - Microsoft AI for Good Lab"
echo "========================================================================"

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo ""
    echo "❌ ERROR: Conda is not installed or not in PATH"
    echo ""
    echo "Please install Miniconda or Anaconda:"
    echo "  - Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    echo "  - Anaconda: https://www.anaconda.com/download"
    exit 1
fi

echo ""
echo "✓ Conda found: $(conda --version)"

# Check if environment already exists
ENV_NAME="yolov11-wildlife"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Using existing environment..."
    fi
fi

# Create or update conda environment
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "Creating conda environment from environment.yml..."
    conda env create -f environment.yml
else
    echo ""
    echo "Updating conda environment..."
    conda env update -f environment.yml --prune
fi

# Activate environment
echo ""
echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# Verify activation
if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo ""
    echo "❌ ERROR: Failed to activate conda environment"
    echo "Please activate manually: conda activate ${ENV_NAME}"
    exit 1
fi

echo "✓ Environment activated: ${ENV_NAME}"

# Check GPU availability
echo ""
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

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
python train_yolov11_wildlife.py "$@"

echo ""
echo "========================================================================"
echo "Setup and training complete!"
echo "========================================================================"
echo ""
echo "To use this environment in the future:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To deactivate:"
echo "  conda deactivate"

