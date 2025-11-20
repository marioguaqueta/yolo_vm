#!/bin/bash
# Interactive Environment Creation Script
# Helps you choose the right environment file for your system

set -e

echo "========================================================================"
echo "YOLOv11 Wildlife Detection - Environment Setup"
echo "========================================================================"
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ ERROR: Conda is not installed"
    echo ""
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    echo "Or use pip instead:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

echo "✓ Conda found: $(conda --version)"
echo ""

# Check for GPU
echo "Checking for GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    HAS_GPU=true
else
    echo "⚠ No NVIDIA GPU detected (will use CPU)"
    HAS_GPU=false
fi

echo ""
echo "========================================================================"
echo "Choose Installation Method:"
echo "========================================================================"
echo ""
echo "1) Simple (Recommended) - PyTorch via pip, auto-detects CUDA"
echo "2) Full Conda - All packages via conda (may have conflicts)"
echo "3) CPU Only - No GPU required"
echo "4) Manual - I'll install packages myself"
echo "5) Exit"
echo ""

read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        ENV_FILE="environment-simple.yml"
        echo ""
        echo "Using: $ENV_FILE (PyTorch via pip)"
        ;;
    2)
        ENV_FILE="environment.yml"
        echo ""
        echo "Using: $ENV_FILE (Full conda)"
        echo ""
        echo "⚠ Note: If this fails, try option 1 (Simple) instead"
        ;;
    3)
        ENV_FILE="environment-cpu.yml"
        echo ""
        echo "Using: $ENV_FILE (CPU only)"
        echo ""
        echo "⚠ Warning: Training will be VERY slow on CPU (30-40 hours)"
        ;;
    4)
        echo ""
        echo "Manual Installation Instructions:"
        echo ""
        echo "# Create environment"
        echo "conda create -n yolov11-wildlife python=3.10"
        echo ""
        echo "# Activate"
        echo "conda activate yolov11-wildlife"
        echo ""
        echo "# Install PyTorch (visit https://pytorch.org for your system)"
        echo "conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia"
        echo ""
        echo "# Install other dependencies"
        echo "pip install ultralytics wandb pandas pillow opencv-python pyyaml matplotlib seaborn tqdm"
        echo ""
        exit 0
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# Check if environment already exists
ENV_NAME="yolov11-wildlife"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "⚠ Environment '$ENV_NAME' already exists"
    read -p "Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Keeping existing environment. To update:"
        echo "  conda env update -f $ENV_FILE --prune"
        exit 0
    fi
fi

# Try to create environment
echo ""
echo "========================================================================"
echo "Creating conda environment..."
echo "========================================================================"
echo ""
echo "This may take 5-10 minutes..."
echo ""

if conda env create -f $ENV_FILE; then
    echo ""
    echo "========================================================================"
    echo "✓ Environment created successfully!"
    echo "========================================================================"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Activate environment:"
    echo "   conda activate yolov11-wildlife"
    echo ""
    echo "2. Verify installation:"
    echo "   python test_setup.py"
    echo ""
    echo "3. Start training:"
    echo "   python train_vm.py --epochs 50 --batch 8"
    echo ""
else
    echo ""
    echo "========================================================================"
    echo "❌ Environment creation failed"
    echo "========================================================================"
    echo ""
    echo "Try these solutions:"
    echo ""
    echo "1. Use Mamba (faster solver):"
    echo "   conda install mamba -n base -c conda-forge"
    echo "   mamba env create -f $ENV_FILE"
    echo ""
    echo "2. Try Simple environment (if not already tried):"
    echo "   ./create_environment.sh"
    echo "   (Choose option 1)"
    echo ""
    echo "3. Use pip instead:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    echo ""
    echo "4. See troubleshooting guide:"
    echo "   cat CONDA_TROUBLESHOOTING.md"
    echo ""
    exit 1
fi

