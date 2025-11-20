#!/bin/bash
# Emergency Disk Space Cleanup and Setup Script
# For VMs with limited disk space

set -e

echo "========================================================================"
echo "Emergency Disk Space Cleanup and Setup"
echo "========================================================================"
echo ""

# Check current space
echo "Current disk usage:"
df -h | grep -E "Filesystem|/$|/home"
echo ""

# Get available space in GB
AVAILABLE=$(df -h ~ | tail -1 | awk '{print $4}' | sed 's/G//')

echo "Available space in home: ${AVAILABLE}GB"
echo ""

if [ "${AVAILABLE%.*}" -lt 15 ]; then
    echo "⚠️  WARNING: Less than 15GB available"
    echo "This may not be enough for conda installation"
    echo ""
fi

read -p "Do you want to clean up disk space? (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Cleaning disk space..."
    echo ""
    
    # Clean conda
    echo "1. Cleaning conda cache..."
    conda clean --all -y 2>/dev/null || echo "  (conda not found or already clean)"
    
    # Remove conda package tarballs
    echo "2. Removing conda package archives..."
    rm -rf ~/miniconda/pkgs/*.tar.bz2 2>/dev/null || true
    rm -rf ~/anaconda3/pkgs/*.tar.bz2 2>/dev/null || true
    
    # Clean pip cache
    echo "3. Cleaning pip cache..."
    rm -rf ~/.cache/pip 2>/dev/null || true
    pip cache purge 2>/dev/null || echo "  (pip cache already clean)"
    
    # Clean torch cache
    echo "4. Cleaning PyTorch cache..."
    rm -rf ~/.cache/torch 2>/dev/null || true
    
    # Clean wandb cache
    echo "5. Cleaning wandb cache..."
    rm -rf ~/.cache/wandb 2>/dev/null || true
    
    # Clean temp files
    echo "6. Cleaning temporary files..."
    rm -rf /tmp/pip-* 2>/dev/null || true
    
    echo ""
    echo "✓ Cleanup complete!"
    echo ""
    
    # Check space again
    echo "New disk usage:"
    df -h | grep -E "Filesystem|/$|/home"
    echo ""
fi

# Ask installation method
echo "========================================================================"
echo "Choose Installation Method:"
echo "========================================================================"
echo ""
echo "Conda environments require ~8-10 GB"
echo "Pip virtual environments require ~3 GB (RECOMMENDED for limited space)"
echo ""
echo "1) Pip/venv (Lightweight - 3GB)"
echo "2) Conda (Full features - 8-10GB)"
echo "3) Exit"
echo ""

read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "Installing with pip (lightweight)..."
        echo ""
        
        cd /home/estudiantes/grupo_12/sahariandataset/yolo_vm
        
        # Create venv
        echo "Creating virtual environment..."
        python3 -m venv venv
        
        # Activate
        source venv/bin/activate
        
        # Upgrade pip
        echo "Upgrading pip..."
        pip install --upgrade pip
        
        # Install dependencies
        echo "Installing dependencies (this may take 5-10 minutes)..."
        pip install -r requirements.txt
        
        echo ""
        echo "========================================================================"
        echo "✓ Installation complete!"
        echo "========================================================================"
        echo ""
        echo "To use the environment:"
        echo "  source venv/bin/activate"
        echo ""
        echo "To verify:"
        echo "  python test_setup.py"
        echo ""
        echo "To train:"
        echo "  python train_vm.py --epochs 50 --batch 8"
        echo ""
        ;;
    
    2)
        echo ""
        echo "Installing with conda..."
        echo ""
        
        # Check if we have enough space
        AVAILABLE=$(df -h ~ | tail -1 | awk '{print $4}' | sed 's/G//')
        if [ "${AVAILABLE%.*}" -lt 12 ]; then
            echo "⚠️  WARNING: Only ${AVAILABLE}GB available"
            echo "Conda installation requires at least 12GB"
            echo ""
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Exiting... Please use pip installation (option 1)"
                exit 0
            fi
        fi
        
        cd /home/estudiantes/grupo_12/sahariandataset/yolo_vm
        
        # Try simple environment first
        echo "Creating conda environment..."
        if conda env create -f environment-simple.yml; then
            echo ""
            echo "========================================================================"
            echo "✓ Installation complete!"
            echo "========================================================================"
            echo ""
            echo "To use the environment:"
            echo "  conda activate yolov11-wildlife"
            echo ""
            echo "To verify:"
            echo "  python test_setup.py"
            echo ""
            echo "To train:"
            echo "  python train_vm.py --epochs 50 --batch 8"
            echo ""
        else
            echo ""
            echo "❌ Conda installation failed (likely due to space)"
            echo ""
            echo "Please use pip installation instead:"
            echo "  ./cleanup_and_setup.sh"
            echo "  (Choose option 1)"
            exit 1
        fi
        ;;
    
    3)
        echo "Exiting..."
        exit 0
        ;;
    
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# Final disk check
echo ""
echo "Final disk usage:"
df -h | grep -E "Filesystem|/$|/home"
echo ""
echo "If you need more space, see: DISK_SPACE_FIX.md"


