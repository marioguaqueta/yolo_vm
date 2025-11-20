# 🐍 Conda Environment Setup Guide

This guide explains how to set up the YOLOv11 Wildlife Detection project using **Conda** for environment management.

## Why Use Conda?

- ✅ Better dependency management for scientific packages
- ✅ Handles both Python and system-level dependencies
- ✅ Easier GPU/CUDA configuration
- ✅ Isolated environments prevent conflicts
- ✅ Works consistently across Mac, Linux, and Windows

## Prerequisites

### Install Conda (if not already installed)

Choose one:

**Option 1: Miniconda (Recommended - Lightweight)**
- Download: https://docs.conda.io/en/latest/miniconda.html
- Install for your OS (Mac/Linux/Windows)

**Option 2: Anaconda (Full Distribution)**
- Download: https://www.anaconda.com/download
- Includes more pre-installed packages

Verify installation:
```bash
conda --version
# Should output: conda 23.x.x or similar
```

## 🚀 Quick Start with Conda

### Method 1: One-Command Setup (Recommended)

```bash
./setup_conda_and_train.sh
```

This script will:
1. ✅ Create conda environment from `environment.yml`
2. ✅ Install all dependencies (PyTorch, YOLOv11, wandb, etc.)
3. ✅ Configure GPU/CUDA if available
4. ✅ Setup wandb
5. ✅ Start training

### Method 2: Manual Setup

#### Step 1: Create Environment

```bash
# Create environment from environment.yml
conda env create -f environment.yml
```

This creates an environment named `yolov11-wildlife` with all dependencies.

#### Step 2: Activate Environment

```bash
conda activate yolov11-wildlife
```

#### Step 3: Verify Installation

```bash
# Check Python version
python --version

# Check PyTorch and GPU
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# Check Ultralytics YOLO
python -c "from ultralytics import YOLO; print('YOLOv11 installed successfully')"
```

#### Step 4: Login to Wandb (Optional)

```bash
wandb login
```

Get your API key from: https://wandb.ai/authorize

#### Step 5: Start Training

```bash
python train_yolov11_wildlife.py
```

## 📝 Environment Configuration

The `environment.yml` file includes:

```yaml
name: yolov11-wildlife

dependencies:
  - python=3.10
  - pytorch>=2.0.0
  - torchvision>=0.15.0
  - pytorch-cuda=11.8  # For GPU support
  - pandas
  - numpy
  - pillow
  - opencv
  - matplotlib
  - seaborn
  - pyyaml
  - tqdm
  
  - pip:
    - ultralytics>=8.0.0
    - wandb>=0.15.0
```

### GPU vs CPU Configuration

**For GPU (CUDA 11.8):**
The default `environment.yml` includes:
```yaml
- pytorch-cuda=11.8
```

**For Different CUDA Version:**
Check your CUDA version:
```bash
nvidia-smi  # Look for "CUDA Version"
```

Edit `environment.yml` and change:
```yaml
- pytorch-cuda=12.1  # For CUDA 12.1
# or
- pytorch-cuda=11.7  # For CUDA 11.7
```

**For CPU Only:**
Edit `environment.yml` and replace:
```yaml
- pytorch>=2.0.0
- torchvision>=0.15.0
- pytorch-cuda=11.8
```

With:
```yaml
- pytorch>=2.0.0
- torchvision>=0.15.0
- cpuonly
```

## 🔧 Managing the Environment

### List All Environments
```bash
conda env list
```

### Activate Environment
```bash
conda activate yolov11-wildlife
```

### Deactivate Environment
```bash
conda deactivate
```

### Update Environment
If you modify `environment.yml`:
```bash
conda env update -f environment.yml --prune
```

### Export Environment
Save your exact environment:
```bash
conda env export > environment_exact.yml
```

### Remove Environment
```bash
conda env remove -n yolov11-wildlife
```

## 📦 Installing Additional Packages

### Via Conda
```bash
conda activate yolov11-wildlife
conda install package-name
```

### Via Pip (in conda environment)
```bash
conda activate yolov11-wildlife
pip install package-name
```

**Tip:** Prefer conda for system-level packages (NumPy, SciPy) and pip for Python-only packages.

## 🎯 Training with Conda

Once environment is activated:

### Basic Training
```bash
conda activate yolov11-wildlife
python train_yolov11_wildlife.py
```

### Custom Parameters
```bash
conda activate yolov11-wildlife
python train_yolov11_wildlife.py --epochs 100 --batch 8 --imgsz 2048
```

### Using the Setup Script
```bash
./setup_conda_and_train.sh --epochs 50 --batch 8
```

## 🐛 Troubleshooting

### Issue: "conda: command not found"

**Solution:** Add conda to PATH or restart terminal

Mac/Linux:
```bash
export PATH="$HOME/miniconda3/bin:$PATH"
# or
source ~/.bashrc
```

Windows: Restart Anaconda Prompt

### Issue: "Solving environment: failed"

**Solution 1:** Update conda
```bash
conda update -n base conda
```

**Solution 2:** Use mamba (faster solver)
```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

### Issue: "PackagesNotFoundError"

**Solution:** Check channel priorities
```bash
conda config --show channels
conda config --add channels conda-forge
conda config --add channels pytorch
```

### Issue: GPU not detected after installation

**Solution:** Verify CUDA installation
```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch with correct CUDA version
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Issue: Environment creation is slow

**Solution:** Use mamba instead
```bash
# Install mamba
conda install mamba -n base -c conda-forge

# Create environment with mamba (much faster)
mamba env create -f environment.yml
```

## 🔄 Conda vs Pip Comparison

| Feature | Conda | Pip (venv) |
|---------|-------|------------|
| Speed | Slower (dependency solving) | Faster |
| GPU/CUDA | Easy (automatic) | Manual |
| Scientific packages | Better (NumPy, SciPy) | Good |
| System libraries | Handles | Requires system install |
| Environment isolation | Excellent | Excellent |
| Cross-platform | Excellent | Excellent |

**Recommendation:** Use Conda if you:
- Need GPU support (easier CUDA setup)
- Use scientific packages heavily
- Want system-level dependency management

Use Pip/venv if you:
- Prefer lightweight setup
- Already have CUDA configured
- Want faster environment creation

## 📚 Additional Resources

### Conda Documentation
- Official docs: https://docs.conda.io/
- Cheat sheet: https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html

### Environment Management
- Managing environments: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
- Managing packages: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html

### PyTorch Installation
- PyTorch with conda: https://pytorch.org/get-started/locally/

## 🎓 Best Practices

1. **Always activate before working:**
   ```bash
   conda activate yolov11-wildlife
   ```

2. **Keep environment.yml updated:**
   - Add new dependencies to the file
   - Version control it (already in your project)

3. **Use environment files for reproducibility:**
   - Share `environment.yml` with team
   - Everyone gets same setup

4. **Separate environments for different projects:**
   ```bash
   conda env list  # See all your environments
   ```

5. **Regular updates:**
   ```bash
   conda update --all
   ```

## ✅ Quick Reference

```bash
# Create environment
conda env create -f environment.yml

# Activate
conda activate yolov11-wildlife

# Check installation
python -c "import torch; from ultralytics import YOLO; print('✓ Ready')"

# Train model
python train_yolov11_wildlife.py

# Deactivate
conda deactivate

# Remove environment
conda env remove -n yolov11-wildlife
```

---

## 🚀 Ready to Start?

```bash
# All-in-one command:
./setup_conda_and_train.sh

# Or step by step:
conda env create -f environment.yml
conda activate yolov11-wildlife
python train_yolov11_wildlife.py
```

---

**For more information, see:**
- `README_TRAINING.md` - Complete training guide
- `QUICKSTART.md` - Quick start for all methods
- `USAGE_GUIDE.txt` - Detailed usage instructions

**Happy training with Conda! 🐍🦒🐘**

