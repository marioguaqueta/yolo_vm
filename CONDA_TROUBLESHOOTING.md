# 🔧 Conda Environment Troubleshooting Guide

## Common Installation Issues and Solutions

### Issue 1: "pytorch-cuda is not installable" ❌

**Error message:**
```
Could not solve for environment specs
The following packages are incompatible
└─ pytorch-cuda 11.8** is not installable because there are no viable options
```

**Cause:** NVIDIA channel missing or CUDA packages not available

**Solutions (try in order):**

#### Solution A: Use the Simple Environment File (Recommended) ✅

This installs PyTorch via pip instead of conda, which is more reliable:

```bash
conda env create -f environment-simple.yml
```

This will:
- ✅ Install conda packages normally
- ✅ Install PyTorch via pip (auto-detects CUDA)
- ✅ Usually works without issues

#### Solution B: Update Original Environment File

Make sure `environment.yml` has the nvidia channel:

```yaml
channels:
  - pytorch
  - nvidia      ← Make sure this is here
  - conda-forge
  - defaults
```

Then try:
```bash
conda env create -f environment.yml
```

#### Solution C: Check Your CUDA Version

First, check what CUDA version you have:

```bash
nvidia-smi
# Look for "CUDA Version: X.X"
```

Then use the matching environment:

**CUDA 11.8:**
```bash
conda env create -f environment.yml
```

**CUDA 12.1:**
Edit `environment.yml` and change:
```yaml
- pytorch::pytorch-cuda=12.1  # Change from 11.8 to 12.1
```

**No GPU / CPU only:**
```bash
conda env create -f environment-cpu.yml
```

#### Solution D: Use Mamba (Faster Solver)

Mamba has a better dependency solver:

```bash
# Install mamba
conda install mamba -n base -c conda-forge

# Create environment with mamba
mamba env create -f environment.yml
```

#### Solution E: Manual Installation

Create empty environment and install packages step-by-step:

```bash
# Create empty environment
conda create -n yolov11-wildlife python=3.10

# Activate
conda activate yolov11-wildlife

# Install PyTorch (check https://pytorch.org for your system)
# For CUDA 11.8:
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# For CPU only:
conda install pytorch torchvision cpuonly -c pytorch

# Install other dependencies
conda install pandas numpy pillow opencv pyyaml matplotlib seaborn tqdm -c conda-forge

# Install via pip
pip install ultralytics wandb opencv-python
```

---

### Issue 2: "Solving environment: failed" ❌

**Error message:**
```
Solving environment: failed
```

**Solutions:**

#### Option A: Update Conda

```bash
conda update -n base conda
```

#### Option B: Use Libmamba Solver

```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

Then retry:
```bash
conda env create -f environment.yml
```

#### Option C: Use Simple Environment

```bash
conda env create -f environment-simple.yml
```

---

### Issue 3: Very Slow Environment Creation 🐌

**Solution: Use Mamba**

```bash
# Install mamba (much faster)
conda install mamba -n base -c conda-forge

# Use mamba instead of conda
mamba env create -f environment.yml
```

Mamba is typically 5-10x faster than conda for environment creation.

---

### Issue 4: Channel Errors ❌

**Error message:**
```
PackagesNotFoundError: The following packages are not available from current channels
```

**Solution: Add Required Channels**

```bash
conda config --add channels conda-forge
conda config --add channels pytorch
conda config --add channels nvidia

# Check channels
conda config --show channels
```

Then retry:
```bash
conda env create -f environment.yml
```

---

### Issue 5: GPU Not Detected After Installation ❌

**Check GPU availability:**

```bash
conda activate yolov11-wildlife
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

**If False, reinstall PyTorch:**

```bash
conda activate yolov11-wildlife

# Check your CUDA version first
nvidia-smi

# Install PyTorch with correct CUDA version
# For CUDA 11.8:
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1:
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

---

## Quick Fix: Recommended Approach 🚀

If you're having issues with `environment.yml`, use **environment-simple.yml** instead:

```bash
# Remove failed environment (if exists)
conda env remove -n yolov11-wildlife

# Use simple environment (PyTorch via pip)
conda env create -f environment-simple.yml

# Activate
conda activate yolov11-wildlife

# Verify installation
python -c "import torch; from ultralytics import YOLO; print('✓ Everything installed correctly')"

# Check GPU
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## Environment File Comparison

### environment.yml (Original)
- ✅ Everything via conda
- ✅ Good for reproducibility
- ❌ Can have dependency conflicts
- ❌ Requires nvidia channel

**Use when:** You want pure conda environment

### environment-simple.yml (Recommended)
- ✅ PyTorch via pip (more reliable)
- ✅ Fewer dependency issues
- ✅ Auto-detects CUDA version
- ✅ Faster to create

**Use when:** Having issues with original

### environment-cpu.yml (CPU Only)
- ✅ No GPU/CUDA required
- ✅ Works on any machine
- ❌ Very slow training

**Use when:** No GPU available or testing

---

## Step-by-Step Installation (Foolproof Method)

### Step 1: Choose Your Environment File

**Have GPU?**
```bash
FILE="environment-simple.yml"
```

**No GPU (CPU only)?**
```bash
FILE="environment-cpu.yml"
```

### Step 2: Create Environment

```bash
conda env create -f $FILE
```

**If this fails, try with mamba:**
```bash
conda install mamba -n base -c conda-forge
mamba env create -f $FILE
```

### Step 3: Activate and Verify

```bash
conda activate yolov11-wildlife

# Check Python
python --version

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check GPU (if you have one)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check YOLOv11
python -c "from ultralytics import YOLO; print('✓ YOLO installed')"

# Check wandb
python -c "import wandb; print('✓ Wandb installed')"
```

### Step 4: If Everything Works

```bash
# Run setup test
python test_setup.py

# Start training
python train_vm.py --epochs 50 --batch 8
```

---

## Still Having Issues?

### Last Resort: Use Pip/venv Instead

Conda having too many issues? Use pip instead:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# This usually works better for PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Train
python train_vm.py --epochs 50 --batch 8
```

See `requirements.txt` for pip-based installation.

---

## Quick Reference

### Problem: pytorch-cuda not installable
**Solution:** Use `environment-simple.yml`

### Problem: Solving environment failed
**Solution:** Use mamba: `mamba env create -f environment.yml`

### Problem: GPU not detected
**Solution:** Reinstall PyTorch with correct CUDA version

### Problem: Too slow
**Solution:** Use mamba instead of conda

### Problem: Everything fails
**Solution:** Use pip/venv with `requirements.txt`

---

## Get Your CUDA Version

```bash
# Check CUDA version
nvidia-smi

# Or
nvcc --version

# Or check from PyTorch
python -c "import torch; print(torch.version.cuda)"
```

---

## Recommended Commands

### For Most Users (GPU):
```bash
conda env create -f environment-simple.yml
conda activate yolov11-wildlife
python test_setup.py
```

### For CPU Only:
```bash
conda env create -f environment-cpu.yml
conda activate yolov11-wildlife
python test_setup.py
```

### If Conda Issues:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python test_setup.py
```

---

**Need more help?** Check:
- `CONDA_SETUP.md` - Complete conda guide
- `README_TRAINING.md` - Full documentation
- PyTorch installation guide: https://pytorch.org/get-started/locally/

