# 💾 Disk Space Issue - Solutions

## Error: "No space left on device"

```
InvalidArchiveError: [Errno 28] No space left on device
```

This means your VM has run out of disk space during conda installation.

---

## 🚨 IMMEDIATE SOLUTIONS

### Solution 1: Clean Conda Cache and Use Pip (Recommended) ⭐

Conda environments are large. Using pip is much more space-efficient.

```bash
# Clean conda cache
conda clean --all -y

# Use pip instead (much smaller footprint)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Verify
python test_setup.py
```

**Space comparison:**
- Conda environment: ~5-8 GB
- Pip venv: ~2-3 GB

### Solution 2: Clean Conda and Try Minimal Environment

```bash
# Clean conda thoroughly
conda clean --all -y
rm -rf ~/miniconda/pkgs/*

# Check space
df -h

# Use simple environment (if you have enough space now)
conda env create -f environment-simple.yml
```

### Solution 3: Check and Free Space

```bash
# Check current disk usage
df -h

# Check home directory usage
du -sh ~/*

# Clean common space hogs
conda clean --all -y
rm -rf ~/.cache/pip
rm -rf ~/.cache/torch
rm -rf ~/miniconda/pkgs/*

# Check space again
df -h
```

---

## 📊 Check Your Disk Space

```bash
# Overall disk usage
df -h

# Home directory usage (detailed)
du -sh ~/* | sort -h

# Find large files
find ~ -type f -size +100M -exec ls -lh {} \; 2>/dev/null

# Conda package cache size
du -sh ~/miniconda/pkgs/
```

---

## 🧹 Clean Up Space

### Step 1: Clean Conda

```bash
# Remove unused packages and caches
conda clean --all -y

# Remove package tarballs
rm -rf ~/miniconda/pkgs/*.tar.bz2

# Check conda environments
conda env list

# Remove unused environments (if any)
conda env remove -n unused_env_name
```

### Step 2: Clean System Caches

```bash
# Clear pip cache
rm -rf ~/.cache/pip

# Clear torch cache
rm -rf ~/.cache/torch

# Clear wandb cache (if exists)
rm -rf ~/.cache/wandb

# Clear temporary files
rm -rf /tmp/*
```

### Step 3: Clean Project Files

```bash
# Remove zip files if already extracted
cd /home/estudiantes/grupo_12/sahariandataset/general_dataset
rm -f *.zip

# Clear any old training runs
cd /home/estudiantes/grupo_12/sahariandataset/yolo_vm
rm -rf runs/old_*  # If you have old runs
```

---

## ✅ RECOMMENDED APPROACH: Use Pip Instead

Since space is limited, **pip is strongly recommended** over conda:

```bash
cd /home/estudiantes/grupo_12/sahariandataset/yolo_vm

# Clean conda completely
conda clean --all -y

# Create lightweight virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies (much smaller)
pip install -r requirements.txt

# Verify installation
python test_setup.py

# Check it works
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Start training
python train_vm.py --epochs 50 --batch 8
```

**Benefits of pip:**
- ✅ 50-60% less disk space
- ✅ Faster installation
- ✅ Auto-detects CUDA
- ✅ Fewer conflicts

---

## 🔍 Troubleshooting Disk Space

### If you still get "No space" errors:

#### Check where space is used:

```bash
# Top 20 largest directories in home
du -ah ~ | sort -rh | head -20

# Conda-specific usage
du -sh ~/miniconda
du -sh ~/miniconda/pkgs
du -sh ~/miniconda/envs
```

#### Clean aggressively:

```bash
# Remove ALL conda package cache
rm -rf ~/miniconda/pkgs/*

# Remove conda environment (if partially created)
conda env remove -n yolov11-wildlife -y

# Clean pip completely
pip cache purge
```

#### Contact VM administrator:

```bash
# Check quota (if applicable)
quota -v

# Show disk usage
df -h
```

You may need to request more disk space from your VM administrator.

---

## 📦 Space Requirements

### Conda Installation:
- Python + PyTorch + CUDA: ~5 GB
- Other packages: ~1-2 GB
- Cache/temp files: ~1-2 GB
- **Total: ~8-10 GB**

### Pip Installation:
- Python venv: ~200 MB
- PyTorch + CUDA: ~2 GB
- Other packages: ~500 MB
- Cache/temp files: ~300 MB
- **Total: ~3 GB**

### Training:
- Model checkpoints: ~100-500 MB
- YOLO dataset conversion: ~2-3 GB
- Logs and results: ~100-200 MB
- **Total: ~3-4 GB**

**Recommended minimum:** 15 GB free space
**Comfortable amount:** 20+ GB free space

---

## 🚀 Quick Fix Script

Create this script to clean and setup:

```bash
#!/bin/bash
# clean_and_setup.sh

echo "Cleaning up disk space..."

# Clean conda
conda clean --all -y 2>/dev/null || true
rm -rf ~/miniconda/pkgs/*.tar.bz2 2>/dev/null || true

# Clean caches
rm -rf ~/.cache/pip 2>/dev/null || true
rm -rf ~/.cache/torch 2>/dev/null || true

# Check space
echo ""
echo "Available space:"
df -h | grep -E "Filesystem|/$|/home"

echo ""
echo "If you have enough space (15+ GB free), press Enter to continue..."
read

echo ""
echo "Creating lightweight environment with pip..."

cd /home/estudiantes/grupo_12/sahariandataset/yolo_vm

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

echo ""
echo "✓ Installation complete!"
echo ""
echo "To activate: source venv/bin/activate"
echo "To train: python train_vm.py --epochs 50 --batch 8"
```

---

## 🔄 Complete Workflow

### Step 1: Check Space
```bash
df -h
# Need at least 15 GB free
```

### Step 2: Clean Up
```bash
conda clean --all -y
rm -rf ~/.cache/pip ~/.cache/torch
rm -rf ~/miniconda/pkgs/*
```

### Step 3: Verify Space
```bash
df -h
# Should have freed up several GB
```

### Step 4: Install with Pip
```bash
cd /home/estudiantes/grupo_12/sahariandataset/yolo_vm
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 5: Verify
```bash
python test_setup.py
```

### Step 6: Train
```bash
python train_vm.py --epochs 50 --batch 8
```

---

## ⚠️ If Nothing Works

If you still don't have enough space:

### Option 1: Use Different Disk
```bash
# Check all mounted disks
df -h

# If there's a larger disk (e.g., /data, /scratch)
# Create venv there
python3 -m venv /data/grupo_12/venv
source /data/grupo_12/venv/bin/activate
```

### Option 2: Install Only Essential Packages
```bash
python3 -m venv venv
source venv/bin/activate

# Install minimal packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics wandb pandas pillow

# Skip optional packages
```

### Option 3: Request More Space
Contact your VM administrator:
```
Subject: Disk Space Request

Hello,

I'm running out of disk space on the VM at:
/home/estudiantes/grupo_12/

Current usage: [insert df -h output]
Required: ~20 GB for deep learning environment

Could you please increase the quota or suggest an alternative location?

Thank you!
```

---

## 📝 Prevention for Future

To avoid this issue:

1. **Always check space before installing:**
   ```bash
   df -h
   ```

2. **Use pip instead of conda when space-limited**

3. **Clean caches regularly:**
   ```bash
   conda clean --all -y
   pip cache purge
   ```

4. **Monitor space during training:**
   ```bash
   watch -n 60 df -h
   ```

5. **Set up automatic cleanup:**
   ```bash
   # Add to crontab
   0 0 * * 0 conda clean --all -y
   ```

---

## ✅ Summary

**Problem:** VM ran out of disk space during conda installation

**Best Solution:** Use pip instead of conda (saves ~5 GB)

**Commands:**
```bash
# Clean
conda clean --all -y
rm -rf ~/.cache/*

# Install with pip
cd /home/estudiantes/grupo_12/sahariandataset/yolo_vm
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train
python train_vm.py --epochs 50 --batch 8
```

**Space needed:**
- Conda: ~10 GB
- Pip: ~3 GB ✅

**Use pip!** 🚀


