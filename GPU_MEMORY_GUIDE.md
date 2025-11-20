# 🎮 GPU Memory Management Guide

## Error: CUDA Out of Memory

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.
GPU 0 has a total capacity of 23.76 GiB of which 1.23 GiB is free.
```

This error occurs when training with images that are too large or batch sizes that are too high.

---

## 🚨 IMMEDIATE SOLUTION

### Your GPU: NVIDIA L40-24C (24GB)

For **2048px images** on **24GB GPU**, use these settings:

```bash
# RECOMMENDED: Batch size 4
python train_vm.py --epochs 50 --batch 4 --imgsz 2048

# SAFE: Batch size 2 (if batch 4 still fails)
python train_vm.py --epochs 50 --batch 2 --imgsz 2048

# ALTERNATIVE: Smaller images, larger batch
python train_vm.py --epochs 50 --batch 8 --imgsz 1024
```

---

## 📊 Memory Requirements (Approximate)

### Image Size vs Batch Size (24GB GPU)

| Image Size | Max Batch Size | Memory Used | Recommended |
|------------|----------------|-------------|-------------|
| 640px | 32 | ~8 GB | For testing |
| 1024px | 16 | ~12 GB | Balanced |
| 1280px | 12 | ~15 GB | Good quality |
| 1536px | 8 | ~18 GB | High quality |
| **2048px** | **4** | **~20 GB** | **Aerial images** ✅ |
| 2560px | 2 | ~22 GB | Maximum quality |

**For your 24GB L40 GPU with 2048px images: Use batch size 4 or less**

---

## 🔧 Solution Options (Choose One)

### Option 1: Reduce Batch Size (Recommended) ⭐

**Best for maintaining image quality**

```bash
# Try batch size 4 first
python train_vm.py --epochs 50 --batch 4 --imgsz 2048

# If still fails, use batch size 2
python train_vm.py --epochs 50 --batch 2 --imgsz 2048
```

**Pros:**
- ✅ Maintains high resolution (better for small animals)
- ✅ Better detection accuracy
- ✅ Good for aerial wildlife images

**Cons:**
- ⚠️ Slower training (fewer images per batch)
- ⚠️ May need more epochs to converge

### Option 2: Reduce Image Size

**Best for faster training**

```bash
# 1536px - good balance
python train_vm.py --epochs 50 --batch 8 --imgsz 1536

# 1024px - much faster
python train_vm.py --epochs 50 --batch 16 --imgsz 1024
```

**Pros:**
- ✅ Faster training
- ✅ More stable
- ✅ Can use larger batch size

**Cons:**
- ⚠️ Lower resolution (may miss small animals)
- ⚠️ Reduced accuracy for aerial images

### Option 3: Smaller Model

**If you still have memory issues**

```bash
# Use yolo11n (nano) instead of yolo11s (small)
python train_vm.py --epochs 50 --batch 8 --imgsz 2048

# Then edit train_vm.py:
# MODEL = "yolo11n.pt"  # Nano model (smaller)
```

**Model sizes:**
- `yolo11n.pt` - Nano (smallest, 2.6M params)
- `yolo11s.pt` - Small (current, 9.4M params) ⭐
- `yolo11m.pt` - Medium (larger, more memory)

---

## 🛠️ Before Starting Training

### 1. Check GPU Memory

```bash
# Check current GPU usage
nvidia-smi

# Watch GPU memory in real-time
watch -n 1 nvidia-smi
```

**Make sure GPU is idle before training** (should show ~1MB usage, not 21GB)

### 2. Clear Any Existing Processes

```bash
# Find processes using GPU
nvidia-smi

# If something is using GPU, kill it
# Find PID from nvidia-smi, then:
kill -9 PID_NUMBER
```

### 3. Clear GPU Cache

The updated script now does this automatically, but you can also do it manually:

```python
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"
```

---

## 🎯 Recommended Settings by GPU

### Your GPU: L40-24C (24GB) ⭐

```bash
# HIGH QUALITY (Recommended for wildlife)
python train_vm.py --epochs 50 --batch 4 --imgsz 2048

# BALANCED (Good speed/quality)
python train_vm.py --epochs 50 --batch 8 --imgsz 1536

# FAST TRAINING (Testing)
python train_vm.py --epochs 50 --batch 16 --imgsz 1024
```

### Other Common GPUs:

**RTX 3090 / A6000 (24GB):**
```bash
python train_vm.py --epochs 50 --batch 4 --imgsz 2048
```

**RTX 4090 (24GB):**
```bash
python train_vm.py --epochs 50 --batch 6 --imgsz 2048  # Slightly better
```

**V100 (16GB):**
```bash
python train_vm.py --epochs 50 --batch 2 --imgsz 2048
# or
python train_vm.py --epochs 50 --batch 6 --imgsz 1536
```

**T4 (16GB):**
```bash
python train_vm.py --epochs 50 --batch 2 --imgsz 2048
# or
python train_vm.py --epochs 50 --batch 4 --imgsz 1536
```

**A100 (40GB/80GB):**
```bash
python train_vm.py --epochs 50 --batch 8 --imgsz 2048  # Plenty of room
```

---

## 🚀 Optimized Training Commands

### For Your L40-24C GPU:

```bash
cd /home/estudiante/grupo_12/subsaharian_dataset/yolo_vm

# 1. Make sure GPU is free
nvidia-smi

# 2. Activate environment
conda activate yolov11-wildlife

# 3. Train with optimal settings
python train_vm.py --epochs 50 --batch 4 --imgsz 2048

# Optional: Set PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_vm.py --epochs 50 --batch 4 --imgsz 2048
```

---

## 🔍 Monitoring During Training

### Watch GPU Usage:

**Terminal 1** (Training):
```bash
python train_vm.py --epochs 50 --batch 4 --imgsz 2048
```

**Terminal 2** (Monitor):
```bash
watch -n 1 nvidia-smi
```

**What to look for:**
- Memory usage should be ~18-20 GB (for batch 4, 2048px)
- GPU utilization should be 80-100%
- Temperature should be reasonable (<85°C)

---

## ⚙️ Advanced Memory Optimization

### 1. Environment Variables

Add these before training:

```bash
# Memory fragmentation fix
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Train
python train_vm.py --epochs 50 --batch 4 --imgsz 2048
```

### 2. Gradient Accumulation (If batch size too small)

If you need batch size 2 but want the effect of batch size 8:

The script uses `batch=4` effectively like `batch=16` through gradient accumulation (handled by Ultralytics automatically).

### 3. Mixed Precision Training

Already enabled in the script (`amp=True`). This saves ~30-40% memory.

---

## 📈 Training Time Estimates

### L40-24C GPU (your setup):

| Image Size | Batch | Time/Epoch | 50 Epochs |
|------------|-------|------------|-----------|
| 2048px | 2 | ~15 min | ~12.5 hours |
| 2048px | 4 | ~12 min | ~10 hours ✅ |
| 1536px | 8 | ~8 min | ~6.5 hours |
| 1024px | 16 | ~5 min | ~4 hours |

**Recommended:** `batch=4, imgsz=2048` (~10 hours for 50 epochs)

---

## 🐛 Troubleshooting

### Error Persists with Batch 4?

```bash
# Try batch 2
python train_vm.py --epochs 50 --batch 2 --imgsz 2048

# Or reduce image size slightly
python train_vm.py --epochs 50 --batch 4 --imgsz 1792
```

### GPU Memory Not Clearing?

```bash
# Restart Python kernel
pkill -9 python

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Worst case: restart training
```

### Other Process Using GPU?

```bash
# Check what's using GPU
nvidia-smi

# Find and kill the process
nvidia-smi | grep python
kill -9 PID
```

### Still Out of Memory?

```bash
# Last resort: smallest settings that will work
python train_vm.py --epochs 50 --batch 2 --imgsz 1536

# Or use nano model
# Edit train_vm.py: MODEL = "yolo11n.pt"
python train_vm.py --epochs 50 --batch 4 --imgsz 2048
```

---

## ✅ Quick Reference

### Your Situation (L40-24C, 24GB):

**Problem:** batch=8, imgsz=2048 → Out of Memory ❌

**Solution:** batch=4, imgsz=2048 → Works ✅

**Command:**
```bash
python train_vm.py --epochs 50 --batch 4 --imgsz 2048
```

**Expected Memory:** ~18-20 GB used  
**Training Time:** ~10 hours for 50 epochs  
**Quality:** Excellent for aerial wildlife  

---

## 📊 Memory Calculator

**Formula:** 
```
Memory ≈ (Image_Size / 1024)² × Batch_Size × 0.5 GB
```

**Examples:**
- 2048px, batch 4: (2048/1024)² × 4 × 0.5 = 8 GB (model+overhead ~18 GB total)
- 2048px, batch 8: (2048/1024)² × 8 × 0.5 = 16 GB (model+overhead ~26 GB total) ❌
- 1536px, batch 8: (1536/1024)² × 8 × 0.5 = 9 GB (model+overhead ~19 GB total) ✅

---

## 🎓 Best Practices

1. **Start conservative:**
   ```bash
   python train_vm.py --epochs 5 --batch 2 --imgsz 2048
   ```
   Watch memory usage, then increase batch size if there's room.

2. **Monitor first epoch:**
   The first epoch will show peak memory usage. If it works, the rest will too.

3. **Keep GPU clean:**
   Make sure no other processes are using GPU before training.

4. **Use wandb:**
   Track memory usage across runs to find optimal settings.

5. **Save early, save often:**
   The script saves every 5 epochs automatically.

---

## 🎯 Summary

**For your L40-24C GPU (24GB) training aerial wildlife images:**

✅ **Use these settings:**
```bash
python train_vm.py --epochs 50 --batch 4 --imgsz 2048
```

❌ **Don't use:**
```bash
python train_vm.py --epochs 50 --batch 8 --imgsz 2048  # Too much memory
```

**Expected results:**
- Memory usage: ~18-20 GB
- Training time: ~10 hours
- Quality: Excellent for small animal detection

**The script has been updated with these optimal defaults!** 🚀

