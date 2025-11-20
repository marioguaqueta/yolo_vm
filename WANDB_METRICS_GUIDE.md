# 📊 Wandb Metrics Guide for YOLOv11 Training

## Automatic Metrics Tracking

Ultralytics YOLO **automatically logs comprehensive metrics** to Wandb when training. No additional configuration needed!

---

## 🎯 Key Metrics Tracked

### 1. **Training Losses** (Lower is Better)

| Metric | Description | Good Value |
|--------|-------------|------------|
| `train/box_loss` | Bounding box localization loss | < 1.0 |
| `train/cls_loss` | Classification loss | < 0.5 |
| `train/dfl_loss` | Distribution Focal Loss | < 1.0 |
| `train/loss` | **Total training loss** | Decreasing |

**What to watch:** All losses should **decrease over time**. If they plateau or increase, training may have issues.

### 2. **Validation Metrics** (Higher is Better)

| Metric | Description | Target Value |
|--------|-------------|--------------|
| `metrics/mAP50(B)` | **Main metric: mAP @ IoU 0.5** | > 0.60 |
| `metrics/mAP50-95(B)` | mAP averaged over IoU 0.5-0.95 | > 0.40 |
| `metrics/precision(B)` | Precision (true positives) | > 0.70 |
| `metrics/recall(B)` | Recall (detection rate) | > 0.60 |

**What to watch:** 
- **mAP50** is your primary metric (target: >60% for wildlife)
- **Precision** = How many detections are correct
- **Recall** = How many animals you detect

### 3. **Per-Class Performance** (Higher is Better)

For each of your 6 species, you'll see:

| Metric | Species | Description |
|--------|---------|-------------|
| `metrics/mAP50(B)_Buffalo` | Buffalo | mAP50 for buffalo |
| `metrics/mAP50(B)_Elephant` | Elephant | mAP50 for elephants |
| `metrics/mAP50(B)_Kudu` | Kudu | mAP50 for kudus |
| `metrics/mAP50(B)_Topi` | Topi | mAP50 for topis |
| `metrics/mAP50(B)_Warthog` | Warthog | mAP50 for warthogs |
| `metrics/mAP50(B)_Waterbuck` | Waterbuck | mAP50 for waterbucks |

**Expected values** (based on your notebook):
- Elephant: ~80% (best)
- Kudu: ~77% (excellent)
- Buffalo: ~65% (good)
- Waterbuck: ~40% (fair)
- Warthog: ~29% (challenging - few samples)

### 4. **Learning Rate**

| Metric | Description |
|--------|-------------|
| `lr/pg0` | Learning rate for parameter group 0 |
| `lr/pg1` | Learning rate for parameter group 1 |
| `lr/pg2` | Learning rate for parameter group 2 |

**What to watch:** Learning rate should **decrease gradually** (cosine/linear schedule).

### 5. **System Metrics**

| Metric | Description | Good Value |
|--------|-------------|------------|
| `train/epoch` | Current epoch | Increases |
| `train/time` | Time per epoch | Consistent |
| `system/gpu_mem` | GPU memory usage | < 23 GB (your L40) |
| `system/gpu_util` | GPU utilization | 80-100% |

---

## 📈 Wandb Dashboard Panels

### Main Dashboard View

When you open your wandb run, you'll see:

#### 1. **Overview Panel**
- Run name, status, duration
- Final mAP50 score
- Hardware info (GPU, CUDA version)

#### 2. **Charts Panel** (Most Important!)

**Loss Curves:**
```
train/box_loss     →  Should decrease steadily
train/cls_loss     →  Should decrease steadily  
train/dfl_loss     →  Should decrease steadily
```

**Validation Metrics:**
```
metrics/mAP50(B)       →  Should increase (target: >60%)
metrics/mAP50-95(B)    →  Should increase
metrics/precision(B)   →  Should increase
metrics/recall(B)      →  Should increase
```

**Per-Class mAP50:**
```
metrics/mAP50(B)_Elephant   →  Best performer (~80%)
metrics/mAP50(B)_Kudu       →  Second best (~77%)
metrics/mAP50(B)_Buffalo    →  Good (~65%)
...
```

#### 3. **System Panel**
```
system/gpu_mem     →  Memory usage (should be ~18-20 GB)
system/gpu_util    →  Utilization (should be 80-100%)
lr/pg0             →  Learning rate (decreases over time)
```

#### 4. **Images Panel**
- Training batch samples
- Validation predictions
- Ground truth vs predictions
- Confusion matrix (at end)

#### 5. **Model Panel**
- Model architecture summary
- Parameter count
- FLOPs

---

## 🎯 What to Monitor During Training

### Every 5 Epochs (Real-time)

**Check these metrics:**

1. **`metrics/mAP50(B)`** - Your main success metric
   - Epoch 10: ~20-30%
   - Epoch 25: ~40-50%
   - Epoch 50: ~60-70% (target)

2. **`train/box_loss`** - Should decrease smoothly
   - Start: ~1.5-2.0
   - End: ~0.5-1.0

3. **`metrics/precision(B)` and `metrics/recall(B)`**
   - Both should increase
   - Balance is important (not just one high)

4. **`system/gpu_mem`** - Should be stable
   - Your setup: ~18-20 GB (out of 24 GB)
   - If it spikes to 24 GB → OOM risk

### Red Flags to Watch For

❌ **Loss not decreasing** after epoch 10
   - May need to adjust learning rate
   - Check data augmentation

❌ **mAP50 stuck below 30%** after epoch 20
   - Possible data quality issues
   - Check label accuracy

❌ **GPU memory slowly increasing**
   - Memory leak (rare with Ultralytics)
   - Restart training if it happens

❌ **One class mAP = 0%**
   - No samples for that class, or
   - Labels might be wrong

---

## 📊 Key Wandb Features to Use

### 1. **Line Plots** (Automatic)

Compare metrics across epochs:
```
X-axis: Epoch
Y-axis: Metric value
Lines: Different metrics
```

**Best plots to create:**
- All losses together (train/box_loss, train/cls_loss, train/dfl_loss)
- All mAP metrics (mAP50, mAP50-95, precision, recall)
- Per-class mAP50 (all 6 species)

### 2. **Compare Runs**

If you train multiple times with different settings:
```
Wandb → Select multiple runs → Compare
```

Compare:
- Different batch sizes
- Different image sizes
- Different models (yolo11s vs yolo11m)

### 3. **Custom Panels**

Create custom views:
```python
# These are logged automatically, but you can create custom panels:
- Training progress: epoch vs mAP50
- GPU efficiency: gpu_util vs time
- Class performance: bar chart of per-class mAP50
```

### 4. **Download Data**

Export metrics to CSV:
```
Wandb → Run page → Overview → Export data → CSV
```

### 5. **Alerts**

Set up alerts for:
- Training completion
- mAP50 reaches target (e.g., >60%)
- GPU memory too high
- Training failure

---

## 🎨 Recommended Wandb Dashboard Setup

### Panel 1: Training Progress
**Charts to add:**
- `metrics/mAP50(B)` vs epoch (main metric)
- `train/loss` vs epoch
- `metrics/precision(B)` vs epoch
- `metrics/recall(B)` vs epoch

### Panel 2: Losses
**Charts to add:**
- `train/box_loss` vs epoch
- `train/cls_loss` vs epoch
- `train/dfl_loss` vs epoch

### Panel 3: Per-Class Performance
**Bar chart:**
- All `metrics/mAP50(B)_[species]` at final epoch
- Shows which species are detected best

### Panel 4: System Metrics
**Charts to add:**
- `system/gpu_mem` vs epoch
- `system/gpu_util` vs epoch
- `train/time` vs epoch

---

## 📋 Expected Timeline (50 Epochs)

### Early Training (Epochs 1-10)
```
mAP50:          10-30%  (Learning basics)
box_loss:       2.0 → 1.5
cls_loss:       1.0 → 0.7
GPU mem:        18-20 GB (stable)
Time/epoch:     ~12 min
```

### Mid Training (Epochs 11-30)
```
mAP50:          30-50%  (Improving steadily)
box_loss:       1.5 → 1.0
cls_loss:       0.7 → 0.5
Best epoch:     Usually around epoch 20-25
```

### Late Training (Epochs 31-50)
```
mAP50:          50-65%  (Fine-tuning)
box_loss:       1.0 → 0.7
cls_loss:       0.5 → 0.4
Early stopping: May trigger around epoch 40-45
```

### Final Results (Epoch 50 or early stop)
```
Overall mAP50:      60-65% ✅
Elephant mAP50:     ~80%   ✅
Kudu mAP50:         ~77%   ✅
Buffalo mAP50:      ~65%   ✅
Waterbuck mAP50:    ~40%   ⚠️
Warthog mAP50:      ~29%   ⚠️ (difficult class)
```

---

## 🔧 Accessing Metrics in Code

### During Training (Automatic)

All metrics are logged automatically when wandb is initialized.

### After Training

```python
import wandb

# Initialize API
api = wandb.Api()

# Get your run
run = api.run("your-username/yolov11-wildlife-detection/run-id")

# Get all metrics
history = run.history()

# Get specific metric
mAP50 = history['metrics/mAP50(B)']
print(f"Final mAP50: {mAP50.iloc[-1]:.4f}")

# Get best mAP50
best_mAP50 = mAP50.max()
print(f"Best mAP50: {best_mAP50:.4f}")

# Get per-class metrics
elephant_mAP = history['metrics/mAP50(B)_Elephant'].iloc[-1]
kudu_mAP = history['metrics/mAP50(B)_Kudu'].iloc[-1]

print(f"Elephant: {elephant_mAP:.3f}")
print(f"Kudu: {kudu_mAP:.3f}")
```

### Export to DataFrame

```python
import pandas as pd

# Get all metrics as DataFrame
df = run.history()

# Save to CSV
df.to_csv('training_metrics.csv')

# Plot with matplotlib
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(df['_step'], df['metrics/mAP50(B)'])
plt.xlabel('Epoch')
plt.ylabel('mAP50')
plt.title('mAP50 Progress')

plt.subplot(1, 3, 2)
plt.plot(df['_step'], df['train/box_loss'])
plt.xlabel('Epoch')
plt.ylabel('Box Loss')
plt.title('Training Loss')

plt.subplot(1, 3, 3)
# Per-class final values
classes = ['Buffalo', 'Elephant', 'Kudu', 'Topi', 'Warthog', 'Waterbuck']
values = [df[f'metrics/mAP50(B)_{cls}'].iloc[-1] for cls in classes]
plt.bar(classes, values)
plt.ylabel('mAP50')
plt.title('Per-Class Performance')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('training_summary.png', dpi=300)
```

---

## 🎯 Success Criteria

### Good Training Run
✅ mAP50 > 60%  
✅ All losses decreasing smoothly  
✅ No class with mAP50 = 0%  
✅ Precision and recall balanced (not one very high, one very low)  
✅ GPU memory stable  
✅ No NaN or inf values  

### Needs Improvement
⚠️ mAP50 < 50% (may need more epochs or different settings)  
⚠️ High precision but low recall (model too conservative)  
⚠️ Low precision but high recall (too many false positives)  
⚠️ One or more classes with very low mAP (<10%)  

### Training Issues
❌ Losses increasing or oscillating wildly  
❌ mAP50 stuck at 0-10%  
❌ GPU memory growing continuously  
❌ NaN or inf losses  
❌ Training crashes  

---

## 📱 Wandb Mobile App

Monitor training from your phone!

1. Install Wandb app (iOS/Android)
2. Login with your account
3. View real-time metrics
4. Get notifications when training completes

---

## 🔔 Setting Up Alerts

### In Wandb Dashboard:

1. Go to your run
2. Click "Alerts" in left sidebar
3. Create alert:
   ```
   Condition: metrics/mAP50(B) > 0.65
   Action: Send email
   ```

4. Other useful alerts:
   ```
   - Run finishes
   - Run fails
   - mAP50 stops improving (plateau)
   - GPU memory > 22 GB
   ```

---

## 📊 Summary: Top 5 Metrics to Watch

### 1. **`metrics/mAP50(B)`** ⭐⭐⭐
   - **Most important!**
   - Target: >60%
   - Should steadily increase

### 2. **`train/box_loss`** ⭐⭐⭐
   - Shows if model is learning
   - Should decrease smoothly
   - Final: <1.0

### 3. **`metrics/precision(B)` & `metrics/recall(B)`** ⭐⭐
   - Both should be >60%
   - Balance is important
   - Trade-off between false positives and misses

### 4. **Per-class mAP50** ⭐⭐
   - Identify which species need work
   - Expect variation (Elephant best, Warthog worst)

### 5. **`system/gpu_mem`** ⭐
   - Stability check
   - Should stay around 18-20 GB
   - Prevents OOM crashes

---

## 🎓 Quick Reference Card

```
DURING TRAINING, CHECK THESE:

Every 5 epochs:
  ✓ mAP50 increasing? (Target: >60%)
  ✓ Losses decreasing? (Target: <1.0)
  ✓ GPU memory stable? (18-20 GB)

Halfway through (Epoch 25):
  ✓ mAP50 > 40%? If not, may need adjustments
  ✓ All classes detected? No 0% classes?

Final (Epoch 50):
  ✓ mAP50 > 60%? Success!
  ✓ Elephant + Kudu > 70%? Best performers
  ✓ All losses < 1.0? Model converged

WANDB DASHBOARD:
  🔗 https://wandb.ai/your-username/yolov11-wildlife-detection
  
  Main panels to watch:
    1. mAP50 chart (overall progress)
    2. Loss curves (learning progress)
    3. Per-class bar chart (species performance)
    4. GPU memory (stability)
```

---

## 🚀 That's It!

All these metrics are **automatically logged** - you don't need to do anything extra!

Just:
1. Train: `python train_vm.py --epochs 50 --batch 4`
2. Open wandb dashboard (link shown when training starts)
3. Watch the metrics update in real-time
4. Celebrate when mAP50 hits >60%! 🎉

**Your wandb dashboard will have everything you need to monitor training!** 📊✨

