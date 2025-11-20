# 🏷️ Labels Setup Guide

## Issue: No YOLO Label Files

The simplified `train_vm.py` expects **YOLO format label files** to already exist, but your dataset has **CSV annotations**.

---

## 🔍 Understanding Your Data

### What You Have (CSV Format):
```
general_dataset/
├── train/              ← Images only
├── val/                ← Images only
├── test/               ← Images only
└── groundtruth/
    └── csv/
        ├── train_big_size_A_B_E_K_WH_WB.csv  ← Annotations
        ├── val_big_size_A_B_E_K_WH_WB.csv
        └── test_big_size_A_B_E_K_WH_WB.csv
```

**CSV Format:**
```
Image,x1,y1,x2,y2,Label
L_07_05_16_DSC00126.JPG,2488,357,2520,427,3
```

### What YOLO Needs:
```
general_dataset/
├── train/
│   ├── image1.JPG
│   ├── image2.JPG
│   └── labels/         ← YOLO label files needed!
│       ├── image1.txt
│       └── image2.txt
├── val/
│   └── labels/         ← YOLO label files needed!
└── test/
    └── labels/         ← YOLO label files needed!
```

**YOLO Format (in .txt files):**
```
3 0.512000 0.192000 0.016000 0.035000
class x_center y_center width height (all normalized 0-1)
```

---

## ⚡ SOLUTION: Convert CSV to YOLO Format

### Step 1: Run the Conversion Script

```bash
cd /home/estudiante/grupo_12/subsaharian_dataset/yolo_vm
python convert_csv_to_yolo.py
```

**This will:**
1. ✅ Read CSV annotations
2. ✅ Convert to YOLO format
3. ✅ Create `labels/` folders in train/val/test
4. ✅ Generate .txt label files for each image

### Step 2: Verify Conversion

```bash
# Check labels were created
ls -la ../general_dataset/train/labels/
ls -la ../general_dataset/val/labels/
ls -la ../general_dataset/test/labels/

# Should see .txt files matching your images
```

### Step 3: Train

```bash
python train_vm.py --epochs 50 --batch 4
```

---

## 🚨 IMPORTANT: Class Label Issue

### I noticed you changed the class mapping:

**Your Change:**
```python
CLASS_NAMES = {
    1: "Buffalo",      # Started from 1
    2: "Elephant",
    3: "Kudu",
    4: "Topi",
    5: "Warthog",
    6: "Waterbuck"
}
```

**But your CSV has labels 0-5 (or 1-6)?**

### Check Your CSV Labels

```bash
# See what labels are in your CSV
cd ../general_dataset/groundtruth/csv/
cut -d',' -f6 train_big_size_A_B_E_K_WH_WB.csv | sort -u
```

### Two Scenarios:

#### Scenario A: CSV has labels 0-5

**Fix train_vm.py:**
```python
CLASS_NAMES = {
    0: "Buffalo",      # Start from 0
    1: "Elephant",
    2: "Kudu",
    3: "Topi",
    4: "Warthog",
    5: "Waterbuck"
}
```

#### Scenario B: CSV has labels 1-6

**Keep your current mapping** (1-6) but note:
- YOLO typically uses 0-indexed classes
- You may need to subtract 1 from labels during conversion

**Update `convert_csv_to_yolo.py` line 73:**
```python
# If CSV has 1-6, convert to 0-5 for YOLO
class_id = int(row['Label']) - 1  # Subtract 1
```

---

## 🔧 Quick Check Commands

### 1. Check CSV Label Range
```bash
cd /home/estudiante/grupo_12/subsaharian_dataset/general_dataset/groundtruth/csv
cat train_big_size_A_B_E_K_WH_WB.csv | cut -d',' -f6 | grep -v "Label" | sort -u
```

**If output is:** `0 1 2 3 4 5` → Use 0-5 in CLASS_NAMES  
**If output is:** `1 2 3 4 5 6` → Use 1-6 in CLASS_NAMES (and adjust conversion)

### 2. Check File Structure
```bash
cd /home/estudiante/grupo_12/subsaharian_dataset

# Should see both directories
ls -la
# Expected: yolo_vm/ and general_dataset/

# Check CSV files exist
ls -la general_dataset/groundtruth/csv/

# Check images exist
ls general_dataset/train/ | head
```

### 3. Count Images and Annotations
```bash
# Count images
echo "Train images: $(ls general_dataset/train/*.JPG 2>/dev/null | wc -l)"
echo "Val images: $(ls general_dataset/val/*.JPG 2>/dev/null | wc -l)"
echo "Test images: $(ls general_dataset/test/*.JPG 2>/dev/null | wc -l)"

# Count annotations in CSV
echo "Train annotations: $(wc -l < general_dataset/groundtruth/csv/train_big_size_A_B_E_K_WH_WB.csv)"
```

---

## 📋 Complete Workflow

### Before Training (One Time Only):

```bash
# 1. Navigate to code directory
cd /home/estudiante/grupo_12/subsaharian_dataset/yolo_vm

# 2. Check your label range in CSV
cd ../general_dataset/groundtruth/csv
head -20 train_big_size_A_B_E_K_WH_WB.csv
# Look at the "Label" column values

# 3. Fix CLASS_NAMES in train_vm.py if needed
# Edit line ~38 to match your label range (0-5 or 1-6)

# 4. Go back to code dir
cd ../../yolo_vm

# 5. Convert CSV to YOLO format
python convert_csv_to_yolo.py

# 6. Verify labels were created
ls -la ../general_dataset/train/labels/ | head

# 7. Train!
python train_vm.py --epochs 50 --batch 4
```

---

## 🎯 Expected Results After Conversion

```
general_dataset/
├── train/
│   ├── L_07_05_16_DSC00126.JPG
│   ├── L_07_05_16_DSC00127.JPG
│   └── labels/                    ← NEW
│       ├── L_07_05_16_DSC00126.txt  ← NEW
│       └── L_07_05_16_DSC00127.txt  ← NEW
├── val/
│   └── labels/                    ← NEW
│       └── *.txt                  ← NEW
└── test/
    └── labels/                    ← NEW
        └── *.txt                  ← NEW
```

**Label file content (example):**
```
3 0.5000 0.2000 0.0160 0.0350
3 0.3500 0.2300 0.0195 0.0130
```

---

## 🐛 Troubleshooting

### Issue: "CSV file not found"

**Check path:**
```bash
cd /home/estudiante/grupo_12/subsaharian_dataset
ls -R general_dataset/groundtruth/
```

**If CSV is elsewhere**, edit `convert_csv_to_yolo.py` line 90-102 with correct paths.

### Issue: "Images not found"

**Check image extensions:**
```bash
ls general_dataset/train/ | head -5
```

Images might be `.jpg` (lowercase) instead of `.JPG`. The script handles this.

### Issue: Wrong number of classes

**YOLO classes must be 0-indexed and continuous:**
- 6 classes → 0, 1, 2, 3, 4, 5 ✅
- NOT 1, 2, 3, 4, 5, 6 ❌

**If CSV has 1-6**, subtract 1 during conversion.

### Issue: "No labels after conversion"

**Check CSV file format:**
```bash
head -5 general_dataset/groundtruth/csv/train_big_size_A_B_E_K_WH_WB.csv
```

Should show:
```
Image,x1,y1,x2,y2,Label
L_07_05_16_DSC00126.JPG,2488,357,2520,427,3
```

---

## 📌 Note About OBB Model

You changed to `yolo11x-obb.pt` (Oriented Bounding Boxes).

**OBB vs Regular Detection:**
- **Regular (bbox):** Axis-aligned rectangles (x, y, w, h)
- **OBB:** Rotated rectangles (x, y, w, h, angle)

**Your CSV has regular bounding boxes** (x1, y1, x2, y2), not rotated ones.

**Recommendation:**
```python
# Use regular detection model
MODEL = "yolo11s.pt"  # or "yolo11m.pt" or "yolo11x.pt"

# NOT oriented bounding boxes (unless you have angle data)
# MODEL = "yolo11x-obb.pt"
```

---

## ✅ Summary

1. **Your data has CSV annotations, not YOLO labels**
2. **Run `convert_csv_to_yolo.py` to create labels**
3. **Fix CLASS_NAMES to match your label range** (0-5 or 1-6)
4. **Use regular model** (`yolo11s.pt`) not OBB
5. **Then train**: `python train_vm.py --epochs 50 --batch 4`

---

## 🚀 Quick Start

```bash
# Full setup in 5 commands:
cd /home/estudiante/grupo_12/subsaharian_dataset/yolo_vm
python convert_csv_to_yolo.py
# Edit train_vm.py: Fix CLASS_NAMES and MODEL
python train_vm.py --epochs 50 --batch 4
```

**That's it!** 🎉

