#!/usr/bin/env python3
"""
Advanced Data Generation & Balancing Script
Project: Guacamaya - Microsoft AI for Good Lab

PURPOSE:
1. Read labels from CSV ('train_big_size_A_B_E_K_WH_WB.csv').
2. Generate "Positive Crops": Crop around every animal with random padding/zoom.
3. Generate "Negative Crops": Crop random background areas from images that have NO labels (or empty areas of labeled images).
4. Balance the dataset: Ensure rare classes get more crops or augmentations.
5. Output standard YOLO structure for training.

INPUT:
- Images: general_dataset/images/train
- CSV: general_dataset/groundtruth/csv/train_big_size_A_B_E_K_WH_WB.csv

OUTPUT:
- general_dataset_balanced/
"""

import os
import cv2
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
import random
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.absolute()
DATASET_ROOT = BASE_DIR.parent / "Yolo" / "general_dataset"
CSV_PATH = DATASET_ROOT / "groundtruth" / "csv" / "train_big_size_A_B_E_K_WH_WB.csv"
IMAGES_DIR = DATASET_ROOT / "train"

OUTPUT_DIR = BASE_DIR.parent / "Yolo" / "general_dataset_balanced"

# Generation Settings
CROP_SIZE = 1024        # Target size for crops (will resize if needed)
MIN_CROP_SIZE = 512     # Minimum size of the crop area before resizing
MAX_CROP_SIZE = 2048    # Maximum size of the crop area

# Balancing Strategy
# If a class has fewer than TARGET_SAMPLES, we will oversample/augment it more.
TARGET_SAMPLES_PER_CLASS = 1000 
BACKGROUND_SAMPLES = 2000 # How many empty background images to generate

# Class Mapping (CSV Label ID -> Class Name)
# Assuming the CSV uses 0-5 or similar. Adjust if CSV uses strings.
CLASS_MAP = {
    0: "Buffalo",
    1: "Elephant",
    2: "Kudu",
    3: "Topi",
    4: "Warthog",
    5: "Waterbuck",
}

# ============================================================================
# UTILS
# ============================================================================

def iou(boxA, boxB):
    # box: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)

def random_crop_around_object(img, box, boxes_in_img, crop_w, crop_h):
    """
    Crop the image such that 'box' is centered-ish, but include other boxes if they fit.
    """
    h_img, w_img = img.shape[:2]
    bx1, by1, bx2, by2 = box
    bw, bh = bx2 - bx1, by2 - by1
    
    # Random jitter for center
    cx = (bx1 + bx2) / 2 + random.randint(int(-crop_w*0.2), int(crop_w*0.2))
    cy = (by1 + by2) / 2 + random.randint(int(-crop_h*0.2), int(crop_h*0.2))
    
    x1 = int(cx - crop_w / 2)
    y1 = int(cy - crop_h / 2)
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    
    # Handle boundaries (shift crop if it goes out of bounds)
    if x1 < 0: x2 -= x1; x1 = 0
    if y1 < 0: y2 -= y1; y1 = 0
    if x2 > w_img: x1 -= (x2 - w_img); x2 = w_img
    if y2 > h_img: y1 -= (y2 - h_img); y2 = h_img
    
    # Final check
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    
    # Crop
    crop_img = img[y1:y2, x1:x2]
    
    # Find all boxes that fall into this crop
    new_boxes = []
    for b in boxes_in_img:
        cls, ox1, oy1, ox2, oy2 = b
        
        # Intersection
        ix1 = max(x1, ox1)
        iy1 = max(y1, oy1)
        ix2 = min(x2, ox2)
        iy2 = min(y2, oy2)
        
        if ix2 > ix1 and iy2 > iy1:
            # Check visibility > 30%
            orig_area = (ox2 - ox1) * (oy2 - oy1)
            inter_area = (ix2 - ix1) * (iy2 - iy1)
            
            if (inter_area / orig_area) > 0.3:
                # Adjust coords to crop
                nx1 = ix1 - x1
                ny1 = iy1 - y1
                nx2 = ix2 - x1
                ny2 = iy2 - y1
                new_boxes.append([cls, nx1, ny1, nx2, ny2])
                
    return crop_img, new_boxes

def augment_image(image):
    """Apply random brightness/contrast/noise"""
    # Random Brightness/Contrast
    if random.random() < 0.5:
        alpha = random.uniform(0.8, 1.2) # Contrast
        beta = random.uniform(-30, 30)   # Brightness
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # Random Noise
    if random.random() < 0.3:
        noise = np.random.normal(0, 15, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
        
    return image

def rotate_image_and_boxes(image, boxes, angle):
    """Rotate image and bounding boxes"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(image, M, (w, h))
    
    new_boxes = []
    for box in boxes:
        cls, x1, y1, x2, y2 = box
        
        # Rotate all 4 corners
        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ])
        
        ones = np.ones(shape=(len(corners), 1))
        corners_ones = np.hstack([corners, ones])
        
        transformed_corners = M.dot(corners_ones.T).T
        
        nx1 = min(transformed_corners[:, 0])
        ny1 = min(transformed_corners[:, 1])
        nx2 = max(transformed_corners[:, 0])
        ny2 = max(transformed_corners[:, 1])
        
        # Clip to image bounds
        nx1 = max(0, min(nx1, w))
        ny1 = max(0, min(ny1, h))
        nx2 = max(0, min(nx2, w))
        ny2 = max(0, min(ny2, h))
        
        if nx2 > nx1 and ny2 > ny1:
            new_boxes.append([cls, nx1, ny1, nx2, ny2])
            
    return rotated_img, new_boxes

def save_yolo_label(boxes, path, img_w, img_h):
    lines = []
    for b in boxes:
        cls, x1, y1, x2, y2 = b
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    
    with open(path, "w") as f:
        f.write("\n".join(lines))

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("="*70)
    print("DATASET BALANCING & GENERATION")
    print("="*70)
    
    # 1. Load CSV
    print(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # Group by image
    # CSV cols: Image,x1,y1,x2,y2,Label
    img_groups = df.groupby("Image")
    
    # Track stats
    class_counts = defaultdict(int)
    
    # Prepare Output Dirs
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    
    (OUTPUT_DIR / "images" / "train").mkdir(parents=True)
    (OUTPUT_DIR / "labels" / "train").mkdir(parents=True)
    (OUTPUT_DIR / "images" / "val").mkdir(parents=True) # Will be empty initially
    (OUTPUT_DIR / "labels" / "val").mkdir(parents=True)
    
    # 2. Process Labeled Images (Positive Crops)
    print("\nGenerating Positive Crops...")
    
    all_images = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.JPG"))
    all_image_names = {p.name for p in all_images}
    labeled_image_names = set(df["Image"].unique())
    
    # Identify images WITHOUT labels (Backgrounds)
    background_images = [p for p in all_images if p.name not in labeled_image_names]
    
    generated_count = 0
    
    for img_name, group in tqdm(img_groups):
        img_path = IMAGES_DIR / img_name
        if not img_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        # Parse boxes for this image
        boxes = []
        for _, row in group.iterrows():
            boxes.append([row['Label'], row['x1'], row['y1'], row['x2'], row['y2']])
            class_counts[row['Label']] += 1
            
        # Generate crops for EACH object in the image
        # To balance, we can repeat this loop for rare classes
        for box in boxes:
            cls = box[0]
            
            # Simple heuristic: if class is rare, generate more crops
            # (You can tune this logic later)
            num_crops = 1
            if class_counts[cls] < 500: num_crops = 3
            if class_counts[cls] < 100: num_crops = 5
            
            for i in range(num_crops):
                # Random crop size
                cw = random.randint(MIN_CROP_SIZE, MAX_CROP_SIZE)
                ch = cw # Square crop
                
                crop, crop_boxes = random_crop_around_object(img, box[1:], boxes, cw, ch)
                
                if len(crop_boxes) > 0:
                    # Resize to standard size (e.g. 1024)
                    crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
                    
                    # Adjust box coordinates for resize
                    scale_x = CROP_SIZE / crop.shape[1]
                    scale_y = CROP_SIZE / crop.shape[0]
                    
                    final_boxes = []
                    for b in crop_boxes:
                        c, x1, y1, x2, y2 = b
                        final_boxes.append([c, x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y])
                    
                    # Apply Augmentations (Color/Noise)
                    crop_aug = augment_image(crop_resized)
                    
                    # Apply Rotation (Randomly)
                    if random.random() < 0.3:
                        angle = random.choice([90, 180, 270])
                        crop_aug, final_boxes = rotate_image_and_boxes(crop_aug, final_boxes, angle)

                    # Save
                    fname = f"{Path(img_name).stem}_crop_{generated_count}"
                    cv2.imwrite(str(OUTPUT_DIR / "images" / "train" / f"{fname}.jpg"), crop_aug)
                    save_yolo_label(final_boxes, OUTPUT_DIR / "labels" / "train" / f"{fname}.txt", CROP_SIZE, CROP_SIZE)
                    generated_count += 1

    print(f"✓ Generated {generated_count} positive crops.")
    
    # 3. Generate Negative Crops (Backgrounds)
    print("\nGenerating Negative Crops (Backgrounds)...")
    bg_count = 0
    
    # Shuffle background images
    random.shuffle(background_images)
    
    while bg_count < BACKGROUND_SAMPLES and len(background_images) > 0:
        # Pick a random image
        img_path = random.choice(background_images)
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        h, w = img.shape[:2]
        if w < MIN_CROP_SIZE or h < MIN_CROP_SIZE: continue
        
        # Random crop
        cw = random.randint(MIN_CROP_SIZE, min(w, MAX_CROP_SIZE))
        ch = cw
        
        x = random.randint(0, w - cw)
        y = random.randint(0, h - ch)
        
        crop = img[y:y+ch, x:x+cw]
        crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
        
        # Save (Empty label file)
        fname = f"bg_{Path(img_path.name).stem}_{bg_count}"
        cv2.imwrite(str(OUTPUT_DIR / "images" / "train" / f"{fname}.jpg"), crop_resized)
        with open(OUTPUT_DIR / "labels" / "train" / f"{fname}.txt", "w") as f:
            pass # Empty file
            
        bg_count += 1
        
    print(f"✓ Generated {bg_count} negative background crops.")
    
    # 4. Create Validation Split (Simple random move)
    print("\nCreating Validation Split (20%)...")
    all_gen_imgs = list((OUTPUT_DIR / "images" / "train").glob("*.jpg"))
    random.shuffle(all_gen_imgs)
    
    val_count = int(len(all_gen_imgs) * 0.2)
    for i in range(val_count):
        img_p = all_gen_imgs[i]
        lbl_p = OUTPUT_DIR / "labels" / "train" / f"{img_p.stem}.txt"
        
        shutil.move(str(img_p), str(OUTPUT_DIR / "images" / "val" / img_p.name))
        shutil.move(str(lbl_p), str(OUTPUT_DIR / "labels" / "val" / lbl_p.name))
        
    print(f"✓ Moved {val_count} images to validation.")
    
    print("\n" + "="*70)
    print("BALANCED DATASET READY")
    print(f"Location: {OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
