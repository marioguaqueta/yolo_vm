#!/usr/bin/env python3
"""
Offline Data Augmentation & Tiling Script
Project: Guacamaya - Microsoft AI for Good Lab

PURPOSE:
This script drastically increases dataset size and small-object visibility by:
1. TILING (Slicing): Cutting high-res aerial images into smaller, overlapping chips.
   - Why? Preserves full resolution. A 20px gazelle in a 4000px image becomes a 
     20px gazelle in a 1024px image (much "larger" for the model).
2. AUGMENTATION: Generating synthetic variations (Contrast, Brightness, Noise) 
   for the new tiles.

OUTPUT:
Creates a new dataset folder 'general_dataset_tiled' that you can point your training to.
"""

import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.absolute()
INPUT_DIR = BASE_DIR.parent / "general_dataset"
OUTPUT_DIR = BASE_DIR.parent / "general_dataset_tiled"

# Tiling Settings
TILE_SIZE = 1024        # Size of the square chips (e.g., 1024x1024)
OVERLAP = 0.20          # 20% overlap between tiles to catch objects on edges
MIN_VISIBILITY = 0.3    # If an object is cut, keep it if >30% is visible

# Augmentation Settings (Applied to tiles)
AUGMENT_PROB = 0.5      # 50% chance to apply extra color/noise augmentation to a tile
GENERATE_FLIPS = True   # Also generate horizontal flips for every tile

# ============================================================================
# UTILS
# ============================================================================

def load_yolo_labels(label_path, img_width, img_height):
    """Read YOLO labels and convert to pixel coordinates [class, x1, y1, x2, y2]"""
    boxes = []
    if not label_path.exists():
        return boxes
    
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            w = float(parts[3]) * img_width
            h = float(parts[4]) * img_height
            
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2
            boxes.append([cls, x1, y1, x2, y2])
    return boxes

def save_yolo_labels(boxes, output_path, img_width, img_height):
    """Convert pixel coordinates back to YOLO format and save"""
    lines = []
    for box in boxes:
        cls, x1, y1, x2, y2 = box
        
        # Clip to image bounds
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        
        w = x2 - x1
        h = y2 - y1
        
        if w <= 0 or h <= 0:
            continue
            
        x_center = (x1 + w / 2) / img_width
        y_center = (y1 + h / 2) / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        
        lines.append(f"{int(cls)} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
    if lines:
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

def augment_image(image):
    """Apply random brightness/contrast/noise"""
    # Random Brightness/Contrast
    alpha = random.uniform(0.8, 1.2) # Contrast
    beta = random.uniform(-30, 30)   # Brightness
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # Random Noise
    if random.random() > 0.5:
        noise = np.random.normal(0, 15, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
        
    return image

# ============================================================================
# PROCESSING
# ============================================================================

def process_dataset(split="train"):
    print(f"\nProcessing split: {split}...")
    
    img_dir = INPUT_DIR / "images" / split
    lbl_dir = INPUT_DIR / "labels" / split
    
    out_img_dir = OUTPUT_DIR / "images" / split
    out_lbl_dir = OUTPUT_DIR / "labels" / split
    
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.JPG"))
    
    for img_path in tqdm(images):
        # Load Image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]
        
        # Load Labels
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        boxes = load_yolo_labels(lbl_path, w_img, h_img)
        
        # Calculate Tiles
        stride = int(TILE_SIZE * (1 - OVERLAP))
        
        # Grid generation
        x_steps = list(range(0, w_img - TILE_SIZE, stride))
        if (w_img - TILE_SIZE) not in x_steps and w_img > TILE_SIZE:
             x_steps.append(w_img - TILE_SIZE) # Ensure last patch covers edge
             
        y_steps = list(range(0, h_img - TILE_SIZE, stride))
        if (h_img - TILE_SIZE) not in y_steps and h_img > TILE_SIZE:
            y_steps.append(h_img - TILE_SIZE)

        # If image is smaller than tile size, just copy it (or pad)
        if w_img < TILE_SIZE or h_img < TILE_SIZE:
            x_steps = [0]
            y_steps = [0]
            # Note: Ideally we should pad here, but for now we skip or resize logic could be added
            # For simplicity, we assume aerial images are large.
        
        tile_count = 0
        for y in y_steps:
            for x in x_steps:
                # Define tile window
                tile_x1, tile_y1 = x, y
                tile_x2, tile_y2 = x + TILE_SIZE, y + TILE_SIZE
                
                # Crop Image
                tile_img = img[tile_y1:tile_y2, tile_x1:tile_x2]
                
                # Filter Boxes in this tile
                tile_boxes = []
                for box in boxes:
                    cls, bx1, by1, bx2, by2 = box
                    
                    # Intersection
                    ix1 = max(tile_x1, bx1)
                    iy1 = max(tile_y1, by1)
                    ix2 = min(tile_x2, bx2)
                    iy2 = min(tile_y2, by2)
                    
                    iw = ix2 - ix1
                    ih = iy2 - iy1
                    
                    if iw > 0 and ih > 0:
                        # Check visibility (how much of the original object is in this tile?)
                        box_area = (bx2 - bx1) * (by2 - by1)
                        inter_area = iw * ih
                        if (inter_area / box_area) >= MIN_VISIBILITY:
                            # Adjust coordinates relative to tile
                            new_x1 = ix1 - tile_x1
                            new_y1 = iy1 - tile_y1
                            new_x2 = ix2 - tile_x1
                            new_y2 = iy2 - tile_y1
                            tile_boxes.append([cls, new_x1, new_y1, new_x2, new_y2])
                
                # Save Tile ONLY if it has labels (optional: keep empty backgrounds?)
                # For wildlife, keeping SOME empty backgrounds is good to reduce False Positives.
                # Let's keep all tiles for now, or maybe 10% of empty ones.
                has_labels = len(tile_boxes) > 0
                
                if has_labels or random.random() < 0.1: # Keep 10% of empty background tiles
                    tile_name = f"{img_path.stem}_tile_{tile_count}"
                    
                    # 1. Save Original Tile
                    cv2.imwrite(str(out_img_dir / f"{tile_name}.jpg"), tile_img)
                    if has_labels:
                        save_yolo_labels(tile_boxes, out_lbl_dir / f"{tile_name}.txt", TILE_SIZE, TILE_SIZE)
                    
                    # 2. Augmentation (Color/Noise)
                    if split == "train" and random.random() < AUGMENT_PROB:
                        aug_img = augment_image(tile_img)
                        aug_name = f"{tile_name}_aug"
                        cv2.imwrite(str(out_img_dir / f"{aug_name}.jpg"), aug_img)
                        if has_labels:
                            save_yolo_labels(tile_boxes, out_lbl_dir / f"{aug_name}.txt", TILE_SIZE, TILE_SIZE)

                    # 3. Horizontal Flip
                    if split == "train" and GENERATE_FLIPS:
                        flip_img = cv2.flip(tile_img, 1)
                        flip_name = f"{tile_name}_flip"
                        cv2.imwrite(str(out_img_dir / f"{flip_name}.jpg"), flip_img)
                        
                        if has_labels:
                            # Flip boxes
                            flip_boxes = []
                            for b in tile_boxes:
                                cls, fx1, fy1, fx2, fy2 = b
                                # x' = width - x
                                new_fx1 = TILE_SIZE - fx2
                                new_fx2 = TILE_SIZE - fx1
                                flip_boxes.append([cls, new_fx1, fy1, new_fx2, fy2])
                            save_yolo_labels(flip_boxes, out_lbl_dir / f"{flip_name}.txt", TILE_SIZE, TILE_SIZE)

                    tile_count += 1

def main():
    print("="*70)
    print("DATASET TILING & AUGMENTATION TOOL")
    print("="*70)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Tile:   {TILE_SIZE}x{TILE_SIZE}")
    print(f"Overlap:{OVERLAP*100}%")
    
    if OUTPUT_DIR.exists():
        print("⚠ Output directory exists. Deleting to start fresh...")
        shutil.rmtree(OUTPUT_DIR)
    
    process_dataset("train")
    process_dataset("val")
    process_dataset("test")
    
    print("\n" + "="*70)
    print("DONE! New dataset created.")
    print(f"Location: {OUTPUT_DIR}")
    print("="*70)
    print("\nNEXT STEPS:")
    print("1. Update train_vm_v3.py to point to 'general_dataset_tiled'")
    print("   (Change DATASET_ROOT line)")
    print("2. Run training again.")

if __name__ == "__main__":
    main()
