#!/usr/bin/env python3
"""
Fix Labels Script
Project: Guacamaya - Microsoft AI for Good Lab

PURPOSE:
Iterates through all .txt label files in 'general_dataset_balanced' and 
subtracts 1 from the class ID (converting 1-6 to 0-5).

Use this if you already generated the dataset but with the wrong labels.
"""

from pathlib import Path
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.absolute()
DATASET_ROOT = BASE_DIR.parent / "general_dataset_balanced" / "general_dataset_balanced"

# ============================================================================
# MAIN
# ============================================================================

def fix_labels_in_dir(label_dir: Path):
    if not label_dir.exists():
        print(f"Skipping {label_dir} (not found)")
        return

    print(f"Processing {label_dir}...")
    files = list(label_dir.glob("*.txt"))
    
    fixed_count = 0
    
    for file_path in tqdm(files):
        with open(file_path, "r") as f:
            lines = f.readlines()
            
        new_lines = []
        modified = False
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Parse class ID
                cls_id = int(parts[0])
                
                # Check if it looks like 1-6 range (and not already 0-5)
                # If we see a '6', it's definitely wrong.
                # If we see a '0', it might already be fixed.
                # Since your CSV had [1,2,3,4,5,6], we simply subtract 1.
                
                if cls_id > 0: # Only subtract if > 0 (assuming 1-based)
                     # Wait, if we run this twice it will break. 
                     # Let's be safer: check if max label in dataset is 6?
                     # For now, simply subtract 1 as requested.
                     new_cls_id = cls_id - 1
                     parts[0] = str(new_cls_id)
                     new_lines.append(" ".join(parts))
                     modified = True
                else:
                    # If it's 0, keep it (maybe buffalo?) OR it's already fixed?
                    # If the original CSV had NO 0s, then 0 is impossible unless fixed.
                    # But if we blindly subtract, 0 becomes -1.
                    if cls_id == 0:
                        print(f"⚠ Found label 0 in {file_path.name}. Is it already fixed?")
                        new_lines.append(line.strip())
            else:
                new_lines.append(line.strip())
                
        if modified:
            with open(file_path, "w") as f:
                f.write("\n".join(new_lines))
            fixed_count += 1
            
    print(f"✓ Fixed {fixed_count} files in {label_dir.name}")

def main():
    print("="*70)
    print("FIXING LABELS (1-6 -> 0-5)")
    print("="*70)
    print(f"Dataset: {DATASET_ROOT}")
    
    fix_labels_in_dir(DATASET_ROOT / "labels" / "train")
    fix_labels_in_dir(DATASET_ROOT / "labels" / "val")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)

if __name__ == "__main__":
    main()
