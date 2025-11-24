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
                cls_id = int(parts[0])
                
                # LOGIC:
                # If label is 6, it MUST be fixed (6 -> 5).
                # If label is > 5, it's definitely wrong.
                # If label is 0, it's likely already fixed or correct (Buffalo).
                # We assume the error is a 1-based shift.
                
                if cls_id > 5:
                    new_cls_id = cls_id - 1
                    parts[0] = str(new_cls_id)
                    new_lines.append(" ".join(parts))
                    modified = True
                elif cls_id > 0:
                    # Check if we should subtract. 
                    # If we have a mix of 0-5 and 1-6, this is tricky.
                    # But the error says "Label class 6 exceeds...".
                    # This implies we definitely have 6s.
                    # Let's assume ALL non-zero labels > 5 need shifting, 
                    # OR if we want to force 1-6 -> 0-5 mapping:
                    new_cls_id = cls_id - 1
                    parts[0] = str(new_cls_id)
                    new_lines.append(" ".join(parts))
                    modified = True
                else:
                    # Label is 0. Keep it.
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
