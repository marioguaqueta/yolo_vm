#!/usr/bin/env python3
"""
Reorganize dataset to standard YOLO structure

Current structure:
    general_dataset/
    ├── train/
    │   ├── image1.JPG
    │   └── labels/
    │       └── image1.txt
    ├── val/
    └── test/

Target YOLO structure:
    general_dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
"""

import shutil
from pathlib import Path
from tqdm import tqdm


def reorganize_dataset(dataset_root):
    """
    Reorganize dataset from mixed structure to standard YOLO structure
    """
    print("="*70)
    print("REORGANIZING TO YOLO STANDARD STRUCTURE")
    print("="*70)
    
    dataset_root = Path(dataset_root)
    
    # Check if already reorganized
    if (dataset_root / "images").exists() and (dataset_root / "labels").exists():
        print("\n✓ Dataset already in YOLO structure!")
        print(f"  Images: {dataset_root / 'images'}")
        print(f"  Labels: {dataset_root / 'labels'}")
        return True
    
    print(f"\nDataset root: {dataset_root}")
    
    # Create new structure
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    
    images_root.mkdir(exist_ok=True)
    labels_root.mkdir(exist_ok=True)
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()} set")
        print(f"{'='*70}")
        
        old_images_dir = dataset_root / split
        old_labels_dir = dataset_root / split / "labels"
        
        new_images_dir = images_root / split
        new_labels_dir = labels_root / split
        
        # Check if old directories exist
        if not old_images_dir.exists():
            print(f"⚠ {split} directory not found, skipping...")
            continue
        
        # Create new directories
        new_images_dir.mkdir(parents=True, exist_ok=True)
        new_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Move images
        print(f"\nMoving images from {old_images_dir} to {new_images_dir}")
        image_files = list(old_images_dir.glob("*.JPG")) + list(old_images_dir.glob("*.jpg"))
        
        if not image_files:
            print(f"⚠ No images found in {old_images_dir}")
        else:
            for img_file in tqdm(image_files, desc="Moving images", unit="img"):
                dest = new_images_dir / img_file.name
                if not dest.exists():
                    shutil.move(str(img_file), str(dest))
            
            print(f"✓ Moved {len(image_files)} images")
        
        # Move labels
        if old_labels_dir.exists():
            print(f"\nMoving labels from {old_labels_dir} to {new_labels_dir}")
            label_files = list(old_labels_dir.glob("*.txt"))
            
            if not label_files:
                print(f"⚠ No labels found in {old_labels_dir}")
            else:
                for label_file in tqdm(label_files, desc="Moving labels", unit="file"):
                    dest = new_labels_dir / label_file.name
                    if not dest.exists():
                        shutil.move(str(label_file), str(dest))
                
                print(f"✓ Moved {len(label_files)} label files")
            
            # Remove old labels directory
            if old_labels_dir.exists() and not any(old_labels_dir.iterdir()):
                old_labels_dir.rmdir()
                print(f"✓ Removed empty {old_labels_dir}")
        else:
            print(f"⚠ No labels directory found at {old_labels_dir}")
        
        # Remove old split directory if empty
        if old_images_dir.exists() and not any(old_images_dir.iterdir()):
            old_images_dir.rmdir()
            print(f"✓ Removed empty {old_images_dir}")
    
    print("\n" + "="*70)
    print("REORGANIZATION COMPLETE")
    print("="*70)
    
    # Summary
    print("\nNew structure:")
    for split in splits:
        images_dir = images_root / split
        labels_dir = labels_root / split
        
        if images_dir.exists():
            img_count = len(list(images_dir.glob("*.JPG")) + list(images_dir.glob("*.jpg")))
            lbl_count = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0
            
            print(f"\n{split}:")
            print(f"  Images: {images_dir} ({img_count} files)")
            print(f"  Labels: {labels_dir} ({lbl_count} files)")
    
    print("\n" + "="*70)
    print("✓ Dataset ready for YOLO training!")
    print("="*70)
    
    return True


def main():
    """Main function"""
    
    # Determine dataset root
    BASE_DIR = Path(__file__).parent.absolute()
    DATASET_ROOT = BASE_DIR.parent / "general_dataset"
    
    print("\n" + "="*70)
    print("YOLO DATASET STRUCTURE REORGANIZER")
    print("="*70)
    print(f"\nDataset location: {DATASET_ROOT}")
    
    if not DATASET_ROOT.exists():
        print(f"\n❌ ERROR: Dataset not found at {DATASET_ROOT}")
        print("\nExpected location:")
        print("  /home/estudiante/grupo_12/subsaharian_dataset/general_dataset/")
        return 1
    
    # Show current structure
    print("\nCurrent structure:")
    for item in sorted(DATASET_ROOT.iterdir()):
        if item.is_dir():
            print(f"  {item.name}/")
            for subitem in sorted(item.iterdir())[:5]:  # Show first 5 items
                if subitem.is_dir():
                    print(f"    {subitem.name}/")
                else:
                    print(f"    {subitem.name}")
            if len(list(item.iterdir())) > 5:
                print(f"    ... and {len(list(item.iterdir())) - 5} more")
    
    # Confirm reorganization
    print("\n" + "="*70)
    print("This will reorganize your dataset to YOLO standard structure:")
    print("  images/train/, images/val/, images/test/")
    print("  labels/train/, labels/val/, labels/test/")
    print("="*70)
    
    response = input("\nProceed with reorganization? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return 0
    
    # Reorganize
    success = reorganize_dataset(DATASET_ROOT)
    
    if success:
        print("\n✓ Ready to train!")
        print("  Run: python train_vm.py --epochs 50 --batch 4")
        return 0
    else:
        print("\n❌ Reorganization failed")
        return 1


if __name__ == "__main__":
    exit(main())

