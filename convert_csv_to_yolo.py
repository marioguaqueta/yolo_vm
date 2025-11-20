#!/usr/bin/env python3
"""
CSV to YOLO Format Converter
Converts CSV annotations (x1,y1,x2,y2,Label) to YOLO format (class x_center y_center width height)
"""

import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def convert_csv_to_yolo(csv_path, images_dir, output_labels_dir):
    """
    Convert CSV annotations to YOLO format
    
    Args:
        csv_path: Path to CSV file (e.g., train_big_size_A_B_E_K_WH_WB.csv)
        images_dir: Directory containing images (e.g., general_dataset/train/)
        output_labels_dir: Where to save YOLO label files (e.g., general_dataset/train/labels/)
    """
    print(f"\nProcessing: {csv_path}")
    print(f"Images dir: {images_dir}")
    print(f"Output labels: {output_labels_dir}")
    
    # Check if CSV exists
    if not csv_path.exists():
        print(f"❌ ERROR: CSV file not found at {csv_path}")
        return 0, 0
    
    # Create output directory
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Total annotations in CSV: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample data:\n{df.head()}")
    
    # Group by image
    grouped = df.groupby('Image')
    images_processed = 0
    annotations_written = 0
    images_not_found = []
    
    for image_name, group in tqdm(grouped, desc="Converting", unit="img"):
        # Find image file
        image_path = images_dir / image_name
        
        if not image_path.exists():
            # Try different extensions
            found = False
            for ext in ['.JPG', '.jpg', '.png', '.jpeg', '.PNG']:
                alt_path = images_dir / image_name.replace('.JPG', ext)
                if alt_path.exists():
                    image_path = alt_path
                    found = True
                    break
            
            if not found:
                images_not_found.append(image_name)
                continue
        
        # Get image dimensions
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error reading {image_name}: {e}")
            continue
        
        # Create YOLO label file
        label_file = output_labels_dir / f"{image_path.stem}.txt"
        
        with open(label_file, 'w') as f:
            for _, row in group.iterrows():
                # CSV format: Image, x1, y1, x2, y2, Label
                class_id = int(row['Label'])
                x1, y1, x2, y2 = float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2'])
                
                # Convert to YOLO format (normalized)
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # Clamp to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                # YOLO format: class x_center y_center width height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                annotations_written += 1
        
        images_processed += 1
    
    print(f"\n✓ Images processed: {images_processed}")
    print(f"✓ Annotations written: {annotations_written}")
    
    if images_not_found:
        print(f"\n⚠ {len(images_not_found)} images not found:")
        for img in images_not_found[:10]:  # Show first 10
            print(f"  - {img}")
        if len(images_not_found) > 10:
            print(f"  ... and {len(images_not_found) - 10} more")
    
    return images_processed, annotations_written


def main():
    """Convert all CSV files to YOLO format"""
    
    # Paths
    BASE_DIR = Path(__file__).parent.absolute()
    DATASET_ROOT = BASE_DIR.parent / "general_dataset"
    
    print("="*70)
    print("CSV TO YOLO FORMAT CONVERTER")
    print("="*70)
    print(f"\nDataset root: {DATASET_ROOT}")
    
    # Check if dataset exists
    if not DATASET_ROOT.exists():
        print(f"\n❌ ERROR: Dataset not found at {DATASET_ROOT}")
        print("\nExpected structure:")
        print("  sahariandataset/")
        print("  ├── yolo_vm/")
        print("  └── general_dataset/")
        print("      ├── train/")
        print("      ├── val/")
        print("      ├── test/")
        print("      └── groundtruth/csv/")
        return 1
    
    # Define conversions - using YOLO standard structure
    conversions = [
        {
            'name': 'Training',
            'csv': DATASET_ROOT / "groundtruth/csv/train_big_size_A_B_E_K_WH_WB.csv",
            'images': DATASET_ROOT / "train",  # Source images
            'labels': DATASET_ROOT / "labels/train"  # YOLO standard: labels/train/
        },
        {
            'name': 'Validation',
            'csv': DATASET_ROOT / "groundtruth/csv/val_big_size_A_B_E_K_WH_WB.csv",
            'images': DATASET_ROOT / "val",
            'labels': DATASET_ROOT / "labels/val"  # YOLO standard: labels/val/
        },
        {
            'name': 'Test',
            'csv': DATASET_ROOT / "groundtruth/csv/test_big_size_A_B_E_K_WH_WB.csv",
            'images': DATASET_ROOT / "test",
            'labels': DATASET_ROOT / "labels/test"  # YOLO standard: labels/test/
        }
    ]
    
    print("\n💡 Note: Creating labels in YOLO standard structure (labels/train/, labels/val/, labels/test/)")
    print("   Images remain in current location (train/, val/, test/)")
    print("   Run reorganize_to_yolo_structure.py to complete YOLO standard structure")
    
    # Check what exists
    print("\nChecking files:")
    for conv in conversions:
        csv_exists = "✓" if conv['csv'].exists() else "✗"
        img_exists = "✓" if conv['images'].exists() else "✗"
        print(f"  {csv_exists} {conv['name']} CSV: {conv['csv'].name}")
        print(f"  {img_exists} {conv['name']} Images: {conv['images']}")
    
    print("\n" + "="*70)
    
    # Convert each split
    total_images = 0
    total_annotations = 0
    
    for conv in conversions:
        print(f"\n{'='*70}")
        print(f"{conv['name'].upper()} SET")
        print(f"{'='*70}")
        
        if not conv['csv'].exists():
            print(f"⚠ Skipping - CSV not found")
            continue
        
        if not conv['images'].exists():
            print(f"⚠ Skipping - Images directory not found")
            continue
        
        images, annotations = convert_csv_to_yolo(
            conv['csv'],
            conv['images'],
            conv['labels']
        )
        
        total_images += images
        total_annotations += annotations
    
    # Summary
    print("\n" + "="*70)
    print("CONVERSION COMPLETE")
    print("="*70)
    print(f"Total images: {total_images}")
    print(f"Total annotations: {total_annotations}")
    print(f"\nLabel files created in:")
    for conv in conversions:
        if conv['labels'].exists():
            count = len(list(conv['labels'].glob("*.txt")))
            print(f"  {conv['labels']}: {count} files")
    
    print("\n✓ Ready for training!")
    print("  Run: python train_vm.py --epochs 50 --batch 4")
    
    return 0


if __name__ == "__main__":
    exit(main())

