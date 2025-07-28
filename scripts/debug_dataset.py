#!/usr/bin/env python3
"""
Debug script to test dataset path resolution
"""

import yaml
from pathlib import Path

def debug_dataset(dataset_path):
    """Debug a single dataset"""
    dataset_path = Path(dataset_path)
    data_yaml_path = dataset_path / "data.yaml"
    
    print(f"\n=== Debugging Dataset: {dataset_path.name} ===")
    print(f"Dataset path: {dataset_path}")
    print(f"Data.yaml path: {data_yaml_path}")
    print(f"Data.yaml exists: {data_yaml_path.exists()}")
    
    if not data_yaml_path.exists():
        return
    
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Config: {config}")
    
    for split in ['train', 'val', 'test']:
        if split in config:
            split_path = config[split]
            print(f"\n--- {split.upper()} ---")
            print(f"Raw path: {split_path}")
            
            # Resolve path
            if not Path(split_path).is_absolute():
                resolved_path = dataset_path / split_path
            else:
                resolved_path = Path(split_path)
            
            resolved_path = resolved_path.resolve()
            print(f"Resolved path: {resolved_path}")
            print(f"Path exists: {resolved_path.exists()}")
            
            if resolved_path.exists():
                # Check for images directory
                if resolved_path.name == 'images' or (resolved_path / 'images').exists():
                    images_dir = resolved_path if resolved_path.name == 'images' else resolved_path / 'images'
                    labels_dir = images_dir.parent / 'labels'
                    
                    print(f"Images dir: {images_dir}")
                    print(f"Labels dir: {labels_dir}")
                    print(f"Images dir exists: {images_dir.exists()}")
                    print(f"Labels dir exists: {labels_dir.exists()}")
                    
                    if images_dir.exists():
                        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.png'))
                        print(f"Image count: {len(image_files)}")
                        if image_files:
                            print(f"Sample images: {[f.name for f in image_files[:3]]}")
                    
                    if labels_dir.exists():
                        label_files = list(labels_dir.glob('*.txt'))
                        print(f"Label count: {len(label_files)}")
                        if label_files:
                            print(f"Sample labels: {[f.name for f in label_files[:3]]}")

if __name__ == "__main__":
    data_dir = Path("Data")
    
    for dataset_path in data_dir.iterdir():
        if dataset_path.is_dir() and (dataset_path / 'data.yaml').exists():
            debug_dataset(dataset_path)