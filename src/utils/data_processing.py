"""
File: data_processing.py
Purpose: Data preprocessing and dataset unification utilities
Author: HSE Vision Team
Date: 2024
Project: Construction Safety Detection System
"""

import os
import shutil
import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DatasetUnifier:
    """Unify multiple datasets into a single training dataset."""
    
    def __init__(self, source_datasets: List[str], output_dir: str):
        """
        Initialize dataset unifier.
        
        Args:
            source_datasets: List of dataset directory paths
            output_dir: Output directory for unified dataset
        """
        self.source_datasets = source_datasets
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output subdirectories
        (self.output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "test").mkdir(parents=True, exist_ok=True)
    
    def unify_datasets(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """
        Unify all source datasets into a single dataset.
        
        Args:
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
        """
        logger.info("Starting dataset unification...")
        
        all_images = []
        all_labels = []
        
        # Collect all images and labels from source datasets
        for dataset_path in self.source_datasets:
            dataset_path = Path(dataset_path)
            logger.info(f"Processing dataset: {dataset_path}")
            
            # Find all images and labels
            for split in ['train', 'valid', 'test']:
                img_dir = dataset_path / split / 'images'
                lbl_dir = dataset_path / split / 'labels'
                
                if img_dir.exists() and lbl_dir.exists():
                    images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
                    for img_path in images:
                        lbl_path = lbl_dir / f"{img_path.stem}.txt"
                        if lbl_path.exists():
                            all_images.append(img_path)
                            all_labels.append(lbl_path)
        
        logger.info(f"Found {len(all_images)} image-label pairs")
        
        # Shuffle and split data
        indices = np.random.permutation(len(all_images))
        
        train_end = int(len(indices) * train_ratio)
        val_end = train_end + int(len(indices) * val_ratio)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Copy files to unified dataset
        self._copy_files(all_images, all_labels, train_indices, 'train')
        self._copy_files(all_images, all_labels, val_indices, 'val')
        self._copy_files(all_images, all_labels, test_indices, 'test')
        
        logger.info("Dataset unification completed!")
        logger.info(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    def _copy_files(self, images: List[Path], labels: List[Path], indices: np.ndarray, split: str):
        """Copy files for a specific split."""
        logger.info(f"Copying {len(indices)} files to {split} split...")
        
        for i, idx in enumerate(tqdm(indices, desc=f"Copying {split}")):
            img_src = images[idx]
            lbl_src = labels[idx]
            
            # Create unique filename to avoid conflicts
            new_name = f"{split}_{i:06d}{img_src.suffix}"
            
            img_dst = self.output_dir / "images" / split / new_name
            lbl_dst = self.output_dir / "labels" / split / f"{Path(new_name).stem}.txt"
            
            shutil.copy2(img_src, img_dst)
            shutil.copy2(lbl_src, lbl_dst)

class ImageProcessor:
    """Image preprocessing utilities."""
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image
            target_size: Target (width, height)
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        
        # Calculate padding offsets
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # Place resized image in center
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """
        Normalize image for model input.
        
        Args:
            image: Input image (0-255)
            
        Returns:
            Normalized image (0-1)
        """
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def preprocess_for_inference(image: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
        """
        Complete preprocessing pipeline for model inference.
        
        Args:
            image: Input image
            input_size: Model input size (width, height)
            
        Returns:
            Preprocessed image ready for inference
        """
        # Resize with padding
        resized = ImageProcessor.resize_image(image, input_size)
        
        # Normalize
        normalized = ImageProcessor.normalize_image(resized)
        
        # Convert to CHW format and add batch dimension
        preprocessed = np.transpose(normalized, (2, 0, 1))
        preprocessed = np.expand_dims(preprocessed, axis=0)
        
        return preprocessed

def create_unified_config(output_dir: str, num_classes: int, class_names: List[str]):
    """
    Create unified data.yaml configuration file.
    
    Args:
        output_dir: Output directory path
        num_classes: Number of classes
        class_names: List of class names
    """
    config = {
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': num_classes,
        'names': class_names
    }
    
    config_path = Path(output_dir) / 'data.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Created unified config: {config_path}")

def validate_dataset(dataset_dir: str) -> Dict[str, int]:
    """
    Validate dataset structure and count files.
    
    Args:
        dataset_dir: Dataset directory path
        
    Returns:
        Dictionary with file counts for each split
    """
    dataset_path = Path(dataset_dir)
    counts = {}
    
    for split in ['train', 'val', 'test']:
        img_dir = dataset_path / 'images' / split
        lbl_dir = dataset_path / 'labels' / split
        
        if img_dir.exists() and lbl_dir.exists():
            images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
            labels = list(lbl_dir.glob('*.txt'))
            
            counts[split] = {
                'images': len(images),
                'labels': len(labels),
                'matched': len([img for img in images 
                              if (lbl_dir / f"{img.stem}.txt").exists()])
            }
        else:
            counts[split] = {'images': 0, 'labels': 0, 'matched': 0}
    
    return counts