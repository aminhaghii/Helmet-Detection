"""
File: verify_training_setup.py
Purpose: Verify that the project is ready for YOLOv8s training
Author: HSE Vision Team
Date: 2025
Project: Construction Safety Detection System
"""

import os
import sys
import yaml
from pathlib import Path
import torch
from ultralytics import YOLO

def check_file_exists(file_path, description):
    """Check if a file exists and print status."""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} (NOT FOUND)")
        return False

def check_directory_exists(dir_path, description):
    """Check if a directory exists and print status."""
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        print(f"✅ {description}: {dir_path}")
        return True
    else:
        print(f"❌ {description}: {dir_path} (NOT FOUND)")
        return False

def verify_training_setup():
    """Verify that all components are ready for training."""
    print("🔍 HSE Vision - Training Setup Verification")
    print("=" * 50)
    
    project_root = Path(__file__).parent
    all_checks_passed = True
    
    # Check essential files
    print("\n📁 Essential Files:")
    essential_files = [
        (project_root / "yolov8s.pt", "YOLOv8s pretrained weights"),
        (project_root / "config" / "train_config.yaml", "Training configuration"),
        (project_root / "unified_dataset" / "data.yaml", "Dataset configuration"),
        (project_root / "src" / "scripts" / "train_model.py", "Training script"),
    ]
    
    for file_path, description in essential_files:
        if not check_file_exists(file_path, description):
            all_checks_passed = False
    
    # Check directories
    print("\n📂 Essential Directories:")
    essential_dirs = [
        (project_root / "unified_dataset" / "train" / "images", "Training images"),
        (project_root / "unified_dataset" / "train" / "labels", "Training labels"),
        (project_root / "unified_dataset" / "val" / "images", "Validation images"),
        (project_root / "unified_dataset" / "val" / "labels", "Validation labels"),
        (project_root / "models", "Models directory"),
        (project_root / "logs", "Logs directory"),
    ]
    
    for dir_path, description in essential_dirs:
        if not check_directory_exists(dir_path, description):
            all_checks_passed = False
    
    # Check training configuration
    print("\n⚙️ Training Configuration:")
    try:
        config_path = project_root / "config" / "train_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_name = config['model']['name']
        epochs = config['training']['epochs']
        batch_size = config['training']['batch_size']
        
        print(f"✅ Model: {model_name}")
        print(f"✅ Epochs: {epochs}")
        print(f"✅ Batch Size: {batch_size}")
        
        if model_name != "yolov8s":
            print(f"⚠️  Warning: Model is set to {model_name}, expected yolov8s")
            
    except Exception as e:
        print(f"❌ Failed to load training configuration: {e}")
        all_checks_passed = False
    
    # Check dataset configuration
    print("\n📊 Dataset Configuration:")
    try:
        data_config_path = project_root / "unified_dataset" / "data.yaml"
        with open(data_config_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        train_path = data_config.get('train', 'Not specified')
        val_path = data_config.get('val', 'Not specified')
        nc = data_config.get('nc', 'Not specified')
        names = data_config.get('names', [])
        
        print(f"✅ Training path: {train_path}")
        print(f"✅ Validation path: {val_path}")
        print(f"✅ Number of classes: {nc}")
        print(f"✅ Class names: {names}")
        
    except Exception as e:
        print(f"❌ Failed to load dataset configuration: {e}")
        all_checks_passed = False
    
    # Check PyTorch and CUDA
    print("\n🔧 System Requirements:")
    try:
        print(f"✅ PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✅ CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA not available - training will use CPU (slower)")
    except Exception as e:
        print(f"❌ PyTorch check failed: {e}")
        all_checks_passed = False
    
    # Check Ultralytics
    print("\n🤖 YOLO Framework:")
    try:
        # Try to load YOLOv8s
        model = YOLO('yolov8s.pt')
        print("✅ YOLOv8s model loaded successfully")
        print(f"✅ Ultralytics version: Available")
    except Exception as e:
        print(f"❌ Failed to load YOLOv8s: {e}")
        all_checks_passed = False
    
    # Count dataset images
    print("\n📈 Dataset Statistics:")
    try:
        train_images_dir = project_root / "unified_dataset" / "train" / "images"
        val_images_dir = project_root / "unified_dataset" / "val" / "images"
        
        if train_images_dir.exists():
            train_count = len(list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.png")))
            print(f"✅ Training images: {train_count}")
        
        if val_images_dir.exists():
            val_count = len(list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png")))
            print(f"✅ Validation images: {val_count}")
            
    except Exception as e:
        print(f"❌ Failed to count dataset images: {e}")
    
    # Final status
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("🎉 ALL CHECKS PASSED! Ready to start training.")
        print("\n🚀 To start training, run:")
        print("python src/scripts/train_model.py --data unified_dataset/data.yaml --no-wandb")
    else:
        print("❌ Some checks failed. Please resolve the issues above before training.")
    
    print("=" * 50)
    return all_checks_passed

if __name__ == "__main__":
    verify_training_setup()