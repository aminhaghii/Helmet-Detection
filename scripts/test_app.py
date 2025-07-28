"""
File: test_app.py
Purpose: Simple test application to verify basic functionality
Author: HSE Vision Team
Date: 2024
Project: Construction Safety Detection System
"""

import sys
import os
from pathlib import Path

def test_basic_imports():
    """Test basic Python imports."""
    print("🔍 Testing basic imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA: Available - {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA: Not available - CPU mode only")
            
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO imported successfully")
    except ImportError as e:
        print(f"❌ Ultralytics import failed: {e}")
        return False
    
    return True

def test_project_structure():
    """Test project structure."""
    print("\n🏗️  Testing project structure...")
    
    required_dirs = [
        "src/core",
        "src/desktop_app", 
        "src/utils",
        "src/scripts",
        "config",
        "models/pretrained",
        "models/trained",
        "models/optimized",
        "logs",
        "outputs/predictions"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
        else:
            print(f"✅ Directory exists: {dir_path}")
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    return True

def test_config_files():
    """Test configuration files."""
    print("\n⚙️  Testing configuration files...")
    
    config_files = [
        "config/data.yaml",
        "config/model_config.yaml", 
        "config/deployment_config.yaml"
    ]
    
    missing_files = []
    for file_path in config_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ Config file exists: {file_path}")
    
    if missing_files:
        print(f"❌ Missing config files: {missing_files}")
        return False
    
    return True

def test_source_files():
    """Test source files."""
    print("\n📄 Testing source files...")
    
    source_files = [
        "main.py",
        "src/core/model.py",
        "src/core/inference.py",
        "src/desktop_app/main_window.py",
        "src/desktop_app/camera_handler.py",
        "src/desktop_app/detection_display.py",
        "src/utils/logger.py",
        "src/utils/data_processing.py",
        "src/scripts/train_model.py"
    ]
    
    missing_files = []
    for file_path in source_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ Source file exists: {file_path}")
    
    if missing_files:
        print(f"❌ Missing source files: {missing_files}")
        return False
    
    return True

def test_camera_access():
    """Test camera access."""
    print("\n📹 Testing camera access...")
    
    try:
        import cv2
        
        # Try to open default camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("⚠️  Camera not accessible - check permissions")
            return False
        
        # Try to read a frame
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            print(f"✅ Camera test passed - Frame shape: {frame.shape}")
            return True
        else:
            print("⚠️  Camera accessible but frame capture failed")
            return False
            
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_model_download():
    """Test model download capability."""
    print("\n🤖 Testing model download...")
    
    try:
        from ultralytics import YOLO
        
        # Try to load YOLOv8 model (will download if not present)
        print("Downloading YOLOv8l model...")
        model = YOLO('yolov8l.pt')
        
        print("✅ Model download successful")
        print(f"Model info: {model.info()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 HSE Vision - System Test")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Project Structure", test_project_structure),
        ("Config Files", test_config_files),
        ("Source Files", test_source_files),
        ("Camera Access", test_camera_access),
        ("Model Download", test_model_download)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        print("\nNext steps:")
        print("1. Run the application: python main.py")
        print("2. Train a model: python src/scripts/train_model.py")
        print("3. Unify datasets: python src/scripts/dataset_unifier.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)