"""
File: setup.py
Purpose: Project setup and installation script
Author: HSE Vision Team
Date: 2024
Project: Construction Safety Detection System
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HSEVisionSetup:
    """
    Setup and installation manager for HSE Vision project.
    """
    
    def __init__(self):
        """Initialize setup manager."""
        self.project_root = Path(__file__).parent
        self.python_version = sys.version_info
        self.platform = platform.system()
        
        logger.info("HSE Vision Setup Manager")
        logger.info(f"Python Version: {self.python_version.major}.{self.python_version.minor}")
        logger.info(f"Platform: {self.platform}")
        logger.info(f"Project Root: {self.project_root}")
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        if self.python_version.major != 3 or self.python_version.minor < 8:
            logger.error("Python 3.8+ is required")
            return False
        
        logger.info("✅ Python version check passed")
        return True
    
    def check_gpu_support(self) -> dict:
        """Check GPU and CUDA support."""
        gpu_info = {
            'cuda_available': False,
            'gpu_count': 0,
            'gpu_name': None,
            'cuda_version': None
        }
        
        try:
            import torch
            gpu_info['cuda_available'] = torch.cuda.is_available()
            
            if gpu_info['cuda_available']:
                gpu_info['gpu_count'] = torch.cuda.device_count()
                gpu_info['gpu_name'] = torch.cuda.get_device_name(0)
                gpu_info['cuda_version'] = torch.version.cuda
                
                logger.info("✅ CUDA support detected")
                logger.info(f"   GPU: {gpu_info['gpu_name']}")
                logger.info(f"   CUDA Version: {gpu_info['cuda_version']}")
            else:
                logger.warning("⚠️  CUDA not available - CPU mode only")
                
        except ImportError:
            logger.warning("⚠️  PyTorch not installed - will install during setup")
        
        return gpu_info
    
    def install_dependencies(self) -> bool:
        """Install project dependencies."""
        try:
            logger.info("Installing dependencies...")
            
            # Upgrade pip first
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=True)
            
            # Install requirements
            requirements_file = self.project_root / "requirements.txt"
            if requirements_file.exists():
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], 
                             check=True, capture_output=True)
                logger.info("✅ Dependencies installed successfully")
                return True
            else:
                logger.error("❌ requirements.txt not found")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    
    def setup_directories(self) -> bool:
        """Create necessary project directories."""
        try:
            directories = [
                "models/pretrained",
                "models/trained", 
                "models/optimized",
                "logs",
                "outputs/predictions",
                "unified_dataset"
            ]
            
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"📁 Created directory: {directory}")
            
            logger.info("✅ Directory structure created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create directories: {e}")
            return False
    
    def download_pretrained_model(self) -> bool:
        """Download pretrained YOLOv8 model."""
        try:
            logger.info("Downloading pretrained YOLOv8 model...")
            
            from ultralytics import YOLO
            
            # Download YOLOv8l model
            model = YOLO('yolov8l.pt')
            
            # Move to pretrained directory
            pretrained_dir = self.project_root / "models" / "pretrained"
            model_path = pretrained_dir / "yolov8l.pt"
            
            # The model is automatically downloaded to the current directory
            import shutil
            if Path("yolov8l.pt").exists():
                shutil.move("yolov8l.pt", model_path)
                logger.info(f"✅ Pretrained model saved: {model_path}")
                return True
            else:
                logger.warning("⚠️  Model download may have failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to download pretrained model: {e}")
            return False
    
    def test_camera(self) -> bool:
        """Test camera functionality."""
        try:
            logger.info("Testing camera access...")
            
            import cv2
            
            # Try to open default camera
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                logger.warning("⚠️  Default camera (ID: 0) not accessible")
                return False
            
            # Try to read a frame
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                logger.info("✅ Camera test passed")
                return True
            else:
                logger.warning("⚠️  Camera accessible but frame capture failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Camera test failed: {e}")
            return False
    
    def verify_installation(self) -> bool:
        """Verify that installation was successful."""
        try:
            logger.info("Verifying installation...")
            
            # Test core imports
            import torch
            import ultralytics
            import cv2
            import customtkinter
            
            # Test CUDA if available
            if torch.cuda.is_available():
                logger.info(f"✅ CUDA verification: {torch.cuda.get_device_name(0)}")
            
            # Test project imports
            sys.path.append(str(self.project_root))
            from src.core.model import HelmetDetector
            from src.desktop_app.main_window import SafetyDetectionApp
            
            logger.info("✅ Installation verification passed")
            return True
            
        except ImportError as e:
            logger.error(f"❌ Import verification failed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return False
    
    def create_launch_script(self) -> bool:
        """Create convenient launch scripts."""
        try:
            # Windows batch script
            if self.platform == "Windows":
                batch_content = f"""@echo off
cd /d "{self.project_root}"
python main.py
pause
"""
                with open(self.project_root / "launch_app.bat", "w") as f:
                    f.write(batch_content)
                
                # Training script
                train_batch_content = f"""@echo off
cd /d "{self.project_root}"
python src/scripts/train_model.py --validate --export
pause
"""
                with open(self.project_root / "train_model.bat", "w") as f:
                    f.write(train_batch_content)
                
                logger.info("✅ Windows launch scripts created")
            
            # Shell script for Unix-like systems
            else:
                shell_content = f"""#!/bin/bash
cd "{self.project_root}"
python main.py
"""
                script_path = self.project_root / "launch_app.sh"
                with open(script_path, "w") as f:
                    f.write(shell_content)
                
                # Make executable
                os.chmod(script_path, 0o755)
                
                logger.info("✅ Shell launch script created")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create launch scripts: {e}")
            return False
    
    def run_setup(self) -> bool:
        """Run complete setup process."""
        logger.info("🚀 Starting HSE Vision setup...")
        
        # Check prerequisites
        if not self.check_python_version():
            return False
        
        # Setup directories
        if not self.setup_directories():
            return False
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        # Check GPU support
        gpu_info = self.check_gpu_support()
        
        # Download pretrained model
        if not self.download_pretrained_model():
            logger.warning("⚠️  Pretrained model download failed - you can download manually")
        
        # Test camera
        if not self.test_camera():
            logger.warning("⚠️  Camera test failed - check camera permissions")
        
        # Verify installation
        if not self.verify_installation():
            return False
        
        # Create launch scripts
        if not self.create_launch_script():
            logger.warning("⚠️  Launch script creation failed")
        
        # Final summary
        logger.info("🎉 Setup completed successfully!")
        logger.info("\n" + "="*50)
        logger.info("SETUP SUMMARY")
        logger.info("="*50)
        logger.info(f"✅ Python {self.python_version.major}.{self.python_version.minor}")
        logger.info(f"✅ Dependencies installed")
        logger.info(f"✅ Project structure created")
        
        if gpu_info['cuda_available']:
            logger.info(f"✅ GPU: {gpu_info['gpu_name']}")
        else:
            logger.info("⚠️  GPU: Not available (CPU mode)")
        
        logger.info("\nNEXT STEPS:")
        logger.info("1. Run the application: python main.py")
        logger.info("2. Or use launch script: launch_app.bat (Windows)")
        logger.info("3. Train custom model: python src/scripts/train_model.py")
        logger.info("4. Check README.md for detailed usage instructions")
        logger.info("="*50)
        
        return True

def main():
    """Main setup function."""
    setup = HSEVisionSetup()
    
    try:
        success = setup.run_setup()
        
        if success:
            print("\n🎉 HSE Vision setup completed successfully!")
            print("You can now run the application with: python main.py")
        else:
            print("\n❌ Setup failed. Please check the logs above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Setup failed with unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()