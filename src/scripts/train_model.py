"""
File: train_model.py
Purpose: Model training script for helmet detection
Author: HSE Vision Team
Date: 2024
Project: Construction Safety Detection System
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
import wandb
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import logger setup function
def setup_logger():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Handles model training for helmet detection.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize model trainer.
        
        Args:
            config_path: Path to model configuration file
        """
        # Setup logging
        setup_logger()
        logger.info("Initializing Model Trainer")
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Setup paths
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "Data"
        self.models_dir = self.project_root / "models"
        self.logs_dir = self.project_root / "logs"
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        (self.models_dir / "trained").mkdir(exist_ok=True)
        (self.models_dir / "optimized").mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize model
        self.model = None
        self.unified_data_path = None
        
        logger.info("Model Trainer initialized successfully")
    
    def load_config(self, config_path: str = None) -> dict:
        """
        Load training configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "train_config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from: {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> dict:
        """Get default training configuration."""
        return {
            'model': {
                'architecture': 'yolov8l',
                'input_size': 640,
                'pretrained': True
            },
            'training': {
                'epochs': 300,
                'batch_size': 16,
                'learning_rate': 0.01,
                'optimizer': 'AdamW',
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'workers': 8,
                'patience': 50,
                'save_period': 10
            },
            'data_augmentation': {
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.0
            },
            'validation': {
                'val_split': 0.2,
                'save_json': True,
                'save_hybrid': False,
                'conf_threshold': 0.001,
                'iou_threshold': 0.6,
                'max_det': 300
            }
        }
    
    def prepare_datasets(self) -> str:
        """
        Prepare and unify datasets for training.
        
        Returns:
            Path to unified dataset configuration
        """
        logger.info("Preparing datasets for training...")
        
        try:
            # Check if unified dataset already exists
            unified_config_path = self.project_root / "unified_dataset" / "data.yaml"
            
            if unified_config_path.exists():
                logger.info(f"Using existing unified dataset: {unified_config_path}")
                self.unified_data_path = str(unified_config_path)
                return str(unified_config_path)
            else:
                logger.error("Unified dataset not found. Please run dataset_unifier.py first.")
                raise FileNotFoundError("Unified dataset not found")
            
        except Exception as e:
            logger.error(f"Failed to prepare datasets: {e}")
            raise
    
    def initialize_model(self):
        """Initialize YOLO model for training."""
        try:
            model_name = self.config['model']['name']
            pretrained = self.config['model']['pretrained']
            
            logger.info(f"Initializing {model_name} model (pretrained: {pretrained})")
            
            # Load model
            if pretrained:
                self.model = YOLO(f"{model_name}.pt")
            else:
                self.model = YOLO(f"{model_name}.yaml")
            
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def setup_wandb(self, project_name: str = "hse-vision-helmet-detection"):
        """
        Setup Weights & Biases logging.
        
        Args:
            project_name: W&B project name
        """
        try:
            # Initialize wandb
            wandb.init(
                project=project_name,
                config=self.config,
                name=f"helmet-detection-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=["helmet-detection", "yolov8", "construction-safety"]
            )
            
            logger.info("W&B logging initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
    
    def train(self, 
              data_config_path: str = None,
              resume: bool = False,
              use_wandb: bool = True) -> str:
        """
        Train the model.
        
        Args:
            data_config_path: Path to data configuration file
            resume: Whether to resume training from checkpoint
            use_wandb: Whether to use W&B logging
            
        Returns:
            Path to trained model
        """
        try:
            logger.info("Starting model training...")
            
            # Prepare datasets if not provided
            if data_config_path is None:
                data_config_path = self.prepare_datasets()
            
            # Initialize model
            if self.model is None:
                self.initialize_model()
            
            # Setup W&B if requested
            if use_wandb:
                self.setup_wandb()
            
            # Prepare training arguments
            train_args = {
                'data': data_config_path,
                'epochs': self.config['training']['epochs'],
                'batch': self.config['training']['batch_size'],
                'imgsz': self.config['model']['input_size'],
                'device': self.config['training']['device'],
                'workers': self.config['training']['workers'],
                'patience': self.config['training']['patience'],
                'save_period': self.config['training']['save_period'],
                'project': str(self.models_dir / "trained"),
                'name': f"helmet_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'exist_ok': True,
                'pretrained': self.config['model']['pretrained'],
                'optimizer': self.config['training']['optimizer'],
                'lr0': self.config['training']['learning_rate'],
                'verbose': True,
                'save': True,
                'save_json': self.config['validation']['save_json'],
                'conf': self.config['validation']['conf_threshold'],
                'iou': self.config['validation']['iou_threshold'],
                'max_det': self.config['validation']['max_det']
            }
            
            # Add data augmentation parameters
            train_args.update(self.config['data_augmentation'])
            
            # Resume training if requested
            if resume:
                # Find latest checkpoint
                checkpoints = list((self.models_dir / "trained").glob("**/last.pt"))
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=os.path.getctime)
                    train_args['resume'] = str(latest_checkpoint)
                    logger.info(f"Resuming training from: {latest_checkpoint}")
            
            logger.info("Training arguments:")
            for key, value in train_args.items():
                logger.info(f"  {key}: {value}")
            
            # Start training
            logger.info("Starting training process...")
            results = self.model.train(**train_args)
            
            # Get best model path
            best_model_path = results.save_dir / "weights" / "best.pt"
            
            logger.info(f"Training completed successfully!")
            logger.info(f"Best model saved at: {best_model_path}")
            
            # Log final metrics
            if hasattr(results, 'results_dict'):
                logger.info("Final training metrics:")
                for metric, value in results.results_dict.items():
                    logger.info(f"  {metric}: {value}")
            
            return str(best_model_path)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Close W&B run
            if use_wandb:
                wandb.finish()
    
    def validate_model(self, model_path: str, data_config_path: str = None) -> dict:
        """
        Validate trained model.
        
        Args:
            model_path: Path to trained model
            data_config_path: Path to data configuration
            
        Returns:
            Validation results
        """
        try:
            logger.info(f"Validating model: {model_path}")
            
            # Load model
            model = YOLO(model_path)
            
            # Use unified data config if not provided
            if data_config_path is None:
                data_config_path = self.unified_data_path
            
            # Run validation
            results = model.val(
                data=data_config_path,
                imgsz=self.config['model']['input_size'],
                batch=self.config['training']['batch_size'],
                conf=self.config['validation']['conf_threshold'],
                iou=self.config['validation']['iou_threshold'],
                max_det=self.config['validation']['max_det'],
                save_json=True,
                save_hybrid=False,
                verbose=True
            )
            
            # Extract metrics
            metrics = {
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'f1_score': 2 * (float(results.box.mp) * float(results.box.mr)) / (float(results.box.mp) + float(results.box.mr))
            }
            
            logger.info("Validation results:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
    
    def export_model(self, 
                    model_path: str, 
                    export_formats: list = ['onnx', 'engine']) -> dict:
        """
        Export model to different formats for deployment.
        
        Args:
            model_path: Path to trained model
            export_formats: List of export formats
            
        Returns:
            Dictionary of exported model paths
        """
        try:
            logger.info(f"Exporting model: {model_path}")
            
            # Load model
            model = YOLO(model_path)
            
            exported_paths = {}
            
            for format_name in export_formats:
                try:
                    logger.info(f"Exporting to {format_name}...")
                    
                    if format_name == 'onnx':
                        export_path = model.export(
                            format='onnx',
                            imgsz=self.config['model']['input_size'],
                            dynamic=False,
                            simplify=True
                        )
                    elif format_name == 'engine':
                        # TensorRT export
                        export_path = model.export(
                            format='engine',
                            imgsz=self.config['model']['input_size'],
                            half=True,  # FP16 precision
                            dynamic=False,
                            simplify=True,
                            workspace=4  # 4GB workspace
                        )
                    elif format_name == 'torchscript':
                        export_path = model.export(
                            format='torchscript',
                            imgsz=self.config['model']['input_size']
                        )
                    else:
                        logger.warning(f"Unsupported export format: {format_name}")
                        continue
                    
                    exported_paths[format_name] = export_path
                    logger.info(f"Exported {format_name}: {export_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to export {format_name}: {e}")
            
            return exported_paths
            
        except Exception as e:
            logger.error(f"Model export failed: {e}")
            raise
    
    def benchmark_model(self, model_path: str, num_runs: int = 100) -> dict:
        """
        Benchmark model performance.
        
        Args:
            model_path: Path to model
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        try:
            logger.info(f"Benchmarking model: {model_path}")
            
            # Load model
            model = YOLO(model_path)
            
            # Run benchmark
            results = model.benchmark(
                imgsz=self.config['model']['input_size'],
                half=True,
                device=self.config['training']['device'],
                verbose=True
            )
            
            logger.info("Benchmark results:")
            logger.info(f"  Speed: {results}")
            
            return results
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            raise

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train helmet detection model")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data', type=str, help='Path to data configuration file')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--no-wandb', action='store_true', help='Disable W&B logging')
    parser.add_argument('--validate', action='store_true', help='Run validation after training')
    parser.add_argument('--export', action='store_true', help='Export model after training')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark model after training')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(config_path=args.config)
        
        # Train model
        best_model_path = trainer.train(
            data_config_path=args.data,
            resume=args.resume,
            use_wandb=not args.no_wandb
        )
        
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Best model saved at: {best_model_path}")
        
        # Run validation if requested
        if args.validate:
            print("\n🔍 Running validation...")
            metrics = trainer.validate_model(best_model_path)
            print("📊 Validation metrics:")
            for metric, value in metrics.items():
                print(f"   {metric}: {value:.4f}")
        
        # Export model if requested
        if args.export:
            print("\n📤 Exporting model...")
            exported_paths = trainer.export_model(best_model_path)
            print("📁 Exported models:")
            for format_name, path in exported_paths.items():
                print(f"   {format_name}: {path}")
        
        # Benchmark model if requested
        if args.benchmark:
            print("\n⚡ Benchmarking model...")
            benchmark_results = trainer.benchmark_model(best_model_path)
            print(f"🚀 Benchmark results: {benchmark_results}")
        
        print(f"\n🎉 All tasks completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()