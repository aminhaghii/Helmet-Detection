"""
File: model.py
Purpose: YOLOv8 model wrapper for helmet detection with GPU optimization
Author: HSE Vision Team
Date: 2024
Project: Construction Safety Detection System
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import yaml

logger = logging.getLogger(__name__)


class HelmetDetector:
    """
    YOLOv8-based helmet detection model with GPU optimization.
    """

    def __init__(
        self, model_path: Optional[str] = None, config_path: Optional[str] = None
    ):
        """
        Initialize the helmet detector.

        Args:
            model_path: Path to trained model weights
            config_path: Path to model configuration file
        """
        self.model = None
        self.device = self._get_best_device()
        self.config = self._load_config(config_path)
        self.class_names = ["helm", "no-helm"]
        self.model_loaded = False

        logger.info(f"Initialized HelmetDetector on device: {self.device}")

        if model_path:
            self.load_model(model_path)

    def _get_best_device(self) -> str:
        """Get the best available device for inference."""
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
            gpu_name = torch.cuda.get_device_name()
            logger.info(f"Using GPU: {gpu_name}")
            return device
        else:
            logger.warning("CUDA not available, using CPU")
            return "cpu"

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load model configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config.get("model", {})
        return {}

    def load_model(self, model_path: str):
        """
        Load trained model weights.

        Args:
            model_path: Path to model weights file

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            logger.info(f"Loading model from: {model_path}")
            self.model = YOLO(str(model_path))

            # Move model to GPU if available
            if self.device.startswith("cuda"):
                self.model.to(self.device)

                # Enable optimizations for inference
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False

            self.model_loaded = True
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def load_pretrained(self, model_name: str = "yolov8l.pt"):
        """
        Load pretrained YOLOv8 model.

        Args:
            model_name: Name of pretrained model
        """
        try:
            logger.info(f"Loading pretrained model: {model_name}")
            self.model = YOLO(model_name)

            if self.device.startswith("cuda"):
                self.model.to(self.device)
                torch.backends.cudnn.benchmark = True

            self.model_loaded = True
            logger.info("Pretrained model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            raise RuntimeError(f"Pretrained model loading failed: {e}")

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        max_det: int = 300,
    ) -> List[Dict]:
        """
        Perform helmet detection on image.

        Args:
            image: Input image (BGR format)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            max_det: Maximum number of detections

        Returns:
            List of detection dictionaries

        Raises:
            RuntimeError: If model is not loaded or prediction fails
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Run inference
            results = self.model.predict(
                image,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_det,
                device=self.device,
                verbose=False,
            )

            # Parse results
            detections = []
            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)

                    for i in range(len(boxes)):
                        detection = {
                            "bbox": boxes[i].tolist(),  # [x1, y1, x2, y2]
                            "confidence": float(confidences[i]),
                            "class_id": int(classes[i]),
                            "class_name": (
                                self.class_names[classes[i]]
                                if classes[i] < len(self.class_names)
                                else "unknown"
                            ),
                        }
                        detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    def predict_batch(
        self,
        images: List[np.ndarray],
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ) -> List[List[Dict]]:
        """
        Perform batch prediction on multiple images.

        Args:
            images: List of input images
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold

        Returns:
            List of detection lists for each image
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            results = self.model.predict(
                images,
                conf=conf_threshold,
                iou=iou_threshold,
                device=self.device,
                verbose=False,
            )

            batch_detections = []
            for result in results:
                detections = []
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)

                    for i in range(len(boxes)):
                        detection = {
                            "bbox": boxes[i].tolist(),
                            "confidence": float(confidences[i]),
                            "class_id": int(classes[i]),
                            "class_name": (
                                self.class_names[classes[i]]
                                if classes[i] < len(self.class_names)
                                else "unknown"
                            ),
                        }
                        detections.append(detection)

                batch_detections.append(detections)

            return batch_detections

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise RuntimeError(f"Batch prediction failed: {e}")

    def export_tensorrt(
        self,
        output_path: str,
        imgsz: Tuple[int, int] = (640, 640),
        half: bool = True,
        workspace: int = 4,
    ) -> str:
        """
        Export model to TensorRT format for optimized inference.

        Args:
            output_path: Output path for TensorRT engine
            imgsz: Input image size
            half: Use FP16 precision
            workspace: Workspace size in GB

        Returns:
            Path to exported TensorRT engine

        Raises:
            RuntimeError: If export fails
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            logger.info("Exporting model to TensorRT...")

            # Export to TensorRT
            exported_path = self.model.export(
                format="engine",
                imgsz=imgsz,
                half=half,
                workspace=workspace,
                verbose=True,
            )

            logger.info(f"Model exported to TensorRT: {exported_path}")
            return exported_path

        except Exception as e:
            logger.error(f"TensorRT export failed: {e}")
            raise RuntimeError(f"TensorRT export failed: {e}")

    def get_model_info(self) -> Dict:
        """
        Get model information and statistics.

        Returns:
            Dictionary with model information
        """
        if not self.model_loaded:
            return {"status": "Model not loaded"}

        info = {
            "model_type": "YOLOv8",
            "device": self.device,
            "classes": self.class_names,
            "num_classes": len(self.class_names),
            "model_loaded": self.model_loaded,
        }

        if hasattr(self.model, "model"):
            try:
                # Get model parameters count
                total_params = sum(p.numel() for p in self.model.model.parameters())
                trainable_params = sum(
                    p.numel() for p in self.model.model.parameters() if p.requires_grad
                )

                info.update(
                    {
                        "total_parameters": total_params,
                        "trainable_parameters": trainable_params,
                        "model_size_mb": total_params
                        * 4
                        / (1024 * 1024),  # Assuming float32
                    }
                )
            except:
                pass

        return info

    def benchmark(
        self, image_size: Tuple[int, int] = (640, 640), num_iterations: int = 100
    ) -> Dict:
        """
        Benchmark model performance.

        Args:
            image_size: Input image size for benchmarking
            num_iterations: Number of iterations to run

        Returns:
            Benchmark results dictionary
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import time

        # Create dummy image
        dummy_image = np.random.randint(
            0, 255, (image_size[1], image_size[0], 3), dtype=np.uint8
        )

        # Warmup
        for _ in range(10):
            self.predict(dummy_image)

        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            self.predict(dummy_image)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        fps = 1.0 / avg_time

        results = {
            "total_time": total_time,
            "average_time": avg_time,
            "fps": fps,
            "iterations": num_iterations,
            "image_size": image_size,
        }

        logger.info(
            f"Benchmark results: {fps:.2f} FPS, {avg_time*1000:.2f}ms per frame"
        )
        return results
