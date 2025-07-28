"""
File: inference.py
Purpose: High-performance inference engine with TensorRT optimization
Author: HSE Vision Team
Date: 2024
Project: Construction Safety Detection System
"""

import cv2
import numpy as np
import torch
import time
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import threading
from queue import Queue

from .model import HelmetDetector
from ..utils.logger import PerformanceLogger

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    High-performance inference engine optimized for real-time detection.
    """

    def __init__(self, model_path: str, config: Dict = None, use_tensorrt: bool = True):
        """
        Initialize inference engine.

        Args:
            model_path: Path to model weights
            config: Configuration dictionary
            use_tensorrt: Whether to use TensorRT optimization
        """
        self.model_path = model_path
        self.config = config or {}
        self.use_tensorrt = use_tensorrt

        # Performance monitoring
        self.perf_logger = PerformanceLogger()
        self.frame_times = []
        self.detection_counts = []

        # Threading for real-time processing
        self.use_threading = self.config.get("use_threading", True)
        self.input_queue = Queue(maxsize=2)
        self.output_queue = Queue(maxsize=2)
        self.processing_thread = None
        self.stop_processing = False

        # Initialize model
        self.detector = HelmetDetector()
        self._load_model()

        logger.info("InferenceEngine initialized")

    def _load_model(self):
        """Load and optimize model."""
        try:
            # Load model
            if Path(self.model_path).exists():
                self.detector.load_model(self.model_path)
            else:
                logger.warning(
                    f"Model file not found: {self.model_path}, loading pretrained"
                )
                self.detector.load_pretrained("yolov8l.pt")

            # Export to TensorRT if requested and not already exported
            if self.use_tensorrt and torch.cuda.is_available():
                tensorrt_path = Path(self.model_path).with_suffix(".engine")
                if not tensorrt_path.exists():
                    logger.info("Exporting model to TensorRT...")
                    try:
                        self.detector.export_tensorrt(
                            str(tensorrt_path), imgsz=(640, 640), half=True, workspace=4
                        )
                        # Reload with TensorRT engine
                        self.detector.load_model(str(tensorrt_path))
                    except Exception as e:
                        logger.warning(
                            f"TensorRT export failed: {e}, using PyTorch model"
                        )
                else:
                    # Load existing TensorRT engine
                    self.detector.load_model(str(tensorrt_path))

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    def predict_single(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ) -> Tuple[List[Dict], float]:
        """
        Perform single image inference.

        Args:
            image: Input image
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold

        Returns:
            Tuple of (detections, inference_time)
        """
        start_time = time.time()

        try:
            detections = self.detector.predict(
                image, conf_threshold=conf_threshold, iou_threshold=iou_threshold
            )

            inference_time = (time.time() - start_time) * 1000  # ms

            # Log performance
            self.perf_logger.log_inference_time(inference_time)
            self.perf_logger.log_detection_count(len(detections))

            # Update statistics
            self.frame_times.append(inference_time)
            self.detection_counts.append(len(detections))

            # Keep only last 100 measurements
            if len(self.frame_times) > 100:
                self.frame_times.pop(0)
                self.detection_counts.pop(0)

            return detections, inference_time

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return [], 0.0

    def start_threaded_processing(self):
        """Start threaded processing for real-time inference."""
        if not self.use_threading:
            return

        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.stop_processing = False
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            logger.info("Started threaded processing")

    def stop_threaded_processing(self):
        """Stop threaded processing."""
        self.stop_processing = True
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
            logger.info("Stopped threaded processing")

    def _processing_loop(self):
        """Main processing loop for threaded inference."""
        while not self.stop_processing:
            try:
                # Get input from queue
                if not self.input_queue.empty():
                    frame_data = self.input_queue.get_nowait()
                    frame, conf_threshold, iou_threshold = frame_data

                    # Process frame
                    detections, inference_time = self.predict_single(
                        frame, conf_threshold, iou_threshold
                    )

                    # Put result in output queue
                    if not self.output_queue.full():
                        self.output_queue.put_nowait((detections, inference_time))

                else:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting

            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                time.sleep(0.01)

    def predict_async(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ) -> bool:
        """
        Submit frame for asynchronous processing.

        Args:
            image: Input image
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold

        Returns:
            True if frame was submitted, False if queue is full
        """
        if not self.use_threading:
            return False

        try:
            self.input_queue.put_nowait((image, conf_threshold, iou_threshold))
            return True
        except:
            return False

    def get_async_result(self) -> Optional[Tuple[List[Dict], float]]:
        """
        Get result from asynchronous processing.

        Returns:
            Tuple of (detections, inference_time) or None if no result available
        """
        if not self.use_threading:
            return None

        try:
            return self.output_queue.get_nowait()
        except:
            return None

    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        if not self.frame_times:
            return {}

        avg_time = np.mean(self.frame_times)
        min_time = np.min(self.frame_times)
        max_time = np.max(self.frame_times)
        fps = 1000.0 / avg_time if avg_time > 0 else 0

        avg_detections = np.mean(self.detection_counts) if self.detection_counts else 0

        stats = {
            "average_inference_time_ms": avg_time,
            "min_inference_time_ms": min_time,
            "max_inference_time_ms": max_time,
            "fps": fps,
            "average_detections": avg_detections,
            "total_frames_processed": len(self.frame_times),
        }

        # Add GPU stats if available
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
                gpu_memory_cached = torch.cuda.memory_reserved() / 1024**2  # MB
                stats.update(
                    {
                        "gpu_memory_allocated_mb": gpu_memory,
                        "gpu_memory_cached_mb": gpu_memory_cached,
                    }
                )
            except:
                pass

        return stats

    def benchmark(
        self, num_frames: int = 100, image_size: Tuple[int, int] = (640, 480)
    ) -> Dict:
        """
        Benchmark inference performance.

        Args:
            num_frames: Number of frames to process
            image_size: Size of test images

        Returns:
            Benchmark results
        """
        logger.info(f"Starting benchmark with {num_frames} frames...")

        # Create test image
        test_image = np.random.randint(
            0, 255, (image_size[1], image_size[0], 3), dtype=np.uint8
        )

        # Warmup
        for _ in range(10):
            self.predict_single(test_image)

        # Clear previous stats
        self.frame_times.clear()
        self.detection_counts.clear()

        # Benchmark
        start_time = time.time()
        for i in range(num_frames):
            self.predict_single(test_image)
            if i % 20 == 0:
                logger.info(f"Processed {i}/{num_frames} frames")

        total_time = time.time() - start_time

        # Calculate results
        results = {
            "total_time_seconds": total_time,
            "frames_processed": num_frames,
            "average_fps": num_frames / total_time,
            "image_size": image_size,
        }

        # Add detailed stats
        results.update(self.get_performance_stats())

        logger.info(f"Benchmark completed: {results['average_fps']:.2f} FPS")
        return results

    def optimize_for_realtime(self):
        """Apply optimizations for real-time performance."""
        if torch.cuda.is_available():
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            # Clear GPU cache
            torch.cuda.empty_cache()

            logger.info("Applied CUDA optimizations for real-time performance")

    def cleanup(self):
        """Cleanup resources."""
        self.stop_threaded_processing()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("InferenceEngine cleanup completed")
