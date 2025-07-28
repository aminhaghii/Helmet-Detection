"""
File: logger.py
Purpose: Centralized logging system for the Construction Safety Detection System
Author: HSE Vision Team
Date: 2024
Project: Construction Safety Detection System
"""

import logging
import logging.handlers
import os
from pathlib import Path
from datetime import datetime


def setup_logger(name="HSE_Vision", level=logging.INFO, log_file=None):
    """
    Set up a comprehensive logging system.

    Args:
        name (str): Logger name
        level (int): Logging level
        log_file (str): Optional log file path

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"hse_vision_{timestamp}.log"

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logger initialized: {name}")
    return logger


def get_logger(name="HSE_Vision"):
    """Get existing logger or create new one."""
    return logging.getLogger(name)


class PerformanceLogger:
    """Logger for performance metrics."""

    def __init__(self, logger_name="Performance"):
        self.logger = get_logger(logger_name)
        self.metrics = {}

    def log_fps(self, fps):
        """Log FPS performance."""
        self.logger.info(f"FPS: {fps:.2f}")
        self.metrics["fps"] = fps

    def log_inference_time(self, time_ms):
        """Log model inference time."""
        self.logger.info(f"Inference time: {time_ms:.2f}ms")
        self.metrics["inference_time"] = time_ms

    def log_detection_count(self, count):
        """Log number of detections."""
        self.logger.info(f"Detections: {count}")
        self.metrics["detection_count"] = count

    def log_gpu_usage(self, usage_percent, memory_mb):
        """Log GPU usage statistics."""
        self.logger.info(f"GPU Usage: {usage_percent:.1f}%, Memory: {memory_mb:.1f}MB")
        self.metrics["gpu_usage"] = usage_percent
        self.metrics["gpu_memory"] = memory_mb

    def get_metrics(self):
        """Get current metrics."""
        return self.metrics.copy()
