"""
File: camera_handler.py
Purpose: Camera interface and video capture management
Author: HSE Vision Team
Date: 2024
Project: Construction Safety Detection System
"""

import logging
import threading
import time
from queue import Queue
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraHandler:
    """
    Handles camera capture and video stream management.
    """

    def __init__(
        self,
        camera_id: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
    ):
        """
        Initialize camera handler.

        Args:
            camera_id: Camera device ID
            resolution: Camera resolution (width, height)
            fps: Target FPS
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps

        self.cap = None
        self.is_running = False
        self.capture_thread = None

        # Frame buffer
        self.frame_queue = Queue(maxsize=2)
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # Statistics
        self.frame_count = 0
        self.start_time = None
        self.actual_fps = 0.0

        # Callbacks
        self.frame_callback = None

        logger.info(
            f"CameraHandler initialized: ID={camera_id}, Resolution={resolution}"
        )

    def initialize_camera(self) -> bool:
        """
        Initialize camera capture.

        Returns:
            True if camera initialized successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)

            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency

            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            logger.info(
                f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS"
            )

            # Test frame capture
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to capture test frame")
                self.cap.release()
                return False

            logger.info("Camera initialization successful")
            return True

        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False

    def start_capture(self, frame_callback: Optional[Callable] = None):
        """
        Start camera capture in separate thread.

        Args:
            frame_callback: Optional callback function for each frame
        """
        if self.is_running:
            logger.warning("Camera capture already running")
            return

        if not self.cap or not self.cap.isOpened():
            if not self.initialize_camera():
                raise RuntimeError("Failed to initialize camera")

        self.frame_callback = frame_callback
        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0

        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        logger.info("Camera capture started")

    def stop_capture(self):
        """Stop camera capture."""
        if not self.is_running:
            return

        self.is_running = False

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)

        if self.cap:
            self.cap.release()
            self.cap = None

        logger.info("Camera capture stopped")

    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        frame_interval = 1.0 / self.fps
        last_frame_time = 0

        while self.is_running:
            try:
                current_time = time.time()

                # Control frame rate
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.001)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.01)
                    continue

                # Update statistics
                self.frame_count += 1
                elapsed_time = current_time - self.start_time
                if elapsed_time > 0:
                    self.actual_fps = self.frame_count / elapsed_time

                # Store latest frame
                with self.frame_lock:
                    self.latest_frame = frame.copy()

                # Add to queue (non-blocking)
                try:
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()  # Remove old frame
                    self.frame_queue.put_nowait(frame.copy())
                except:
                    pass

                # Call frame callback
                if self.frame_callback:
                    try:
                        self.frame_callback(frame.copy())
                    except Exception as e:
                        logger.error(f"Frame callback error: {e}")

                last_frame_time = current_time

            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                time.sleep(0.01)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest captured frame.

        Returns:
            Latest frame or None if no frame available
        """
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_frame_from_queue(self) -> Optional[np.ndarray]:
        """
        Get frame from queue (non-blocking).

        Returns:
            Frame from queue or None if queue is empty
        """
        try:
            return self.frame_queue.get_nowait()
        except:
            return None

    def get_camera_info(self) -> dict:
        """
        Get camera information and statistics.

        Returns:
            Dictionary with camera information
        """
        info = {
            "camera_id": self.camera_id,
            "target_resolution": self.resolution,
            "target_fps": self.fps,
            "is_running": self.is_running,
            "frame_count": self.frame_count,
            "actual_fps": self.actual_fps,
        }

        if self.cap and self.cap.isOpened():
            info.update(
                {
                    "actual_width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "actual_height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "backend": self.cap.getBackendName(),
                }
            )

        return info

    def set_resolution(self, width: int, height: int) -> bool:
        """
        Change camera resolution.

        Args:
            width: New width
            height: New height

        Returns:
            True if resolution changed successfully
        """
        if not self.cap or not self.cap.isOpened():
            self.resolution = (width, height)
            return True

        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.resolution = (actual_width, actual_height)
            logger.info(f"Resolution changed to: {actual_width}x{actual_height}")
            return True

        except Exception as e:
            logger.error(f"Failed to change resolution: {e}")
            return False

    def capture_single_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame (for testing or manual capture).

        Returns:
            Captured frame or None if capture failed
        """
        if not self.cap or not self.cap.isOpened():
            if not self.initialize_camera():
                return None

        try:
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                logger.warning("Failed to capture single frame")
                return None
        except Exception as e:
            logger.error(f"Single frame capture error: {e}")
            return None

    def test_camera(self) -> bool:
        """
        Test camera functionality.

        Returns:
            True if camera test passed
        """
        try:
            logger.info("Testing camera...")

            if not self.initialize_camera():
                return False

            # Capture a few test frames
            for i in range(5):
                ret, frame = self.cap.read()
                if not ret:
                    logger.error(f"Test frame {i+1} capture failed")
                    return False

                if frame is None or frame.size == 0:
                    logger.error(f"Test frame {i+1} is empty")
                    return False

            self.cap.release()
            self.cap = None

            logger.info("Camera test passed")
            return True

        except Exception as e:
            logger.error(f"Camera test failed: {e}")
            return False

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_capture()
