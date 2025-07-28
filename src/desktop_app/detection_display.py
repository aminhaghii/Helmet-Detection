"""
File: detection_display.py
Purpose: Detection visualization and display utilities
Author: HSE Vision Team
Date: 2024
Project: Construction Safety Detection System
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import math

logger = logging.getLogger(__name__)


class DetectionDisplay:
    """
    Handles detection visualization and display.
    """

    def __init__(
        self,
        class_names: List[str] = ["helm", "no-helm"],
        colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ):
        """
        Initialize detection display.

        Args:
            class_names: List of class names
            colors: Color mapping for classes
        """
        self.class_names = class_names

        # Default colors (BGR format for OpenCV)
        if colors is None:
            self.colors = {
                "helm": (0, 255, 0),  # Green for helmet
                "no-helm": (0, 0, 255),  # Red for no helmet
                "background": (50, 50, 50),
                "text": (255, 255, 255),
                "alert": (0, 0, 255),
                "safe": (0, 255, 0),
                "warning": (0, 165, 255),
            }
        else:
            self.colors = colors

        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2

        # Display settings
        self.box_thickness = 2
        self.show_confidence = True
        self.show_class_names = True
        self.show_fps = True
        self.show_stats = True

        logger.info("DetectionDisplay initialized")

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        fps: float = 0.0,
        stats: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Draw detections on image.

        Args:
            image: Input image
            detections: List of detection dictionaries
            fps: Current FPS
            stats: Additional statistics to display

        Returns:
            Image with drawn detections
        """
        display_image = image.copy()

        # Draw detections
        for detection in detections:
            self._draw_single_detection(display_image, detection)

        # Draw overlay information
        if self.show_fps or self.show_stats:
            self._draw_overlay_info(display_image, fps, stats, len(detections))

        return display_image

    def _draw_single_detection(self, image: np.ndarray, detection: Dict):
        """
        Draw a single detection on the image.

        Args:
            image: Image to draw on
            detection: Detection dictionary with bbox, class, confidence
        """
        # Extract detection info
        bbox = detection.get("bbox", [])
        class_id = detection.get("class", 0)
        confidence = detection.get("confidence", 0.0)

        if len(bbox) != 4:
            return

        x1, y1, x2, y2 = map(int, bbox)
        class_name = (
            self.class_names[class_id]
            if class_id < len(self.class_names)
            else f"class_{class_id}"
        )

        # Get color for this class
        color = self.colors.get(class_name, (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.box_thickness)

        # Prepare label text
        label_parts = []
        if self.show_class_names:
            label_parts.append(class_name)
        if self.show_confidence:
            label_parts.append(f"{confidence:.2f}")

        label = " ".join(label_parts)

        if label:
            # Calculate label size
            (label_width, label_height), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, self.font_thickness
            )

            # Draw label background
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + label_height + 10
            cv2.rectangle(
                image,
                (x1, label_y - label_height - baseline),
                (x1 + label_width, label_y + baseline),
                color,
                -1,
            )

            # Draw label text
            cv2.putText(
                image,
                label,
                (x1, label_y - baseline),
                self.font,
                self.font_scale,
                (255, 255, 255),
                self.font_thickness,
            )

    def _draw_overlay_info(
        self, image: np.ndarray, fps: float, stats: Optional[Dict], detection_count: int
    ):
        """
        Draw overlay information on the image.

        Args:
            image: Image to draw on
            fps: Current FPS
            stats: Statistics dictionary
            detection_count: Number of detections
        """
        h, w = image.shape[:2]
        overlay_y = 30
        line_height = 25

        # Create semi-transparent overlay
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (10, 10),
            (300, overlay_y + line_height * 4),
            self.colors["background"],
            -1,
        )
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        # Draw FPS
        if self.show_fps:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                image,
                fps_text,
                (20, overlay_y),
                self.font,
                self.font_scale,
                self.colors["text"],
                self.font_thickness,
            )
            overlay_y += line_height

        # Draw detection count
        count_text = f"Detections: {detection_count}"
        cv2.putText(
            image,
            count_text,
            (20, overlay_y),
            self.font,
            self.font_scale,
            self.colors["text"],
            self.font_thickness,
        )
        overlay_y += line_height

        # Draw safety status
        safety_status = self._get_safety_status(detection_count, stats)
        status_color = (
            self.colors["safe"] if safety_status["is_safe"] else self.colors["alert"]
        )
        cv2.putText(
            image,
            f"Status: {safety_status['message']}",
            (20, overlay_y),
            self.font,
            self.font_scale,
            status_color,
            self.font_thickness,
        )
        overlay_y += line_height

        # Draw additional stats
        if stats and self.show_stats:
            if "inference_time" in stats:
                inference_text = f"Inference: {stats['inference_time']:.1f}ms"
                cv2.putText(
                    image,
                    inference_text,
                    (20, overlay_y),
                    self.font,
                    self.font_scale,
                    self.colors["text"],
                    self.font_thickness,
                )

    def _get_safety_status(self, detection_count: int, stats: Optional[Dict]) -> Dict:
        """
        Determine safety status based on detections.

        Args:
            detection_count: Number of detections
            stats: Statistics dictionary

        Returns:
            Dictionary with safety status information
        """
        if detection_count == 0:
            return {"is_safe": True, "message": "No workers detected"}

        # This is a simplified safety check
        # In a real application, you would analyze the specific detections
        # to determine helmet vs no-helmet ratios

        return {"is_safe": True, "message": f"{detection_count} workers detected"}

    def create_detection_summary(self, detections: List[Dict]) -> Dict:
        """
        Create a summary of detections.

        Args:
            detections: List of detection dictionaries

        Returns:
            Summary dictionary
        """
        summary = {
            "total_detections": len(detections),
            "class_counts": {},
            "confidence_stats": {"mean": 0.0, "min": 1.0, "max": 0.0},
            "safety_score": 0.0,
        }

        if not detections:
            return summary

        # Count classes
        confidences = []
        for detection in detections:
            class_id = detection.get("class", 0)
            confidence = detection.get("confidence", 0.0)

            class_name = (
                self.class_names[class_id]
                if class_id < len(self.class_names)
                else f"class_{class_id}"
            )
            summary["class_counts"][class_name] = (
                summary["class_counts"].get(class_name, 0) + 1
            )
            confidences.append(confidence)

        # Calculate confidence statistics
        if confidences:
            summary["confidence_stats"] = {
                "mean": np.mean(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences),
            }

        # Calculate safety score (simplified)
        helm_count = summary["class_counts"].get("helm", 0)
        no_helm_count = summary["class_counts"].get("no-helm", 0)
        total_workers = helm_count + no_helm_count

        if total_workers > 0:
            summary["safety_score"] = helm_count / total_workers
        else:
            summary["safety_score"] = 1.0

        return summary

    def draw_alert_overlay(
        self, image: np.ndarray, alert_message: str, alert_type: str = "warning"
    ) -> np.ndarray:
        """
        Draw alert overlay on image.

        Args:
            image: Input image
            alert_message: Alert message to display
            alert_type: Type of alert ('warning', 'danger', 'info')

        Returns:
            Image with alert overlay
        """
        h, w = image.shape[:2]
        overlay = image.copy()

        # Choose color based on alert type
        if alert_type == "danger":
            color = self.colors["alert"]
        elif alert_type == "warning":
            color = self.colors["warning"]
        else:
            color = self.colors["safe"]

        # Draw alert background
        alert_height = 60
        cv2.rectangle(overlay, (0, h - alert_height), (w, h), color, -1)
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)

        # Draw alert text
        font_scale = 0.8
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            alert_message, self.font, font_scale, thickness
        )

        text_x = (w - text_width) // 2
        text_y = h - (alert_height - text_height) // 2

        cv2.putText(
            image,
            alert_message,
            (text_x, text_y),
            self.font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

        return image

    def save_detection_image(
        self,
        image: np.ndarray,
        detections: List[Dict],
        output_path: str,
        include_timestamp: bool = True,
    ) -> bool:
        """
        Save image with detections drawn.

        Args:
            image: Input image
            detections: List of detections
            output_path: Output file path
            include_timestamp: Whether to include timestamp in filename

        Returns:
            True if saved successfully
        """
        try:
            # Draw detections
            display_image = self.draw_detections(image, detections)

            # Add timestamp to filename if requested
            if include_timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path_parts = output_path.rsplit(".", 1)
                if len(path_parts) == 2:
                    output_path = f"{path_parts[0]}_{timestamp}.{path_parts[1]}"
                else:
                    output_path = f"{output_path}_{timestamp}"

            # Save image
            success = cv2.imwrite(output_path, display_image)

            if success:
                logger.info(f"Detection image saved: {output_path}")
            else:
                logger.error(f"Failed to save detection image: {output_path}")

            return success

        except Exception as e:
            logger.error(f"Error saving detection image: {e}")
            return False

    def create_detection_grid(
        self,
        images_with_detections: List[Tuple[np.ndarray, List[Dict]]],
        grid_size: Tuple[int, int] = (2, 2),
    ) -> np.ndarray:
        """
        Create a grid of images with detections.

        Args:
            images_with_detections: List of (image, detections) tuples
            grid_size: Grid dimensions (rows, cols)

        Returns:
            Grid image
        """
        rows, cols = grid_size
        max_images = rows * cols

        if len(images_with_detections) > max_images:
            images_with_detections = images_with_detections[:max_images]

        # Process images
        processed_images = []
        for image, detections in images_with_detections:
            display_image = self.draw_detections(image, detections)
            processed_images.append(display_image)

        # Pad with empty images if needed
        while len(processed_images) < max_images:
            h, w = processed_images[0].shape[:2] if processed_images else (480, 640)
            empty_image = np.zeros((h, w, 3), dtype=np.uint8)
            processed_images.append(empty_image)

        # Create grid
        grid_rows = []
        for i in range(rows):
            row_images = processed_images[i * cols : (i + 1) * cols]
            grid_row = np.hstack(row_images)
            grid_rows.append(grid_row)

        grid_image = np.vstack(grid_rows)
        return grid_image
