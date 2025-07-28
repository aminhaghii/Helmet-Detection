"""
Real-time Helmet Detection using YOLOv8s on Laptop Camera
This script tests the trained helmet detection model on your laptop camera.
"""

import os
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class HelmetDetector:
    def __init__(self, model_path):
        """Initialize the helmet detector with trained model"""
        self.model = YOLO(model_path)
        self.class_names = ["helm", "no-helm"]
        self.colors = {
            "helm": (0, 255, 0),  # Green for helmet
            "no-helm": (0, 0, 255),  # Red for no helmet
        }

    def detect_helmets(self, frame):
        """Detect helmets in a frame"""
        results = self.model(frame, conf=0.5, iou=0.45)
        return results[0]

    def draw_detections(self, frame, results):
        """Draw bounding boxes and labels on frame"""
        annotated_frame = frame.copy()

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = box.astype(int)
                class_name = self.class_names[cls]
                color = self.colors[class_name]

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # Draw label with confidence
                label = f"{class_name}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

                # Background for text
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    color,
                    -1,
                )

                # Text
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        return annotated_frame

    def add_info_overlay(self, frame, fps, detection_count):
        """Add information overlay to frame"""
        # FPS counter
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Detection count
        cv2.putText(
            frame,
            f"Detections: {detection_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Instructions
        cv2.putText(
            frame,
            "Press 'q' to quit, 's' to save screenshot",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        return frame


def main():
    # Path to the best trained model
    model_path = "models/trained/helmet_detection_20250727_152811/weights/best.pt"

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Available models:")
        trained_dir = "models/trained"
        if os.path.exists(trained_dir):
            for folder in os.listdir(trained_dir):
                weights_path = os.path.join(trained_dir, folder, "weights", "best.pt")
                if os.path.exists(weights_path):
                    print(f"  - {weights_path}")
        return

    print(f"🚀 Loading model from: {model_path}")

    # Initialize detector
    try:
        detector = HelmetDetector(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Initialize camera
    print("📷 Initializing camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("✅ Camera initialized successfully!")
    print("\n🎯 Starting helmet detection...")
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save screenshot")
    print("  - Press 'r' to reset detection counter")

    # Performance tracking
    fps_counter = 0
    fps_start_time = time.time()
    detection_counter = 0
    screenshot_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error reading from camera")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            start_time = time.time()

            # Detect helmets
            results = detector.detect_helmets(frame)

            # Count detections
            current_detections = len(results.boxes) if results.boxes is not None else 0
            if current_detections > 0:
                detection_counter += current_detections

            # Draw detections
            annotated_frame = detector.draw_detections(frame, results)

            # Calculate FPS
            fps_counter += 1
            if fps_counter >= 30:  # Update FPS every 30 frames
                fps_end_time = time.time()
                fps = fps_counter / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                fps_counter = 0
            else:
                fps = 0

            # Add info overlay
            annotated_frame = detector.add_info_overlay(
                annotated_frame, fps, detection_counter
            )

            # Display frame
            cv2.imshow("Helmet Detection - YOLOv8s", annotated_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n👋 Quitting...")
                break
            elif key == ord("s"):
                screenshot_counter += 1
                screenshot_name = f"screenshot_{screenshot_counter:03d}.jpg"
                cv2.imwrite(screenshot_name, annotated_frame)
                print(f"📸 Screenshot saved: {screenshot_name}")
            elif key == ord("r"):
                detection_counter = 0
                print("🔄 Detection counter reset")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")

    except Exception as e:
        print(f"❌ Error during detection: {e}")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n📊 Final Statistics:")
        print(f"  - Total detections: {detection_counter}")
        print(f"  - Screenshots saved: {screenshot_counter}")
        print("✅ Camera test completed!")


if __name__ == "__main__":
    main()
