"""
File: main_window.py
Purpose: Main desktop application window with GUI
Author: HSE Vision Team
Date: 2024
Project: Construction Safety Detection System
"""

import logging
import os
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import customtkinter as ctk
import cv2
import numpy as np
import tkinter as tk
import yaml
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

from ..core.inference import InferenceEngine

# Import project modules
from ..core.model import HelmetDetector
from ..utils.logger import PerformanceLogger, setup_logger
from .camera_handler import CameraHandler
from .detection_display import DetectionDisplay

logger = logging.getLogger(__name__)


class SafetyDetectionApp:
    """
    Main desktop application for construction safety detection.
    """

    def __init__(self):
        """Initialize the application."""
        # Setup logging
        setup_logger()
        logger.info("Initializing Safety Detection App")

        # Load configurations
        self.load_configs()

        # Initialize components
        self.detector = None
        self.inference_engine = None
        self.camera_handler = None
        self.detection_display = DetectionDisplay()
        self.performance_logger = PerformanceLogger()

        # GUI variables
        self.is_detecting = False
        self.current_frame = None
        self.detection_results = []

        # Setup GUI
        self.setup_gui()

        # Initialize model
        self.initialize_model()

        logger.info("Safety Detection App initialized successfully")

    def load_configs(self):
        """Load configuration files."""
        try:
            # Load deployment config
            config_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "config",
                "deployment_config.yaml",
            )
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)

            # Load model config
            model_config_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "config", "model_config.yaml"
            )
            with open(model_config_path, "r") as f:
                self.model_config = yaml.safe_load(f)

            logger.info("Configurations loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
            # Use default configurations
            self.config = self.get_default_config()
            self.model_config = self.get_default_model_config()

    def get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "app": {"name": "HSE Vision", "version": "1.0.0"},
            "camera": {
                "device_id": 0,
                "resolution": {"width": 640, "height": 480},
                "fps": 30,
            },
            "detection": {
                "confidence_threshold": 0.5,
                "iou_threshold": 0.45,
                "max_detections": 100,
            },
            "ui": {"theme": "dark", "update_interval": 33},
        }

    def get_default_model_config(self) -> Dict:
        """Get default model configuration."""
        return {
            "model": {"architecture": "yolov8l", "input_size": 640},
            "training": {"device": "cuda"},
        }

    def setup_gui(self):
        """Setup the GUI interface."""
        # Set appearance mode and color theme
        ctk.set_appearance_mode(self.config["ui"]["theme"])
        ctk.set_default_color_theme("blue")

        # Create main window
        self.root = ctk.CTk()
        self.root.title(
            f"{self.config['app']['name']} v{self.config['app']['version']}"
        )
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Create main layout
        self.create_main_layout()

        # Create menu bar
        self.create_menu_bar()

        # Setup status bar
        self.create_status_bar()

        logger.info("GUI setup completed")

    def create_main_layout(self):
        """Create the main layout of the application."""
        # Main container
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Left panel for video display
        self.video_frame = ctk.CTkFrame(self.main_frame)
        self.video_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        # Video display label
        self.video_label = ctk.CTkLabel(
            self.video_frame, text="Camera Feed", font=("Arial", 16)
        )
        self.video_label.pack(pady=10)

        # Canvas for video display
        self.video_canvas = tk.Canvas(
            self.video_frame, bg="black", width=640, height=480
        )
        self.video_canvas.pack(pady=10, padx=10)

        # Right panel for controls and information
        self.control_frame = ctk.CTkFrame(self.main_frame, width=300)
        self.control_frame.pack(side="right", fill="y", padx=(5, 0))
        self.control_frame.pack_propagate(False)

        # Create control sections
        self.create_camera_controls()
        self.create_detection_controls()
        self.create_model_controls()
        self.create_statistics_display()

    def create_camera_controls(self):
        """Create camera control section."""
        camera_section = ctk.CTkFrame(self.control_frame)
        camera_section.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            camera_section, text="Camera Controls", font=("Arial", 14, "bold")
        ).pack(pady=5)

        # Camera selection
        self.camera_var = tk.StringVar(value="0")
        ctk.CTkLabel(camera_section, text="Camera ID:").pack(anchor="w", padx=10)
        self.camera_entry = ctk.CTkEntry(
            camera_section, textvariable=self.camera_var, width=100
        )
        self.camera_entry.pack(pady=2, padx=10)

        # Resolution selection
        ctk.CTkLabel(camera_section, text="Resolution:").pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self.resolution_var = tk.StringVar(value="640x480")
        resolution_combo = ctk.CTkComboBox(
            camera_section,
            values=["640x480", "800x600", "1024x768", "1280x720"],
            variable=self.resolution_var,
        )
        resolution_combo.pack(pady=2, padx=10)

        # Camera buttons
        button_frame = ctk.CTkFrame(camera_section)
        button_frame.pack(fill="x", padx=10, pady=5)

        self.start_camera_btn = ctk.CTkButton(
            button_frame, text="Start Camera", command=self.start_camera, width=120
        )
        self.start_camera_btn.pack(side="left", padx=2)

        self.stop_camera_btn = ctk.CTkButton(
            button_frame,
            text="Stop Camera",
            command=self.stop_camera,
            width=120,
            state="disabled",
        )
        self.stop_camera_btn.pack(side="right", padx=2)

    def create_detection_controls(self):
        """Create detection control section."""
        detection_section = ctk.CTkFrame(self.control_frame)
        detection_section.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            detection_section, text="Detection Controls", font=("Arial", 14, "bold")
        ).pack(pady=5)

        # Confidence threshold
        ctk.CTkLabel(detection_section, text="Confidence Threshold:").pack(
            anchor="w", padx=10
        )
        self.confidence_var = tk.DoubleVar(
            value=self.config["detection"]["confidence_threshold"]
        )
        self.confidence_slider = ctk.CTkSlider(
            detection_section,
            from_=0.1,
            to=1.0,
            variable=self.confidence_var,
            number_of_steps=90,
        )
        self.confidence_slider.pack(pady=2, padx=10, fill="x")

        self.confidence_label = ctk.CTkLabel(
            detection_section, text=f"Value: {self.confidence_var.get():.2f}"
        )
        self.confidence_label.pack(pady=2)
        self.confidence_var.trace("w", self.update_confidence_label)

        # Detection buttons
        button_frame = ctk.CTkFrame(detection_section)
        button_frame.pack(fill="x", padx=10, pady=5)

        self.start_detection_btn = ctk.CTkButton(
            button_frame,
            text="Start Detection",
            command=self.start_detection,
            width=120,
            state="disabled",
        )
        self.start_detection_btn.pack(side="left", padx=2)

        self.stop_detection_btn = ctk.CTkButton(
            button_frame,
            text="Stop Detection",
            command=self.stop_detection,
            width=120,
            state="disabled",
        )
        self.stop_detection_btn.pack(side="right", padx=2)

        # Capture button
        self.capture_btn = ctk.CTkButton(
            detection_section,
            text="Capture Frame",
            command=self.capture_frame,
            state="disabled",
        )
        self.capture_btn.pack(pady=5)

    def create_model_controls(self):
        """Create model control section."""
        model_section = ctk.CTkFrame(self.control_frame)
        model_section.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            model_section, text="Model Controls", font=("Arial", 14, "bold")
        ).pack(pady=5)

        # Model status
        self.model_status_label = ctk.CTkLabel(
            model_section, text="Model: Not Loaded", text_color="red"
        )
        self.model_status_label.pack(pady=2)

        # Model buttons
        self.load_model_btn = ctk.CTkButton(
            model_section, text="Load Model", command=self.load_model_dialog
        )
        self.load_model_btn.pack(pady=2)

        self.optimize_model_btn = ctk.CTkButton(
            model_section,
            text="Optimize Model",
            command=self.optimize_model,
            state="disabled",
        )
        self.optimize_model_btn.pack(pady=2)

    def create_statistics_display(self):
        """Create statistics display section."""
        stats_section = ctk.CTkFrame(self.control_frame)
        stats_section.pack(fill="both", expand=True, padx=10, pady=5)

        ctk.CTkLabel(stats_section, text="Statistics", font=("Arial", 14, "bold")).pack(
            pady=5
        )

        # Statistics text area
        self.stats_text = ctk.CTkTextbox(stats_section, height=200)
        self.stats_text.pack(fill="both", expand=True, padx=10, pady=5)

        # Update statistics periodically
        self.update_statistics()

    def create_menu_bar(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Model...", command=self.load_model_dialog)
        file_menu.add_command(label="Save Settings", command=self.save_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Fullscreen", command=self.toggle_fullscreen)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def create_status_bar(self):
        """Create status bar."""
        self.status_frame = ctk.CTkFrame(self.root, height=30)
        self.status_frame.pack(side="bottom", fill="x", padx=10, pady=(0, 10))

        self.status_label = ctk.CTkLabel(self.status_frame, text="Ready", anchor="w")
        self.status_label.pack(side="left", padx=10, pady=5)

        self.fps_label = ctk.CTkLabel(self.status_frame, text="FPS: 0.0")
        self.fps_label.pack(side="right", padx=10, pady=5)

    def initialize_model(self):
        """Initialize the detection model."""
        try:
            self.update_status("Initializing model...")

            # Check for trained model
            model_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "models", "trained"
            )
            model_files = (
                [f for f in os.listdir(model_path) if f.endswith(".pt")]
                if os.path.exists(model_path)
                else []
            )

            if model_files:
                # Use trained model
                model_file = os.path.join(model_path, model_files[0])
                self.detector = HelmetDetector(model_path=model_file)
                logger.info(f"Loaded trained model: {model_file}")
            else:
                # Use pretrained YOLOv8 model
                self.detector = HelmetDetector()
                logger.info("Loaded pretrained YOLOv8 model")

            # Initialize inference engine
            self.inference_engine = InferenceEngine(self.detector)

            # Update GUI
            self.model_status_label.configure(text="Model: Loaded", text_color="green")
            self.optimize_model_btn.configure(state="normal")

            self.update_status("Model initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            self.update_status(f"Model initialization failed: {e}")
            messagebox.showerror("Error", f"Failed to initialize model: {e}")

    def start_camera(self):
        """Start camera capture."""
        try:
            camera_id = int(self.camera_var.get())
            resolution = self.resolution_var.get().split("x")
            width, height = int(resolution[0]), int(resolution[1])

            self.camera_handler = CameraHandler(
                camera_id=camera_id,
                resolution=(width, height),
                fps=self.config["camera"]["fps"],
            )

            if not self.camera_handler.initialize_camera():
                raise RuntimeError("Failed to initialize camera")

            self.camera_handler.start_capture(frame_callback=self.on_frame_received)

            # Update GUI
            self.start_camera_btn.configure(state="disabled")
            self.stop_camera_btn.configure(state="normal")
            self.start_detection_btn.configure(state="normal")
            self.capture_btn.configure(state="normal")

            self.update_status("Camera started successfully")
            logger.info("Camera started")

        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            self.update_status(f"Camera start failed: {e}")
            messagebox.showerror("Error", f"Failed to start camera: {e}")

    def stop_camera(self):
        """Stop camera capture."""
        try:
            if self.camera_handler:
                self.camera_handler.stop_capture()
                self.camera_handler = None

            # Stop detection if running
            if self.is_detecting:
                self.stop_detection()

            # Update GUI
            self.start_camera_btn.configure(state="normal")
            self.stop_camera_btn.configure(state="disabled")
            self.start_detection_btn.configure(state="disabled")
            self.stop_detection_btn.configure(state="disabled")
            self.capture_btn.configure(state="disabled")

            # Clear video display
            self.video_canvas.delete("all")

            self.update_status("Camera stopped")
            logger.info("Camera stopped")

        except Exception as e:
            logger.error(f"Error stopping camera: {e}")
            self.update_status(f"Error stopping camera: {e}")

    def start_detection(self):
        """Start real-time detection."""
        if not self.detector or not self.camera_handler:
            messagebox.showerror("Error", "Model and camera must be initialized first")
            return

        self.is_detecting = True
        self.start_detection_btn.configure(state="disabled")
        self.stop_detection_btn.configure(state="normal")

        self.update_status("Detection started")
        logger.info("Detection started")

    def stop_detection(self):
        """Stop real-time detection."""
        self.is_detecting = False
        self.start_detection_btn.configure(state="normal")
        self.stop_detection_btn.configure(state="disabled")

        self.update_status("Detection stopped")
        logger.info("Detection stopped")

    def on_frame_received(self, frame: np.ndarray):
        """Handle received camera frame."""
        try:
            self.current_frame = frame.copy()

            # Perform detection if enabled
            if self.is_detecting and self.inference_engine:
                # Run inference
                results = self.inference_engine.predict(
                    frame, conf_threshold=self.confidence_var.get()
                )

                # Convert results to detection format
                self.detection_results = self.convert_results_to_detections(results)

                # Log performance
                self.performance_logger.log_frame()

                # Draw detections
                display_frame = self.detection_display.draw_detections(
                    frame, self.detection_results, fps=self.performance_logger.get_fps()
                )
            else:
                display_frame = frame
                self.detection_results = []

            # Update video display
            self.update_video_display(display_frame)

        except Exception as e:
            logger.error(f"Error processing frame: {e}")

    def convert_results_to_detections(self, results) -> List[Dict]:
        """Convert YOLO results to detection format."""
        detections = []

        if hasattr(results, "boxes") and results.boxes is not None:
            boxes = results.boxes

            for i in range(len(boxes)):
                # Extract box coordinates
                box = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())

                detection = {
                    "bbox": box.tolist(),
                    "confidence": confidence,
                    "class": class_id,
                }
                detections.append(detection)

        return detections

    def update_video_display(self, frame: np.ndarray):
        """Update video display with new frame."""
        try:
            # Resize frame to fit canvas
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                # Calculate scaling to maintain aspect ratio
                frame_height, frame_width = frame.shape[:2]
                scale_x = canvas_width / frame_width
                scale_y = canvas_height / frame_height
                scale = min(scale_x, scale_y)

                new_width = int(frame_width * scale)
                new_height = int(frame_height * scale)

                # Resize frame
                resized_frame = cv2.resize(frame, (new_width, new_height))

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

                # Convert to PIL Image and then to PhotoImage
                pil_image = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(pil_image)

                # Update canvas
                self.video_canvas.delete("all")
                x = (canvas_width - new_width) // 2
                y = (canvas_height - new_height) // 2
                self.video_canvas.create_image(x, y, anchor="nw", image=photo)

                # Keep a reference to prevent garbage collection
                self.video_canvas.image = photo

        except Exception as e:
            logger.error(f"Error updating video display: {e}")

    def capture_frame(self):
        """Capture current frame and save it."""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No frame available to capture")
            return

        try:
            # Ask user for save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[
                    ("JPEG files", "*.jpg"),
                    ("PNG files", "*.png"),
                    ("All files", "*.*"),
                ],
            )

            if filename:
                # Save frame with detections if available
                if self.detection_results:
                    self.detection_display.save_detection_image(
                        self.current_frame, self.detection_results, filename
                    )
                else:
                    cv2.imwrite(filename, self.current_frame)

                self.update_status(f"Frame saved: {filename}")
                messagebox.showinfo("Success", f"Frame saved successfully:\n{filename}")

        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            messagebox.showerror("Error", f"Failed to capture frame: {e}")

    def load_model_dialog(self):
        """Open dialog to load custom model."""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[
                ("PyTorch files", "*.pt"),
                ("ONNX files", "*.onnx"),
                ("All files", "*.*"),
            ],
        )

        if filename:
            self.load_custom_model(filename)

    def load_custom_model(self, model_path: str):
        """Load custom model from file."""
        try:
            self.update_status("Loading custom model...")

            # Create new detector with custom model
            self.detector = HelmetDetector(model_path=model_path)
            self.inference_engine = InferenceEngine(self.detector)

            # Update GUI
            self.model_status_label.configure(
                text=f"Model: {os.path.basename(model_path)}", text_color="green"
            )
            self.optimize_model_btn.configure(state="normal")

            self.update_status("Custom model loaded successfully")
            logger.info(f"Custom model loaded: {model_path}")

        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
            self.update_status(f"Failed to load model: {e}")
            messagebox.showerror("Error", f"Failed to load model: {e}")

    def optimize_model(self):
        """Optimize model for better performance."""
        if not self.detector:
            messagebox.showerror("Error", "No model loaded")
            return

        try:
            self.update_status("Optimizing model...")

            # Export to TensorRT if available
            optimized_path = self.detector.export_tensorrt()

            if optimized_path:
                self.update_status("Model optimized successfully")
                messagebox.showinfo("Success", "Model optimized for TensorRT")
            else:
                self.update_status("Model optimization failed")
                messagebox.showwarning("Warning", "Model optimization failed")

        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            self.update_status(f"Optimization failed: {e}")
            messagebox.showerror("Error", f"Model optimization failed: {e}")

    def update_confidence_label(self, *args):
        """Update confidence threshold label."""
        value = self.confidence_var.get()
        self.confidence_label.configure(text=f"Value: {value:.2f}")

    def update_statistics(self):
        """Update statistics display."""
        try:
            stats_text = "=== Performance Statistics ===\n\n"

            # Camera statistics
            if self.camera_handler:
                camera_info = self.camera_handler.get_camera_info()
                stats_text += f"Camera FPS: {camera_info['actual_fps']:.1f}\n"
                stats_text += f"Frames Captured: {camera_info['frame_count']}\n"
                stats_text += f"Resolution: {camera_info.get('actual_width', 'N/A')}x{camera_info.get('actual_height', 'N/A')}\n\n"

            # Detection statistics
            if self.detection_results:
                summary = self.detection_display.create_detection_summary(
                    self.detection_results
                )
                stats_text += f"Detections: {summary['total_detections']}\n"
                stats_text += f"Safety Score: {summary['safety_score']:.2f}\n"

                for class_name, count in summary["class_counts"].items():
                    stats_text += f"{class_name}: {count}\n"

                stats_text += f"\nConfidence Stats:\n"
                stats_text += f"  Mean: {summary['confidence_stats']['mean']:.3f}\n"
                stats_text += f"  Min: {summary['confidence_stats']['min']:.3f}\n"
                stats_text += f"  Max: {summary['confidence_stats']['max']:.3f}\n"

            # Performance statistics
            if hasattr(self, "performance_logger"):
                perf_stats = self.performance_logger.get_stats()
                stats_text += f"\n=== Performance ===\n"
                stats_text += f"Processing FPS: {perf_stats.get('fps', 0):.1f}\n"
                stats_text += f"Avg Inference Time: {perf_stats.get('avg_inference_time', 0):.1f}ms\n"

            # Update display
            self.stats_text.delete("1.0", "end")
            self.stats_text.insert("1.0", stats_text)

            # Update FPS label
            if self.camera_handler:
                fps = self.camera_handler.get_camera_info()["actual_fps"]
                self.fps_label.configure(text=f"FPS: {fps:.1f}")

        except Exception as e:
            logger.error(f"Error updating statistics: {e}")

        # Schedule next update
        self.root.after(1000, self.update_statistics)

    def update_status(self, message: str):
        """Update status bar message."""
        self.status_label.configure(text=message)
        logger.info(f"Status: {message}")

    def save_settings(self):
        """Save current settings."""
        try:
            settings = {
                "camera_id": self.camera_var.get(),
                "resolution": self.resolution_var.get(),
                "confidence_threshold": self.confidence_var.get(),
            }

            settings_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "config", "user_settings.yaml"
            )
            with open(settings_path, "w") as f:
                yaml.dump(settings, f)

            self.update_status("Settings saved")
            messagebox.showinfo("Success", "Settings saved successfully")

        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    def toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        current_state = self.root.attributes("-fullscreen")
        self.root.attributes("-fullscreen", not current_state)

    def show_about(self):
        """Show about dialog."""
        about_text = f"""
{self.config['app']['name']} v{self.config['app']['version']}

Construction Safety Detection System
Real-time helmet detection using YOLOv8

Features:
• Real-time camera detection
• GPU acceleration
• TensorRT optimization
• Performance monitoring

Developed for HSE Vision Project
        """
        messagebox.showinfo("About", about_text)

    def on_closing(self):
        """Handle application closing."""
        try:
            logger.info("Application closing...")

            # Stop camera and detection
            if self.camera_handler:
                self.stop_camera()

            # Cleanup
            if hasattr(self, "performance_logger"):
                self.performance_logger.save_logs()

            self.root.destroy()

        except Exception as e:
            logger.error(f"Error during application closing: {e}")
            self.root.destroy()

    def run(self):
        """Run the application."""
        logger.info("Starting application main loop")
        self.root.mainloop()
