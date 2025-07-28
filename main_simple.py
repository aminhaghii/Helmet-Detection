"""
File: main_simple.py
Purpose: Simplified main application without OpenCV dependencies
Author: HSE Vision Team
Date: 2024
Project: Construction Safety Detection System
"""

import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


class SimpleDetectionApp:
    """Simplified detection application for testing."""

    def __init__(self):
        """Initialize the application."""
        self.root = tk.Tk()
        self.root.title("HSE Vision - Construction Safety Detection")
        self.root.geometry("800x600")

        # Variables
        self.is_running = False
        self.detection_count = 0
        self.helmet_count = 0
        self.no_helmet_count = 0

        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(
            main_frame,
            text="HSE Vision - Safety Detection System",
            font=("Arial", 16, "bold"),
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding="10")
        status_frame.grid(
            row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)
        )

        self.status_label = ttk.Label(
            status_frame, text="System Ready", foreground="green", font=("Arial", 12)
        )
        self.status_label.grid(row=0, column=0)

        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(
            row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)
        )

        self.start_button = ttk.Button(
            control_frame, text="Start Detection", command=self.start_detection
        )
        self.start_button.grid(row=0, column=0, padx=(0, 10))

        self.stop_button = ttk.Button(
            control_frame,
            text="Stop Detection",
            command=self.stop_detection,
            state="disabled",
        )
        self.stop_button.grid(row=0, column=1, padx=(0, 10))

        self.test_button = ttk.Button(
            control_frame, text="Test System", command=self.test_system
        )
        self.test_button.grid(row=0, column=2)

        # Statistics frame
        stats_frame = ttk.LabelFrame(
            main_frame, text="Detection Statistics", padding="10"
        )
        stats_frame.grid(
            row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)
        )

        ttk.Label(stats_frame, text="Total Detections:").grid(
            row=0, column=0, sticky=tk.W
        )
        self.total_label = ttk.Label(stats_frame, text="0", font=("Arial", 12, "bold"))
        self.total_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        ttk.Label(stats_frame, text="With Helmet:").grid(row=1, column=0, sticky=tk.W)
        self.helmet_label = ttk.Label(
            stats_frame, text="0", font=("Arial", 12, "bold"), foreground="green"
        )
        self.helmet_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))

        ttk.Label(stats_frame, text="Without Helmet:").grid(
            row=2, column=0, sticky=tk.W
        )
        self.no_helmet_label = ttk.Label(
            stats_frame, text="0", font=("Arial", 12, "bold"), foreground="red"
        )
        self.no_helmet_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0))

        # Log frame
        log_frame = ttk.LabelFrame(main_frame, text="Activity Log", padding="10")
        log_frame.grid(
            row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10)
        )

        # Text widget with scrollbar
        self.log_text = tk.Text(log_frame, height=10, width=70)
        scrollbar = ttk.Scrollbar(
            log_frame, orient="vertical", command=self.log_text.yview
        )
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_message("Application initialized successfully")

    def log_message(self, message):
        """Add a message to the log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)

    def start_detection(self):
        """Start the detection process."""
        if not self.is_running:
            self.is_running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.status_label.config(text="Detection Running", foreground="blue")

            self.log_message("Starting detection process...")

            # Start detection thread
            self.detection_thread = threading.Thread(
                target=self.detection_loop, daemon=True
            )
            self.detection_thread.start()

    def stop_detection(self):
        """Stop the detection process."""
        if self.is_running:
            self.is_running = False
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.status_label.config(text="Detection Stopped", foreground="orange")

            self.log_message("Detection process stopped")

    def detection_loop(self):
        """Simulated detection loop."""
        while self.is_running:
            # Simulate detection
            time.sleep(2)  # Simulate processing time

            if self.is_running:  # Check again in case stopped during sleep
                # Simulate random detection
                import random

                has_helmet = random.choice([True, False])
                confidence = random.uniform(0.7, 0.95)

                self.detection_count += 1
                if has_helmet:
                    self.helmet_count += 1
                    detection_type = "Helmet Detected"
                    color = "green"
                else:
                    self.no_helmet_count += 1
                    detection_type = "No Helmet Detected"
                    color = "red"

                # Update UI in main thread
                self.root.after(
                    0, self.update_detection_ui, detection_type, confidence, color
                )

    def update_detection_ui(self, detection_type, confidence, color):
        """Update the UI with detection results."""
        # Update statistics
        self.total_label.config(text=str(self.detection_count))
        self.helmet_label.config(text=str(self.helmet_count))
        self.no_helmet_label.config(text=str(self.no_helmet_count))

        # Log detection
        message = f"{detection_type} (Confidence: {confidence:.2f})"
        self.log_message(message)

    def test_system(self):
        """Test system components."""
        self.log_message("Running system tests...")

        # Test 1: Check Python version
        python_version = sys.version.split()[0]
        self.log_message(f"✅ Python version: {python_version}")

        # Test 2: Check PyTorch
        try:
            import torch

            self.log_message(f"✅ PyTorch version: {torch.__version__}")

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                self.log_message(f"✅ CUDA available: {gpu_name}")
            else:
                self.log_message("⚠️  CUDA not available - CPU mode only")

        except ImportError:
            self.log_message("❌ PyTorch not installed")

        # Test 3: Check project structure
        required_dirs = ["src/core", "src/desktop_app", "config", "models"]
        missing_dirs = []

        for dir_path in required_dirs:
            if Path(dir_path).exists():
                self.log_message(f"✅ Directory exists: {dir_path}")
            else:
                missing_dirs.append(dir_path)
                self.log_message(f"❌ Missing directory: {dir_path}")

        # Test 4: Check config files
        config_files = [
            "config/data.yaml",
            "config/model_config.yaml",
            "config/deployment_config.yaml",
        ]
        for config_file in config_files:
            if Path(config_file).exists():
                self.log_message(f"✅ Config file exists: {config_file}")
            else:
                self.log_message(f"❌ Missing config file: {config_file}")

        if not missing_dirs:
            self.log_message("✅ All system tests passed!")
        else:
            self.log_message("⚠️  Some tests failed - check missing components")

    def run(self):
        """Run the application."""
        try:
            self.log_message("Starting HSE Vision application...")
            self.root.mainloop()
        except KeyboardInterrupt:
            self.log_message("Application interrupted by user")
        except Exception as e:
            self.log_message(f"Application error: {e}")
            messagebox.showerror("Error", f"Application error: {e}")


def main():
    """Main function."""
    print("🚀 HSE Vision - Construction Safety Detection System")
    print("=" * 60)

    try:
        app = SimpleDetectionApp()
        app.run()
    except Exception as e:
        print(f"Failed to start application: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
