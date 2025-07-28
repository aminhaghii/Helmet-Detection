"""
🎯 PROFESSIONAL HELMET DETECTION SYSTEM
Advanced Real-Time Safety Monitoring with Enhanced UI
Version 2.0 - Professional Edition
"""

import cv2
from ultralytics import YOLO
import numpy as np
import time
import os
import math
from collections import defaultdict, deque
from datetime import datetime

class ProfessionalHelmetDetector:
    def __init__(self, model_path):
        """Initialize the professional helmet detector with enhanced features"""
        self.model = YOLO(model_path)
        self.class_names = ['helm', 'no-helm']
        
        # Professional color scheme
        self.colors = {
            'helm': (34, 139, 34),      # Forest Green for helmet (safe)
            'no-helm': (220, 20, 60),   # Crimson Red for no helmet (danger)
            'background': (45, 45, 45), # Dark gray for professional look
            'accent': (255, 215, 0),    # Gold for highlights
            'text': (255, 255, 255),    # White for text
            'panel': (30, 30, 30),      # Darker panel background
            'success': (0, 255, 127),   # Spring green for success
            'warning': (255, 165, 0),   # Orange for warnings
            'danger': (255, 69, 0)      # Red orange for danger
        }
        
        # Enhanced statistics tracking
        self.detection_history = deque(maxlen=200)  # Increased history
        self.class_counts = defaultdict(int)
        self.fps_history = deque(maxlen=60)  # 2 seconds at 30fps
        self.confidence_history = deque(maxlen=100)
        self.inference_times = deque(maxlen=50)
        
        # Performance metrics
        self.total_frames = 0
        self.total_detections = 0
        self.start_time = time.time()
        self.session_start = datetime.now()
        
        # Alert system
        self.alert_threshold = 5  # Alert after 5 consecutive no-helmet detections
        self.consecutive_no_helmet = 0
        self.alert_active = False
        self.last_alert_time = 0
        
        # UI state
        self.show_confidence_graph = True
        self.show_fps_graph = True
        self.panel_alpha = 0.85
        
    def detect_helmets(self, frame):
        """Detect helmets with enhanced performance tracking and alert system"""
        start_time = time.time()
        results = self.model(frame, conf=0.4, iou=0.45, verbose=False)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        self.total_frames += 1
        self.inference_times.append(inference_time)
        
        # Check for consecutive no-helmet detections for alert system
        if results[0].boxes is not None:
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            has_no_helmet = any(self.class_names[cls] == 'no-helm' for cls in classes)
            
            if has_no_helmet:
                self.consecutive_no_helmet += 1
                if self.consecutive_no_helmet >= self.alert_threshold:
                    self.alert_active = True
                    self.last_alert_time = time.time()
            else:
                self.consecutive_no_helmet = 0
                if time.time() - self.last_alert_time > 3:  # Clear alert after 3 seconds
                    self.alert_active = False
        else:
            self.consecutive_no_helmet = 0
            if time.time() - self.last_alert_time > 3:
                self.alert_active = False
        
        return results[0], inference_time
    
    def update_statistics(self, results, inference_time):
        """Update detection statistics"""
        current_detections = []
        
        if results.boxes is not None:
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy()
            
            for cls, conf in zip(classes, confidences):
                class_name = self.class_names[cls]
                current_detections.append((class_name, conf))
                self.class_counts[class_name] += 1
                self.confidence_history.append(conf)
        
        self.detection_history.append(current_detections)
        self.total_detections += len(current_detections)
        
        # Calculate FPS
        fps = 1000 / inference_time if inference_time > 0 else 0
        self.fps_history.append(fps)
    
    def draw_professional_detections(self, frame, results):
        """Draw professional-grade detection visualization with modern UI elements"""
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Add subtle gradient overlay for professional look
        overlay = np.zeros_like(frame)
        overlay[:50, :] = self.colors['panel']  # Top bar
        overlay[-50:, :] = self.colors['panel']  # Bottom bar
        cv2.addWeighted(annotated_frame, 0.95, overlay, 0.05, 0, annotated_frame)
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = box.astype(int)
                class_name = self.class_names[cls]
                color = self.colors[class_name]
                
                # Dynamic thickness based on confidence
                thickness = max(2, int(conf * 6))
                
                # Draw main bounding box with rounded corners effect
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Add corner accents for modern look
                corner_size = 15
                corner_thickness = 3
                # Top-left corner
                cv2.line(annotated_frame, (x1, y1), (x1 + corner_size, y1), self.colors['accent'], corner_thickness)
                cv2.line(annotated_frame, (x1, y1), (x1, y1 + corner_size), self.colors['accent'], corner_thickness)
                # Top-right corner
                cv2.line(annotated_frame, (x2, y1), (x2 - corner_size, y1), self.colors['accent'], corner_thickness)
                cv2.line(annotated_frame, (x2, y1), (x2, y1 + corner_size), self.colors['accent'], corner_thickness)
                # Bottom-left corner
                cv2.line(annotated_frame, (x1, y2), (x1 + corner_size, y2), self.colors['accent'], corner_thickness)
                cv2.line(annotated_frame, (x1, y2), (x1, y2 - corner_size), self.colors['accent'], corner_thickness)
                # Bottom-right corner
                cv2.line(annotated_frame, (x2, y2), (x2 - corner_size, y2), self.colors['accent'], corner_thickness)
                cv2.line(annotated_frame, (x2, y2), (x2, y2 - corner_size), self.colors['accent'], corner_thickness)
                
                # Professional label design
                label = f"{class_name.upper()}"
                confidence_text = f"{conf:.1%}"
                
                # Calculate label dimensions
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
                conf_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                
                # Create professional label background with gradient effect
                label_height = max(label_size[1], conf_size[1]) + 20
                label_width = max(label_size[0], conf_size[0]) + 20
                
                # Main label background
                label_bg = np.zeros((label_height, label_width, 3), dtype=np.uint8)
                label_bg[:] = color
                
                # Add gradient effect
                for i in range(label_height):
                    alpha = 0.7 + 0.3 * (i / label_height)
                    label_bg[i] = label_bg[i] * alpha
                
                # Position label
                label_y1 = max(0, y1 - label_height - 5)
                label_y2 = label_y1 + label_height
                label_x1 = x1
                label_x2 = min(w, x1 + label_width)
                
                # Blend label background
                roi = annotated_frame[label_y1:label_y2, label_x1:label_x2]
                if roi.shape[:2] == label_bg.shape[:2]:
                    cv2.addWeighted(roi, 0.3, label_bg, 0.7, 0, roi)
                
                # Add text with shadow effect
                # Shadow
                cv2.putText(annotated_frame, label, 
                          (x1 + 11, y1 - label_height + 21), 
                          cv2.FONT_HERSHEY_DUPLEX, 0.8, 
                          (0, 0, 0), 3)
                # Main text
                cv2.putText(annotated_frame, label, 
                          (x1 + 10, y1 - label_height + 20), 
                          cv2.FONT_HERSHEY_DUPLEX, 0.8, 
                          self.colors['text'], 2)
                
                # Confidence text
                cv2.putText(annotated_frame, confidence_text, 
                          (x1 + 11, y1 - 6), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                          (0, 0, 0), 2)
                cv2.putText(annotated_frame, confidence_text, 
                          (x1 + 10, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                          self.colors['text'], 1)
                
                # Professional confidence bar with gradient
                bar_height = 8
                bar_width = int((x2 - x1) * conf)
                bar_y = y2 + 8
                
                # Background bar
                cv2.rectangle(annotated_frame, (x1, bar_y), (x2, bar_y + bar_height), 
                            (50, 50, 50), -1)
                
                # Confidence bar with color gradient
                if bar_width > 0:
                    # Create gradient effect
                    for i in range(bar_width):
                        gradient_color = [
                            int(color[0] * (0.6 + 0.4 * i / bar_width)),
                            int(color[1] * (0.6 + 0.4 * i / bar_width)),
                            int(color[2] * (0.6 + 0.4 * i / bar_width))
                        ]
                        cv2.line(annotated_frame, (x1 + i, bar_y), (x1 + i, bar_y + bar_height), 
                               gradient_color, 1)
        
        return annotated_frame
    
    def draw_professional_statistics_panel(self, frame):
        """Draw a comprehensive professional statistics panel with modern design"""
        h, w = frame.shape[:2]
        
        # Main panel dimensions
        panel_width = 420
        panel_height = 280
        panel_x = 15
        panel_y = 15
        
        # Create main panel with rounded corners effect
        overlay = frame.copy()
        
        # Main panel background with gradient
        panel_bg = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        for i in range(panel_height):
            alpha = 0.9 - 0.1 * (i / panel_height)
            panel_bg[i] = [int(c * alpha) for c in self.colors['panel']]
        
        # Blend panel
        roi = overlay[panel_y:panel_y + panel_height, panel_x:panel_x + panel_width]
        if roi.shape[:2] == panel_bg.shape[:2]:
            cv2.addWeighted(roi, 0.2, panel_bg, 0.8, 0, roi)
        
        cv2.addWeighted(overlay, self.panel_alpha, frame, 1 - self.panel_alpha, 0, frame)
        
        # Add panel border with accent color
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     self.colors['accent'], 2)
        
        # Header section
        header_height = 40
        cv2.rectangle(frame, (panel_x + 2, panel_y + 2), 
                     (panel_x + panel_width - 2, panel_y + header_height), 
                     self.colors['accent'], -1)
        
        # Title with shadow
        title = "🎯 SAFETY MONITORING SYSTEM"
        cv2.putText(frame, title, (panel_x + 16, panel_y + 28), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, title, (panel_x + 15, panel_y + 27), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)
        
        # Calculate statistics
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        avg_conf = np.mean(self.confidence_history) if self.confidence_history else 0
        avg_inference = np.mean(self.inference_times) if self.inference_times else 0
        runtime = time.time() - self.start_time
        detection_rate = (self.total_detections / self.total_frames * 100) if self.total_frames > 0 else 0
        
        # Status indicator
        status_color = self.colors['success'] if avg_fps > 20 else self.colors['warning'] if avg_fps > 10 else self.colors['danger']
        status_text = "OPTIMAL" if avg_fps > 20 else "GOOD" if avg_fps > 10 else "LOW"
        
        # Performance metrics section
        y_start = panel_y + header_height + 15
        metrics = [
            ("STATUS", status_text, status_color),
            ("RUNTIME", f"{runtime:.1f}s", self.colors['text']),
            ("FPS", f"{avg_fps:.1f}", self.colors['success'] if avg_fps > 20 else self.colors['warning']),
            ("INFERENCE", f"{avg_inference:.1f}ms", self.colors['text']),
            ("FRAMES", f"{self.total_frames:,}", self.colors['text']),
            ("DETECTIONS", f"{self.total_detections:,}", self.colors['text']),
            ("DETECTION RATE", f"{detection_rate:.1f}%", self.colors['text']),
            ("AVG CONFIDENCE", f"{avg_conf:.1%}", self.colors['text'])
        ]
        
        # Draw metrics in two columns
        col1_x = panel_x + 15
        col2_x = panel_x + 220
        
        for i, (label, value, color) in enumerate(metrics):
            x_pos = col1_x if i < 4 else col2_x
            y_pos = y_start + (i % 4) * 22
            
            # Label
            cv2.putText(frame, f"{label}:", (x_pos, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors['text'], 1)
            
            # Value with color coding
            cv2.putText(frame, value, (x_pos + 85, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        
        # Class counts section
        class_y = y_start + 100
        cv2.putText(frame, "DETECTION COUNTS:", (col1_x, class_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['accent'], 1)
        
        # Helmet count with icon
        helmet_count = self.class_counts['helm']
        no_helmet_count = self.class_counts['no-helm']
        
        cv2.putText(frame, f"✓ HELMET: {helmet_count}", (col1_x, class_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['helm'], 1)
        cv2.putText(frame, f"⚠ NO HELMET: {no_helmet_count}", (col1_x, class_y + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['no-helm'], 1)
        
        # Mini FPS graph
        if self.show_fps_graph and len(self.fps_history) > 1:
            self.draw_mini_graph(frame, col2_x, class_y, 150, 50, 
                               list(self.fps_history), "FPS", self.colors['success'])
        
        # Alert indicator
        if self.alert_active:
            alert_y = panel_y + panel_height - 35
            # Blinking effect
            if int(time.time() * 3) % 2:
                cv2.rectangle(frame, (panel_x + 10, alert_y), 
                             (panel_x + panel_width - 10, alert_y + 25), 
                             self.colors['danger'], -1)
                cv2.putText(frame, "⚠ SAFETY ALERT: NO HELMET DETECTED!", 
                           (panel_x + 20, alert_y + 18), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        return frame
    
    def draw_mini_graph(self, frame, x, y, width, height, data, title, color):
        """Draw a mini performance graph"""
        if len(data) < 2:
            return
        
        # Graph background
        cv2.rectangle(frame, (x, y), (x + width, y + height), (20, 20, 20), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 1)
        
        # Title
        cv2.putText(frame, title, (x + 5, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Normalize data
        max_val = max(data) if max(data) > 0 else 1
        min_val = min(data)
        
        # Draw graph lines
        points = []
        for i, val in enumerate(data):
            px = x + int((i / (len(data) - 1)) * (width - 10)) + 5
            py = y + height - 5 - int(((val - min_val) / (max_val - min_val)) * (height - 10))
            points.append((px, py))
        
        # Draw lines between points
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, 1)
        
        # Draw points
        for point in points[-10:]:  # Only last 10 points
            cv2.circle(frame, point, 2, color, -1)
    
    def draw_professional_controls_info(self, frame):
        """Draw professional control information with modern styling"""
        h, w = frame.shape[:2]
        
        # Control panel dimensions
        control_width = 250
        control_height = 140
        control_x = w - control_width - 15
        control_y = h - control_height - 15
        
        # Create control panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (control_x, control_y), 
                     (control_x + control_width, control_y + control_height), 
                     self.colors['panel'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Panel border
        cv2.rectangle(frame, (control_x, control_y), 
                     (control_x + control_width, control_y + control_height), 
                     self.colors['accent'], 2)
        
        # Header
        cv2.rectangle(frame, (control_x + 2, control_y + 2), 
                     (control_x + control_width - 2, control_y + 30), 
                     self.colors['accent'], -1)
        
        cv2.putText(frame, "⌨ CONTROLS", (control_x + 15, control_y + 22), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
        
        # Control items with icons
        controls = [
            ("Q", "Quit Application", self.colors['danger']),
            ("S", "Save Screenshot", self.colors['success']),
            ("R", "Reset Statistics", self.colors['warning']),
            ("C", "Toggle Stats Panel", self.colors['text']),
            ("G", "Toggle FPS Graph", self.colors['text'])
        ]
        
        y_start = control_y + 45
        for i, (key, desc, color) in enumerate(controls):
            y_pos = y_start + i * 18
            
            # Key background
            cv2.rectangle(frame, (control_x + 10, y_pos - 12), 
                         (control_x + 30, y_pos + 2), color, -1)
            cv2.putText(frame, key, (control_x + 16, y_pos - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Description
            cv2.putText(frame, desc, (control_x + 40, y_pos - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        return frame
    
    def draw_header_info(self, frame):
        """Draw professional header with system info"""
        h, w = frame.shape[:2]
        
        # Header background
        header_height = 45
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_height), self.colors['panel'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # System info
        current_time = datetime.now().strftime("%H:%M:%S")
        session_duration = datetime.now() - self.session_start
        duration_str = str(session_duration).split('.')[0]  # Remove microseconds
        
        # Left side - System name
        cv2.putText(frame, "🛡 HSE VISION - HELMET DETECTION SYSTEM", 
                   (15, 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['accent'], 2)
        
        # Right side - Time and session info
        info_text = f"TIME: {current_time} | SESSION: {duration_str}"
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(frame, info_text, (w - text_size[0] - 15, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # Separator line
        cv2.line(frame, (0, header_height), (w, header_height), self.colors['accent'], 2)
        
        return frame

def main():
    # Model path
    model_path = "models/trained/helmet_detection_20250727_152811/weights/best.pt"
    
    print("🚀 PROFESSIONAL HELMET DETECTION SYSTEM")
    print("=" * 60)
    print("🎯 Advanced Real-Time Safety Monitoring")
    print("📊 Enhanced UI with Professional Analytics")
    print("⚡ YOLOv8s Model - Optimized Performance")
    print("=" * 60)
    
    # Check model
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"✅ Model loaded: {model_path}")
    
    # Initialize detector
    try:
        detector = ProfessionalHelmetDetector(model_path)
        print("✅ Professional detector initialized")
        print(f"📋 Classes: {', '.join(detector.class_names)}")
        print(f"🎨 Professional color scheme loaded")
        print(f"📊 Enhanced statistics tracking enabled")
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access camera")
        return
    
    # Professional camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"✅ Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
    print(f"🔧 Buffer size optimized for real-time processing")
    print("\n🎯 Starting professional detection system...")
    print("📱 Use keyboard controls for interaction")
    print("🖥️  Press 'Q' to quit when ready")
    
    screenshot_count = 0
    show_stats = True
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to read frame from camera")
                break
            
            # Mirror effect for natural interaction
            frame = cv2.flip(frame, 1)
            
            # Add professional header
            frame = detector.draw_header_info(frame)
            
            # Detect helmets with enhanced tracking
            results, inference_time = detector.detect_helmets(frame)
            
            # Update comprehensive statistics
            detector.update_statistics(results, inference_time)
            
            # Draw professional detection visualization
            annotated_frame = detector.draw_professional_detections(frame, results)
            
            # Add professional statistics panel
            if show_stats:
                annotated_frame = detector.draw_professional_statistics_panel(annotated_frame)
            
            # Add professional controls info
            annotated_frame = detector.draw_professional_controls_info(annotated_frame)
            
            # Display with professional window title
            cv2.imshow('🛡 HSE VISION - Professional Helmet Detection System v2.0', annotated_frame)
            
            # Enhanced keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("🔄 Shutting down system...")
                break
            elif key == ord('s') or key == ord('S'):
                screenshot_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"professional_helmet_detection_{timestamp}_{screenshot_count:03d}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"📸 Professional screenshot saved: {filename}")
            elif key == ord('r') or key == ord('R'):
                # Reset all statistics
                detector.class_counts.clear()
                detector.detection_history.clear()
                detector.confidence_history.clear()
                detector.fps_history.clear()
                detector.inference_times.clear()
                detector.total_detections = 0
                detector.total_frames = 0
                detector.start_time = time.time()
                detector.session_start = datetime.now()
                detector.alert_active = False
                detector.consecutive_no_helmet = 0
                print("🔄 All statistics reset - Fresh start!")
            elif key == ord('c') or key == ord('C'):
                show_stats = not show_stats
                status = "ENABLED" if show_stats else "DISABLED"
                print(f"📊 Statistics panel: {status}")
            elif key == ord('g') or key == ord('G'):
                detector.show_fps_graph = not detector.show_fps_graph
                status = "ENABLED" if detector.show_fps_graph else "DISABLED"
                print(f"📈 FPS graph: {status}")
    
    except KeyboardInterrupt:
        print("\n⚠️ System interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Professional final report
        runtime = time.time() - detector.start_time
        avg_fps = np.mean(detector.fps_history) if detector.fps_history else 0
        avg_inference = np.mean(detector.inference_times) if detector.inference_times else 0
        avg_confidence = np.mean(detector.confidence_history) if detector.confidence_history else 0
        
        print(f"\n📊 PROFESSIONAL SYSTEM REPORT")
        print(f"=" * 50)
        print(f"⏱️  Total Runtime: {runtime:.2f} seconds")
        print(f"🖼️  Frames Processed: {detector.total_frames:,}")
        print(f"🎯 Total Detections: {detector.total_detections:,}")
        print(f"⚡ Average FPS: {avg_fps:.1f}")
        print(f"🔍 Average Inference: {avg_inference:.1f}ms")
        print(f"📈 Average Confidence: {avg_confidence:.1%}")
        print(f"✅ Helmet Detections: {detector.class_counts['helm']:,}")
        print(f"⚠️  No-Helmet Detections: {detector.class_counts['no-helm']:,}")
        print(f"📸 Screenshots Saved: {screenshot_count}")
        
        # Performance rating
        if avg_fps > 25:
            rating = "EXCELLENT ⭐⭐⭐⭐⭐"
        elif avg_fps > 20:
            rating = "VERY GOOD ⭐⭐⭐⭐"
        elif avg_fps > 15:
            rating = "GOOD ⭐⭐⭐"
        elif avg_fps > 10:
            rating = "FAIR ⭐⭐"
        else:
            rating = "NEEDS OPTIMIZATION ⭐"
        
        print(f"🏆 Performance Rating: {rating}")
        print(f"=" * 50)
        print("✅ Professional system shutdown completed successfully!")

if __name__ == "__main__":
    main()