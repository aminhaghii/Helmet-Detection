"""
Simple Helmet Detection Test
Quick test script for helmet detection on camera
"""

import cv2
from ultralytics import YOLO
import os

def test_helmet_detection():
    # Model path
    model_path = "models/trained/helmet_detection_20250727_152811/weights/best.pt"
    
    print("🔍 Helmet Detection Test")
    print("=" * 40)
    
    # Check model
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    print(f"✅ Model found: {model_path}")
    
    # Load model
    try:
        model = YOLO(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Test camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return False
    
    print("✅ Camera opened successfully")
    print("\n🎯 Starting detection...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Run detection
        results = model(frame, conf=0.5)
        
        # Draw results
        annotated_frame = results[0].plot()
        
        # Show frame
        cv2.imshow('Helmet Detection Test', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Test completed!")
    return True

if __name__ == "__main__":
    test_helmet_detection()