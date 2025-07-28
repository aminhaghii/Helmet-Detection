# 📷 Camera Testing Guide

## Available Test Scripts

### 1. Simple Test (`simple_test.py`)
- **Purpose**: Quick basic test of helmet detection
- **Features**: Basic detection with minimal interface
- **Usage**: `python simple_test.py`
- **Best for**: Quick verification that model works

### 2. Advanced Test (`advanced_test.py`)
- **Purpose**: Comprehensive testing with statistics
- **Features**: 
  - Real-time statistics panel
  - Detection confidence visualization
  - Performance metrics (FPS, detection rate)
  - Class counting (helmet vs no-helmet)
  - Enhanced bounding boxes
  - Screenshot capability
- **Usage**: `python advanced_test.py`
- **Best for**: Detailed analysis and performance evaluation

### 3. Full Camera Test (`test_camera.py`)
- **Purpose**: Complete testing suite
- **Features**: All advanced features plus additional controls
- **Usage**: `python test_camera.py` or `test_camera.bat`
- **Best for**: Production-ready testing

## Controls

| Key | Action |
|-----|--------|
| `Q` | Quit the application |
| `S` | Save screenshot |
| `R` | Reset statistics |
| `C` | Toggle statistics panel (advanced test) |

## Model Performance

Based on training results:
- **Model**: YOLOv8s (Small)
- **Size**: ~22MB
- **Classes**: 2 (helm, no-helm)
- **Inference Speed**: ~6ms per frame
- **Expected FPS**: 20-40 on RTX 4050
- **Accuracy**: 99.4% mAP@0.5

## Expected Detection Results

### ✅ Good Detection Scenarios:
- Clear view of person's head
- Good lighting conditions
- Person facing camera
- Helmet clearly visible or absent

### ⚠️ Challenging Scenarios:
- Poor lighting
- Side/back view of person
- Partially occluded helmet
- Multiple people in frame
- Very small faces in distance

## Troubleshooting

### Camera Issues:
```bash
# If camera doesn't open:
# 1. Check if camera is being used by another app
# 2. Try different camera index:
cap = cv2.VideoCapture(1)  # Instead of 0
```

### OpenCV Issues:
```bash
# If GUI doesn't work:
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-python==4.8.1.78
```

### Performance Issues:
- Lower confidence threshold for more detections
- Increase confidence threshold for fewer false positives
- Adjust camera resolution for better performance

## Model Confidence Levels

- **High Confidence (>80%)**: Very reliable detection
- **Medium Confidence (50-80%)**: Good detection, some uncertainty
- **Low Confidence (<50%)**: Uncertain detection, may be false positive

## Statistics Explanation

- **FPS**: Frames processed per second
- **Detection Rate**: Percentage of frames with detections
- **Avg Confidence**: Average confidence of all detections
- **Class Counts**: Total detections per class since start

## Next Steps

After successful camera testing:
1. Test with different lighting conditions
2. Test with multiple people
3. Test with different helmet types
4. Consider fine-tuning if needed
5. Deploy to production environment

## Files Created During Testing

- `helmet_detection_screenshot_XXX.jpg`: Screenshots taken during testing
- Performance logs in terminal output