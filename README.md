# 🔒 HSE Vision - Professional Helmet Detection System

A state-of-the-art computer vision system for real-time helmet detection in construction and industrial environments, built with YOLOv8 and featuring a professional-grade user interface.

## 🌟 Features

### Professional Detection System
- **Real-time helmet detection** using YOLOv8 models
- **Professional UI** with modern dark theme and gradient overlays
- **Advanced statistics panel** with comprehensive metrics
- **Dynamic bounding boxes** with confidence-based thickness
- **Alert system** for safety violations
- **Performance monitoring** with real-time graphs

### Enhanced Analytics
- Real-time FPS monitoring with mini graphs
- Detection rate tracking
- Confidence score analysis
- Session duration tracking
- Inference time measurement
- Automatic performance rating

### Professional Interface
- Modern gradient-based UI design
- Professional control panel
- System header with real-time information
- Enhanced visual feedback
- Professional screenshot capabilities

## 📁 Project Structure

```
HSE_Vision/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── config/                 # Configuration files
├── data/                   # Dataset storage
│   ├── raw/               # Raw datasets
│   └── processed/         # Processed datasets
├── docs/                   # Documentation
├── models/                 # Model storage
│   ├── pretrained/        # Pre-trained YOLO models
│   ├── trained/           # Custom trained models
│   └── optimized/         # Optimized models
├── scripts/               # Standalone scripts
│   ├── advanced_test.py   # Professional detection system
│   ├── simple_test.py     # Basic detection test
│   ├── test_camera.py     # Camera testing utility
│   └── ...               # Other utility scripts
└── src/                   # Source code modules
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenCV
- Ultralytics YOLOv8
- NumPy
- A webcam or camera device

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd HSE_Vision
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

### Usage Options

When you run `main.py`, you'll see a menu with the following options:

1. **Professional Helmet Detection (Recommended)** - Full-featured detection system
2. **Simple Test** - Basic detection functionality
3. **Camera Test** - Test camera connectivity
4. **Exit** - Close the application

## 🎮 Controls

### Professional Detection System Controls:
- **Q** - Quit the application
- **S** - Save screenshot
- **R** - Reset statistics
- **C** - Toggle statistics panel
- **G** - Toggle FPS graph
- **ESC** - Emergency exit

## 📊 Performance Features

### Real-time Monitoring
- **FPS Tracking**: Live frame rate monitoring with historical data
- **Detection Rate**: Percentage of frames with detections
- **Confidence Analysis**: Average confidence scores
- **Inference Time**: Model processing speed
- **Alert System**: Safety violation notifications

### Professional Reporting
- Session summary with performance metrics
- Automatic performance rating (A+ to F)
- Detailed statistics export
- Professional screenshot naming

## 🔧 Configuration

### Camera Settings
- Default resolution: 1280x720
- Frame rate: 30 FPS (auto-adjusts based on performance)
- Buffer optimization for real-time processing

### Detection Parameters
- Confidence threshold: 0.5
- Alert threshold: 5 consecutive "no-helmet" detections
- Statistics history: 1000 frames
- Performance tracking: Extended metrics

## 📈 Model Information

### Supported Models
- **YOLOv8s** (Recommended) - Balance of speed and accuracy
- **YOLOv8n** - Fastest, lower accuracy
- **YOLOv8m** - Higher accuracy, slower
- **Custom trained models** - Project-specific models

### Detection Classes
- **Helmet** - Person wearing safety helmet
- **No-Helmet** - Person without safety helmet

## 🛠️ Development

### Adding New Features
1. Create new scripts in the `scripts/` directory
2. Add documentation to `docs/`
3. Update the main menu in `main.py`
4. Test thoroughly before deployment

### Model Training
1. Prepare datasets in `data/raw/`
2. Process data and save to `data/processed/`
3. Train models and save to `models/trained/`
4. Optimize models for deployment

## 📚 Documentation

Detailed documentation is available in the `docs/` folder:
- **PROFESSIONAL_FEATURES.md** - Complete feature documentation
- **CAMERA_TESTING.md** - Camera setup and testing guide
- Additional technical documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the documentation in `docs/`
- Review the example scripts in `scripts/`
- Test camera setup with `test_camera.py`

## 🏆 Performance Ratings

The system automatically rates performance based on:
- **A+**: >95% detection rate, >25 FPS, <50ms inference
- **A**: >90% detection rate, >20 FPS, <75ms inference
- **B**: >80% detection rate, >15 FPS, <100ms inference
- **C**: >70% detection rate, >10 FPS, <150ms inference
- **D**: >60% detection rate, >5 FPS, <200ms inference
- **F**: Below minimum thresholds

---

**HSE Vision Team** - Professional Safety Monitoring Solutions