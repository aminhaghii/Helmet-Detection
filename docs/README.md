# HSE Vision - Construction Safety Detection System

A real-time helmet detection system using YOLOv8-Small with GPU optimization for construction site safety monitoring.

## 🚀 Features

- **YOLOv8-Small Model**: Fast and efficient helmet detection optimized for real-time performance
- **GPU Acceleration**: Optimized for NVIDIA GPUs with CUDA support
- **Modern GUI**: User-friendly desktop application with real-time statistics
- **High Performance**: Optimized inference engine with threading and memory management
- **Unified Dataset**: 3,856 images (2,999 training + 857 validation) from multiple sources
- **Export Options**: Support for ONNX, TensorRT, and TorchScript formats
- **Comprehensive Logging**: Performance monitoring and detailed statistics

## 📋 Requirements

### Hardware
- NVIDIA GPU (RTX 4050 or better recommended)
- 8GB+ RAM
- USB Camera or built-in webcam

### Software
- Windows 10/11
- Python 3.8+
- CUDA 11.8+
- TensorRT 8.5+ (optional, for optimization)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd HSE_Vision
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify GPU Setup
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

## 📊 Dataset Structure

The project uses 6 unified helmet detection datasets located in the `Data/` directory:

```
Data/
├── 1/  # Dataset 1 - Worksite Safety Monitoring
├── 2/  # Dataset 2 - Helmet Detection
├── 3/  # Dataset 3 - Construction Safety
├── 4/  # Dataset 4 - PPE Detection
├── 5/  # Dataset 5 - Safety Equipment
└── 6/  # Dataset 6 - Worker Safety
```

Each dataset contains:
- `train/`, `valid/`, `test/` splits
- `images/` and `labels/` directories
- `data.yaml` configuration file

## 🏃‍♂️ Quick Start

### 1. Run the Desktop Application
```bash
python main.py
```

### 2. Train Custom Model (Optional)
```bash
python src/scripts/train_model.py --validate --export --benchmark
```

### 3. Test Camera
1. Launch the application
2. Select camera ID (usually 0 for built-in camera)
3. Choose resolution (640x480 recommended for real-time)
4. Click "Start Camera"
5. Click "Start Detection" to begin helmet detection

## 🎯 Usage Guide

### Desktop Application

#### Camera Setup
1. **Camera Selection**: Choose camera device ID (0 for default)
2. **Resolution**: Select appropriate resolution for your hardware
3. **Start Camera**: Initialize camera feed

#### Detection Controls
1. **Confidence Threshold**: Adjust detection sensitivity (0.1-1.0)
2. **Start Detection**: Begin real-time helmet detection
3. **Capture Frame**: Save current frame with detections

#### Model Management
1. **Load Model**: Load custom trained models (.pt, .onnx)
2. **Optimize Model**: Export to TensorRT for better performance

### Training Custom Models

#### Quick Start Training
```bash
# Easy training with batch file
start_training.bat

# Or run directly
python src/scripts/train_model.py --data unified_dataset/data.yaml --no-wandb
```

#### Advanced Training Options
```bash
python src/scripts/train_model.py \
    --data unified_dataset/data.yaml \
    --validate \
    --export \
    --benchmark \
    --resume  # Resume from checkpoint
```

#### Training Parameters (YOLOv8s)
- **Model**: YOLOv8-Small (yolov8s)
- **Epochs**: 150 (optimized for faster training)
- **Batch Size**: 32 (increased for better GPU utilization)
- **Input Size**: 640x640 pixels
- **Optimizer**: AdamW with learning rate 0.01
- **Expected Training Time**: 3-5 hours on RTX 4050

## ⚙️ Configuration

### Training Configuration (`config/train_config.yaml`)
```yaml
model: yolov8s
epochs: 150
batch: 32
imgsz: 640
device: cuda
optimizer: AdamW
lr0: 0.01
```

### Deployment Configuration (`config/deployment_config.yaml`)
```yaml
camera:
  device_id: 0
  resolution:
    width: 640
    height: 480
  fps: 30

detection:
  confidence_threshold: 0.5
  iou_threshold: 0.45
```

## 🚀 Performance Optimization

### GPU Optimization
1. **CUDA**: Automatic GPU detection and utilization
2. **Mixed Precision**: FP16 training for faster performance
3. **TensorRT**: Export models for optimized inference
4. **Memory Management**: Efficient GPU memory usage

### Real-time Performance Tips
1. Use lower resolution (640x480) for real-time processing
2. Adjust confidence threshold based on accuracy needs
3. Enable TensorRT optimization for production use
4. Monitor GPU memory usage in statistics panel

## 📈 Performance Metrics

The application provides comprehensive performance monitoring:

- **FPS**: Real-time frames per second
- **Inference Time**: Model prediction latency
- **Detection Count**: Number of objects detected
- **Safety Score**: Helmet compliance ratio
- **GPU Usage**: Memory and utilization statistics

## 🔧 Troubleshooting

### Common Issues

#### Camera Not Working
```bash
# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.read()[0] else 'Camera Failed')"
```

#### CUDA Not Available
```bash
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Low Performance
1. Reduce camera resolution
2. Lower confidence threshold
3. Close other GPU applications
4. Enable TensorRT optimization

#### Memory Issues
1. Reduce batch size in training
2. Lower image resolution
3. Enable gradient checkpointing
4. Monitor GPU memory usage

## 📁 Project Structure

```
HSE_Vision/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── config/                 # Configuration files
│   ├── data.yaml
│   ├── model_config.yaml
│   └── deployment_config.yaml
├── src/                    # Source code
│   ├── core/              # Core detection modules
│   ├── desktop_app/       # GUI application
│   ├── utils/             # Utility functions
│   └── scripts/           # Training scripts
├── models/                # Model storage
│   ├── pretrained/
│   ├── trained/
│   └── optimized/
├── Data/                  # Training datasets
├── logs/                  # Application logs
└── outputs/               # Detection outputs
```

## 🎯 Model Performance

### YOLOv8s Specifications
- **Model Size**: ~22MB (YOLOv8-Small)
- **Parameters**: 11.2M
- **FLOPs**: 28.6G
- **Inference Speed**: 20-40 FPS (RTX 4050)
- **Training Time**: 3-5 hours (150 epochs)

### Expected Results
- **mAP@0.5**: 0.85+ (helmet detection)
- **mAP@0.5:0.95**: 0.75+ (overall performance)
- **Real-time Performance**: Optimized for speed
- **Memory Usage**: Low GPU memory footprint

### Supported Classes
- `helm`: Person wearing helmet (safe)
- `no-helm`: Person without helmet (unsafe)

## 🔄 Updates and Maintenance

### Model Updates
1. Retrain with new data: `python src/scripts/train_model.py`
2. Export optimized version: Use `--export` flag
3. Update deployment config if needed

### Performance Monitoring
- Check logs in `logs/` directory
- Monitor GPU usage during operation
- Review detection statistics in GUI

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with proper testing
4. Submit pull request with detailed description

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for the base detection framework
- **Datasets**: Multiple Roboflow contributors for helmet detection datasets
- **Libraries**: PyTorch, OpenCV, CustomTkinter communities

## 📞 Support

For issues and questions:
1. Check troubleshooting section
2. Review logs in `logs/` directory
3. Create GitHub issue with detailed description
4. Include system specifications and error logs

---

**HSE Vision Team** - Construction Safety Through AI Detection