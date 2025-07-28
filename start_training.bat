@echo off
echo ========================================
echo HSE Vision - YOLOv8s Training Script
echo ========================================
echo.
echo Starting helmet detection model training...
echo Model: YOLOv8-Small (yolov8s)
echo Dataset: 2,999 training + 857 validation images
echo Expected time: 3-5 hours
echo.

cd /d "c:\Users\aminh\OneDrive\Desktop\Projects\HSE_Vision"

echo Running training command...
python src/scripts/train_model.py --data unified_dataset/data.yaml --no-wandb

echo.
echo ========================================
echo Training completed!
echo Check the models/trained/ directory for results.
echo ========================================
pause