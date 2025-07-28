@echo off
echo ========================================
echo    Helmet Detection Camera Test
echo ========================================
echo.

echo Checking Python environment...
python --version
echo.

echo Starting helmet detection test...
echo.
echo Controls:
echo   - Press 'q' to quit
echo   - Press 's' to save screenshot
echo   - Press 'r' to reset counter
echo.

python test_camera.py

echo.
echo Test completed!
pause