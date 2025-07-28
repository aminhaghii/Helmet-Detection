"""
Tests for core functionality modules.
"""

import pytest
import sys
import os
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestCoreModules:
    """Test core module functionality."""
    
    def test_core_imports(self):
        """Test that core modules can be imported."""
        try:
            # Try to import core modules if they exist
            import core
            assert hasattr(core, '__file__')
        except ImportError:
            # If core module doesn't exist, skip this test
            pytest.skip("Core module not yet implemented")
    
    def test_numpy_functionality(self):
        """Test numpy functionality for image processing."""
        # Test basic numpy operations that would be used in computer vision
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.shape == (5,)
        assert arr.mean() == 3.0
        
        # Test 2D array (image-like)
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        assert img_array.shape == (100, 100, 3)
        assert img_array.dtype == np.uint8


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_path_operations(self):
        """Test path operations."""
        project_root = os.path.join(os.path.dirname(__file__), '..')
        assert os.path.exists(project_root)
        
        # Test that we can create relative paths
        config_path = os.path.join(project_root, 'config')
        assert os.path.exists(config_path)
    
    def test_file_operations(self):
        """Test basic file operations."""
        # Test that we can read requirements.txt
        project_root = os.path.join(os.path.dirname(__file__), '..')
        req_file = os.path.join(project_root, 'requirements.txt')
        
        if os.path.exists(req_file):
            with open(req_file, 'r') as f:
                content = f.read()
                assert len(content) > 0
                assert 'ultralytics' in content or 'torch' in content


class TestDataStructures:
    """Test data structures and types."""
    
    def test_detection_data_structure(self):
        """Test detection result data structure."""
        # Simulate a detection result
        detection = {
            'bbox': [100, 100, 200, 200],
            'confidence': 0.85,
            'class': 'helmet',
            'class_id': 0
        }
        
        assert 'bbox' in detection
        assert 'confidence' in detection
        assert 'class' in detection
        assert len(detection['bbox']) == 4
        assert 0 <= detection['confidence'] <= 1
    
    def test_config_data_structure(self):
        """Test configuration data structure."""
        # Simulate a config structure
        config = {
            'model': {
                'name': 'yolov8l',
                'confidence_threshold': 0.5,
                'iou_threshold': 0.4
            },
            'data': {
                'image_size': 640,
                'batch_size': 16
            }
        }
        
        assert 'model' in config
        assert 'data' in config
        assert config['model']['confidence_threshold'] > 0
        assert config['data']['image_size'] > 0


if __name__ == "__main__":
    pytest.main([__file__])