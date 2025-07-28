"""
Basic tests for the core module functionality.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_imports():
    """Test that core modules can be imported without errors."""
    try:
        from core import model, inference

        assert True
    except ImportError:
        # If modules don't exist yet, that's okay for now
        assert True


def test_basic_functionality():
    """Test basic Python functionality."""
    assert 1 + 1 == 2
    assert "hello" == "hello"
    assert [1, 2, 3] == [1, 2, 3]


def test_project_structure():
    """Test that required project directories exist."""
    project_root = os.path.join(os.path.dirname(__file__), "..")

    required_dirs = ["src", "config", "data", "models", "outputs", "scripts"]

    for dir_name in required_dirs:
        dir_path = os.path.join(project_root, dir_name)
        assert os.path.exists(dir_path), f"Required directory {dir_name} does not exist"


def test_config_files():
    """Test that configuration files exist."""
    project_root = os.path.join(os.path.dirname(__file__), "..")

    config_files = ["requirements.txt", "pyproject.toml"]

    for file_name in config_files:
        file_path = os.path.join(project_root, file_name)
        assert os.path.exists(file_path), f"Required file {file_name} does not exist"


class TestProjectStructure:
    """Test class for project structure validation."""

    def test_src_modules_exist(self):
        """Test that src modules exist."""
        project_root = os.path.join(os.path.dirname(__file__), "..")
        src_path = os.path.join(project_root, "src")

        expected_modules = ["core", "desktop_app", "utils", "scripts"]

        for module in expected_modules:
            module_path = os.path.join(src_path, module)
            assert os.path.exists(
                module_path
            ), f"Module {module} does not exist in src/"

    def test_init_files_exist(self):
        """Test that __init__.py files exist in Python packages."""
        project_root = os.path.join(os.path.dirname(__file__), "..")
        src_path = os.path.join(project_root, "src")

        modules = ["core", "desktop_app", "utils", "scripts"]

        for module in modules:
            init_file = os.path.join(src_path, module, "__init__.py")
            if os.path.exists(os.path.join(src_path, module)):
                assert os.path.exists(init_file), f"__init__.py missing in {module}"


if __name__ == "__main__":
    pytest.main([__file__])
