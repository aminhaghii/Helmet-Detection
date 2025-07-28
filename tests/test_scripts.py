"""
Tests for scripts functionality.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


class TestScripts:
    """Test scripts functionality."""

    def test_scripts_directory_exists(self):
        """Test that scripts directory exists."""
        project_root = os.path.join(os.path.dirname(__file__), "..")
        scripts_path = os.path.join(project_root, "scripts")
        assert os.path.exists(scripts_path), "Scripts directory does not exist"

    def test_script_files_exist(self):
        """Test that expected script files exist."""
        project_root = os.path.join(os.path.dirname(__file__), "..")
        scripts_path = os.path.join(project_root, "scripts")

        if os.path.exists(scripts_path):
            script_files = os.listdir(scripts_path)
            python_scripts = [f for f in script_files if f.endswith(".py")]
            assert (
                len(python_scripts) > 0
            ), "No Python scripts found in scripts directory"

    def test_script_imports(self):
        """Test that scripts can be imported without syntax errors."""
        project_root = os.path.join(os.path.dirname(__file__), "..")
        scripts_path = os.path.join(project_root, "scripts")

        if os.path.exists(scripts_path):
            script_files = [f for f in os.listdir(scripts_path) if f.endswith(".py")]

            for script_file in script_files:
                script_name = script_file[:-3]  # Remove .py extension
                try:
                    # Try to compile the script to check for syntax errors
                    script_path = os.path.join(scripts_path, script_file)
                    with open(script_path, "r", encoding="utf-8") as f:
                        script_content = f.read()

                    # Compile to check for syntax errors
                    compile(script_content, script_path, "exec")
                    assert True  # If we get here, no syntax errors

                except SyntaxError as e:
                    pytest.fail(f"Syntax error in {script_file}: {e}")
                except Exception:
                    # Other errors (like import errors) are okay for now
                    pass


class TestConfigFiles:
    """Test configuration files."""

    def test_yaml_files_exist(self):
        """Test that YAML configuration files exist."""
        project_root = os.path.join(os.path.dirname(__file__), "..")
        config_path = os.path.join(project_root, "config")

        if os.path.exists(config_path):
            yaml_files = [f for f in os.listdir(config_path) if f.endswith(".yaml")]
            assert len(yaml_files) > 0, "No YAML configuration files found"

    def test_yaml_files_valid(self):
        """Test that YAML files are valid."""
        project_root = os.path.join(os.path.dirname(__file__), "..")
        config_path = os.path.join(project_root, "config")

        if os.path.exists(config_path):
            yaml_files = [f for f in os.listdir(config_path) if f.endswith(".yaml")]

            for yaml_file in yaml_files:
                yaml_path = os.path.join(config_path, yaml_file)
                try:
                    import yaml

                    with open(yaml_path, "r", encoding="utf-8") as f:
                        yaml.safe_load(f)
                    assert True  # If we get here, YAML is valid
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {yaml_file}: {e}")
                except ImportError:
                    # If PyYAML is not installed, skip this test
                    pytest.skip("PyYAML not available for YAML validation")


if __name__ == "__main__":
    pytest.main([__file__])
