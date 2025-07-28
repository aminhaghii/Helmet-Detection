# Contributing to HSE Vision

Thank you for your interest in contributing to HSE Vision! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### 1. Fork the Repository
- Fork the HSE Vision repository to your GitHub account
- Clone your fork locally:
```bash
git clone https://github.com/your-username/HSE_Vision.git
cd HSE_Vision
```

### 2. Set Up Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 isort
```

### 3. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 4. Make Your Changes
- Follow the existing code style and conventions
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 5. Commit Your Changes
```bash
git add .
git commit -m "feat: add your feature description"
```

### 6. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```
Then create a pull request on GitHub.

## 📝 Code Style Guidelines

### Python Code Style
- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Maximum line length: 88 characters (Black formatter)

### Commit Message Format
```
type: brief description

Detailed description if needed

Types:
- feat: new feature
- fix: bug fix
- docs: documentation changes
- style: formatting changes
- refactor: code refactoring
- test: adding tests
- chore: maintenance tasks
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_specific.py

# Run with coverage
pytest --cov=src tests/
```

### Writing Tests
- Write unit tests for all new functions
- Use descriptive test names
- Include edge cases and error conditions
- Place tests in the `tests/` directory

## 📚 Documentation

### Code Documentation
- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Include parameter types and return values
- Add usage examples for complex functions

### README Updates
- Update README.md if adding new features
- Include installation and usage instructions
- Add screenshots for UI changes

## 🐛 Bug Reports

When reporting bugs, please include:
- Operating system and version
- Python version
- HSE Vision version
- Steps to reproduce the bug
- Expected vs actual behavior
- Error messages and stack traces

## 💡 Feature Requests

For feature requests, please provide:
- Clear description of the feature
- Use case and motivation
- Proposed implementation approach
- Any relevant examples or mockups

## 🔍 Code Review Process

1. All contributions require code review
2. Maintainers will review pull requests
3. Address feedback and update your PR
4. Once approved, your PR will be merged

## 📋 Development Setup Checklist

- [ ] Fork and clone the repository
- [ ] Set up virtual environment
- [ ] Install dependencies
- [ ] Run tests to ensure everything works
- [ ] Create feature branch
- [ ] Make changes with tests
- [ ] Run linting and formatting
- [ ] Update documentation
- [ ] Submit pull request

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- GitHub contributors page

## 📞 Getting Help

- Open an issue for questions
- Check existing issues and documentation
- Join discussions in pull requests

## 📄 License

By contributing to HSE Vision, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to HSE Vision! 🚀