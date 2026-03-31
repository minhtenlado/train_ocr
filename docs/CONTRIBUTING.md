# Contributing to Square CRNN OCR

First of all, thank you for considering contributing to Square CRNN OCR! It's people like you that make this project such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title**
* **Describe the exact steps which reproduce the problem**
* **Provide specific examples to demonstrate the steps**
* **Describe the behavior you observed after following the steps**
* **Explain which behavior you expected to see instead and why**
* **Include screenshots if possible**
* **Include your environment details** (OS, Python version, PyTorch version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title**
* **Provide a step-by-step description of the suggested enhancement**
* **Provide specific examples to demonstrate the steps**
* **Describe the current behavior and expected behavior**

### Pull Requests

* Follow the Python PEP 8 style guide
* Include appropriate test cases
* Update documentation as needed
* End all files with a newline

## Development Setup

1. Fork the repository
2. Clone your fork
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Linux/macOS) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Create a branch: `git checkout -b feature/your-feature-name`

## Code Style

* Use `black` for code formatting
* Use `flake8` for linting
* Write docstrings for all functions and classes
* Use type hints where applicable

```bash
black your_file.py
flake8 your_file.py
```

## Testing

Ensure your changes don't break existing functionality. Run tests with:

```bash
pytest
```

## Documentation

* Update README.md if behavior changes
* Add docstrings to new functions/classes
* Update comments for significant logic changes

## Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

## License

By contributing to Square CRNN OCR, you agree that your contributions will be licensed under the same MIT License that covers the project.

Thank you for contributing! 🎉
