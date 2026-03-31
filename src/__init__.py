"""
Square CRNN OCR Package

A high-performance OCR system using Convolutional Recurrent Neural Networks
with self-attention mechanism for recognizing alphanumeric characters.

Modules:
    model: Neural network architecture (CRNN with Attention)
    train: Training script with data loading
    test: Inference and evaluation utilities
    data: Data preprocessing utilities
    config: Configuration management

Example:
    >>> from model import SquareCRNN
    >>> import torch
    >>> model = SquareCRNN(num_classes=36)
    >>> x = torch.randn(32, 1, 128, 128)
    >>> output = model(x)
"""

__version__ = "1.0.0"
__author__ = "OCR Development Team"
__license__ = "MIT"

__all__ = [
    'SquareCRNN',
    'OCRDataset',
    'collate_fn',
    'get_config',
]

# Allow direct imports from package
try:
    from model import SquareCRNN
    from train import OCRDataset, collate_fn, main as train_main
    from config import get_config
except ImportError:
    # Fallback if imports fail during setup
    pass
