"""
Square CRNN OCR Package

A high-performance OCR system using Convolutional Recurrent Neural Networks
with self-attention mechanism for recognizing alphanumeric characters.

Modules:
    model: Neural network architecture (CRNN with Attention)
    train: Training script with data loading
    data: Data preprocessing utilities
    config: Configuration management

Example:
    >>> from src.model import SquareCRNN
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
    'ResidualBlock',
    'SimpleAttention',
    'OCRDataset',
    'collate_fn',
    'get_config',
]

try:
    from src.model import SquareCRNN, ResidualBlock, SimpleAttention
    from src.train import OCRDataset, collate_fn, main as train_main
    from src.config import get_config
except ImportError:
    try:
        from model import SquareCRNN, ResidualBlock, SimpleAttention
        from train import OCRDataset, collate_fn, main as train_main
        from config import get_config
    except ImportError:
        pass
