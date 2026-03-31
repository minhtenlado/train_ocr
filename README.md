# Square CRNN OCR - Specialized Optical Character Recognition

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance OCR (Optical Character Recognition) system using **CRNN (Convolutional Recurrent Neural Networks)** with self-attention mechanism for recognizing alphanumeric characters in images.

## 🎯 Features

- **Advanced Architecture**: CRNN with Residual Blocks and Self-Attention mechanism
- **Data Augmentation**: Intelligent augmentation including brightness, rotation, and noise injection
- **Robust Training**: Implemented with:
  - CTC (Connectionist Temporal Classification) loss
  - Gradient clipping for stability
  - Learning rate scheduling with ReduceLROnPlateau
  - Weight initialization (Kaiming for Conv layers)
  - Dropout regularization (0.3-0.4)
- **LSTM Sequence Modeling**: Bidirectional LSTM (2 layers) for sequence recognition
- **GPU Support**: Full CUDA/GPU acceleration

## 📋 Project Structure

```
train_ocr/
├── model.py              # CRNN model architecture with attention
├── train.py              # Training script with CTC loss
├── test.py               # Inference and testing utilities
├── data.py               # Data preprocessing utilities
├── name_img.py           # Image naming utilities
├── train.ipynb           # Jupyter notebook for interactive training
├── train.csv             # Training dataset annotations
├── dataset/              # Dataset directory (images and labels)
├── images/               # Sample images directory
├── best_square_ocr_pro.pth  # Best trained model weights
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- CUDA 11.0+ (optional, for GPU acceleration)
- pip or conda

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/square-crnn-ocr.git
cd square-crnn-ocr
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/macOS
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🏋️ Training

### Basic Training
```bash
python train.py
```

### Using Jupyter Notebook
```bash
jupyter notebook train.ipynb
```

### Configuration
Edit the `main()` function in `train.py` to customize:
- Dataset paths
- Batch size (default: 32)
- Learning rate (default: 0.001)
- Number of epochs (default: 150)
- Image size (default: 128x128)
- Hidden size (default: 256)

## 🧪 Testing/Inference

```bash
python test.py
```

Modify paths in `test.py` to point to your model and images.

## 📊 Model Architecture

### Complete Training Pipeline

![SquareCRNN Training Pipeline](train_ocr/training_pipeline.png)

*Sơ đồ quy trình huấn luyện hoàn chỉnh bao gồm:*
- **Dữ liệu đầu vào**: OCRDataset với hình ảnh và nhãn
- **Tăng cường dữ liệu**: Điều chỉnh độ sáng, xoay, thêm nhiễu
- **Mô hình SquareCRNN**: 4 khối Residual, Self-Attention, Bidirectional LSTM
- **Quy trình huấn luyện**: CTC Loss, optimize với AdamW, ReduceLROnPlateau scheduler
- **Lưu mô hình tốt nhất**: best_square_ocr_pro.pth

### SquareCRNN Components:

1. **CNN Feature Extraction**
   - Initial Conv2d: 1→64 channels
   - Residual Blocks: 64→128→256→512→512
   - Max Pooling layers for dimension reduction
   - Self-Attention mechanism on final 512 features

2. **Self-Attention Module**
   - Query/Key projections (channel // 8)
   - Weighted attention over spatial features
   - Additive residual connection

3. **RNN Sequence Modeling**
   - Bidirectional LSTM: 512 → 256 (×2 layers)
   - Dropout: 0.4 for regularization

4. **Classification**
   - Fully connected layer: 512 → num_classes + 1 (blank token)
   - Output: CTC loss compatible

## 📈 Supported Characters

Default character set: `0-9 A-Z - .`
```
0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.
```

Modify in `train.py`:
```python
chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-."
char_map = {c: i + 1 for i, c in enumerate(chars)}
```

## 🔧 Data Format

### CSV Format (train.csv)
```csv
image_filename.jpg,TEXT_CONTENT
image2.jpg,ANOTHER_TEXT
```

### Image Requirements
- Format: JPEG, PNG, or other OpenCV-supported formats
- Color: Grayscale or RGB (converted to grayscale)
- Size: Flexible (resized to 128×128 during preprocessing)
- Quality: Good contrast for optimal recognition

## 📦 Dependencies

Main dependencies (see `requirements.txt`):
- PyTorch >= 2.0
- OpenCV (cv2)
- NumPy
- Pandas
- Ultralytics (YOLOv8 - optional for detection preprocessing)

## 🎓 Data Augmentation

The model includes intelligent augmentation:
- **Brightness adjustment**: 0.7-1.3 range with ±30 bias
- **Rotation**: ±5 degrees (mimics tilted camera)
- **Gaussian noise**: Random noise injection (realistic camera conditions)

## 📋 Training Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch Size | 32 | Adjust based on GPU memory |
| Learning Rate | 0.001 | Initial; reduced by scheduler |
| Weight Decay | 1e-4 | L2 regularization |
| Epochs | 150 | Stops early if loss plateau |
| Image Size | 128×128 | Input dimension |
| Dropout | 0.3-0.4 | Layer-specific |
| Gradient Clip | 5.0 | Max norm for stability |

## 🔍 Troubleshooting

### Out of Memory (OOM)
- Reduce batch size: `batch_size=16` or lower
- Reduce `hidden_size`: Change from 256 to 128

### Poor accuracy
- Check data quality (clear, high-contrast images)
- Ensure correct character mapping
- Train for more epochs with learning rate scheduler

### Slow training
- Use GPU: Ensure CUDA is properly installed
- Check device with: `torch.cuda.is_available()`

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📧 Contact & Support

For issues, questions, or suggestions, please open an issue on GitHub.

## 🙏 Acknowledgments

- Built with PyTorch and OpenCV
- Inspired by industry-standard OCR architectures (CRNN)
- Self-attention mechanism based on modern deep learning research

---

⭐ If this project helped you, please consider starring it on GitHub!
