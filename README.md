# Square CRNN OCR

🚀 **Specialized Optical Character Recognition using CRNN with Self-Attention**

A high-performance OCR system for recognizing alphanumeric characters in images using Convolutional Recurrent Neural Networks (CRNN) with self-attention mechanism.

## 📖 Documentation

Complete documentation is available in the [`docs/`](docs/) folder:

- **[Main README](docs/README.md)** - Overview, features, and architecture
- **[Quick Start](docs/QUICKSTART.md)** - Get started in minutes
- **[Setup Instructions](docs/SETUP.md)** - Detailed setup guide
- **[API Reference](docs/API.md)** - API documentation
- **[Dataset Documentation](docs/DATASET.md)** - Data format and preparation
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Benchmarks](docs/BENCHMARKS.md)** - Performance metrics
- **[Changelog](docs/CHANGELOG.md)** - Version history

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/minhtenlado/train_ocr.git
cd train_ocr

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run training
python src/train.py

# 4. Test inference
python src/test.py
```

## 📁 Project Structure

```
train_ocr/
├── src/           # Source code (models, training scripts, utilities)
├── docs/          # Documentation and guides
├── notebooks/     # Jupyter notebooks for interactive development
├── models/        # Trained model weights and checkpoints
├── data/          # Training data and datasets
├── images/        # Sample images and diagrams
├── tests/         # Test files
├── requirements.txt
├── LICENSE
└── README.md      # This file
```

## ⭐ Key Features

- 🧠 **Advanced Architecture**: CRNN with Residual Blocks and Self-Attention
- 📊 **Smart Data Augmentation**: Brightness, rotation, noise injection
- 🎯 **Robust Training**: CTC Loss, AdamW optimizer, Learning rate scheduling
- 🔄 **Bidirectional LSTM**: 2-layer LSTM for sequence modeling
- ⚡ **GPU Acceleration**: Full CUDA/GPU support
- 📈 **Production Ready**: Well-tested and optimized

## 🤖 Model Architecture

### Complete Training Pipeline

![SquareCRNN Training Pipeline](images/training_pipeline.png)

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

Modify in `src/train.py`:
```python
chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-."
char_map = {c: i + 1 for i, c in enumerate(chars)}
```

## 🔧 Data Format

### CSV Format (data/train.csv)
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

MIT License - see [LICENSE](LICENSE) for details

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

---

👉 Start with [docs/README.md](docs/README.md) for detailed information!
