# 🔤 Square CRNN OCR: Alphanumeric License Plate Recognition System

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-00FFFF.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-precision, deep learning-powered Optical Character Recognition (OCR) framework designed for Vietnamese and international alphanumeric license plates. The repository provides a end-to-end training and evaluation pipeline combining ResNet convolutional feature extractors, spatial self-attention mechanisms, bidirectional LSTM sequence modeling, and CTC loss decoding.

---

## 🖼️ End-to-End Training Pipeline

The training workflow handles raw images, smart augmentation, model training, and automatic checkpointing:

![SquareCRNN Training Pipeline](training_pipeline.png)

---

## 🏛️ Model Architecture (`SquareCRNN`)

The network architecture seamlessly fuses Computer Vision and Sequence Modeling:

```
Input (1 x 128 x 128 Grayscale Image)
  │
  ├── 🟢 Conv2d (1 -> 64, kernel=3, stride=1, padding=1) + BatchNorm + ReLU
  ├── 🔹 MaxPool2d (2x2, stride 2)
  │
  ├── 🟢 ResidualBlock 1 (64 -> 128, Dropout=0.1)
  ├── 🔹 MaxPool2d (2x2, stride 2)
  │
  ├── 🟢 ResidualBlock 2 (128 -> 256, Dropout=0.3)
  ├── 🔹 MaxPool2d (2x2, stride=(2,1), padding=(0,1))
  │
  ├── 🟢 ResidualBlock 3 (256 -> 512, Dropout=0.3)
  ├── 🔹 MaxPool2d (2x2, stride=(2,1), padding=(0,1))
  │
  ├── 🟢 ResidualBlock 4 (512 -> 512, Dropout=0.3)
  ├── 🌟 SimpleAttention Layer (Spatial Feature Weighting)
  ├── 🔹 AdaptiveAvgPool2d (1 x W)
  │
  ├── 🔁 2-Layer Bidirectional LSTM (Hidden Size: 256, Dropout=0.4)
  └── 🎯 Fully Connected CTC Output Layer (Hidden*2 -> 37 Classes)
```

---

## 📊 Dataset Format & Preprocessing

### CSV Annotation Format (`train.csv`)
Training annotations are stored in a standard CSV format with image filename and target string:

```csv
Tên bức ảnh,Nội dung bức ảnh
img_0.jpg,70C-159.51
img_1.jpg,59H-333.33
img_2.jpg,51F-645.85
img_3.jpg,43A-123.45
```

### Supported Character Set
Default vocabulary consists of **38 tokens** (digits `0-9`, uppercase letters `A-Z`, hyphen `-`, dot `.`, plus CTC blank token at index 0):

```python
CHARACTER_SET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-."
```

### Smart On-the-Fly Data Augmentation
To mimic real-world camera artifacts (blur, lighting, perspective distortion), `OCRDataset` applies:
- 💡 **Brightness & Contrast Scaling**: $\alpha \in [0.7, 1.3]$, $\beta \in [-30, 30]$
- 📐 **Affine Angle Rotation**: Random rotation $\pm 5^\circ$ with edge fill
- 🌫️ **Gaussian Noise Injection**: $\sigma = 15$ for low-light camera simulation

---

## 📁 Repository Directory Structure

```
train_ocr/
├── src/                          # Primary source modules
│   ├── __init__.py               # Package exports
│   ├── config.py                 # Hyperparameter configuration
│   ├── model.py                  # PyTorch SquareCRNN neural network
│   ├── train.py                  # Training loop & dataset loader
│   ├── data.py                   # Data cleaning & format conversion
│   └── name_img.py               # Batch image renamer utility
├── notebooks/                    # Interactive development
│   └── train.ipynb               # Jupyter notebook training pipeline
├── docs/                         # Detailed documentation
│   ├── API.md                    # Module and function reference
│   ├── DATASET.md                # Dataset specification
│   ├── QUICKSTART.md             # Fast setup instructions
│   └── TROUBLESHOOTING.md        # Debugging guide
├── images/                       # Training and sample dataset images
├── models/                       # Weights directory (best_square_ocr_pro.pth)
├── tests/                        # Integration pipeline tests
│   └── test.py                   # E2E YOLO + OCR pipeline test
├── train.csv                     # Training dataset annotations
├── training_pipeline.png         # Pipeline flowchart diagram
├── requirements.txt              # Environment dependencies
├── LICENSE                       # MIT License
└── README.md                     # Main documentation
```

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/phanhuynhvando/train_ocr.git
cd train_ocr

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Model Training
Run the training script to train `SquareCRNN` with CTC Loss and adaptive learning rate scheduling:

```bash
python3 src/train.py
```

The script will automatically detect CUDA GPU acceleration and save the best model weights to `models/best_square_ocr_pro.pth`.

### 3. Training Hyperparameters

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| **Input Dimensions** | `128 x 128` | Grayscale image input size |
| **Batch Size** | `32` | Samples per mini-batch |
| **Learning Rate** | `0.001` | Initial AdamW learning rate |
| **Scheduler** | `ReduceLROnPlateau` | Patience: 4 epochs, Factor: 0.5 |
| **Weight Decay** | `1e-4` | L2 regularization |
| **Gradient Clip** | `5.0` | Max norm clipping for stability |
| **Loss Function** | `CTCLoss` | Connectionist Temporal Classification |

### 4. End-to-End Inference (YOLO Detection + OCR)
Run the full detection and recognition test pipeline:

```bash
python3 tests/test.py
```

```python
# Code snippet for inference usage:
import cv2
import torch
from src.model import SquareCRNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SquareCRNN(num_classes=38).to(device)
model.load_state_dict(torch.load("models/best_square_ocr_pro.pth", map_location=device))
model.eval()

# Preprocess image
img = cv2.imread("images/img_0.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (128, 128))
tensor = torch.from_numpy((img.astype('float32') / 127.5) - 1.0).unsqueeze(0).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    preds = model(tensor)
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Phan Huỳnh Văn Đô**  
GitHub: [@phanhuynhvando](https://github.com/phanhuynhvando)
