# Quick Start Guide

Get started with Square CRNN OCR in 5 minutes!

## 1️⃣ Installation (2 minutes)

```bash
# Clone repository
git clone https://github.com/yourusername/square-crnn-ocr.git
cd square-crnn-ocr

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

## 2️⃣ Prepare Your Data (1 minute)

Create a CSV file `train.csv`:
```csv
image_001.jpg,ABC123
image_002.jpg,XYZ789
image_003.jpg,HELLO-WORLD
```

Create folder structure:
```
train_ocr/
├── train.csv          # Your CSV file
└── images/            # Your images here
    ├── image_001.jpg
    ├── image_002.jpg
    └── image_003.jpg
```

## 3️⃣ Train Model (2 minutes of setup)

```python
# Run training
python train.py

# Or use Jupyter notebook
jupyter notebook train.ipynb
```

That's it! Your model will start training.

---

## 📖 Key Links

**Before Training:**
- [SETUP.md](SETUP.md) - Detailed installation
- [DATASET.md](DATASET.md) - Prepare your data
- [FAQ](#faq-section) - Common questions

**During Training:**
- [BENCHMARKS.md](BENCHMARKS.md) - Expected performance
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - When things go wrong

**After Training:**
- [API.md](API.md) - Use your model
- [README.md](README.md) - Full documentation

---

## ⚡ Quick Configuration

### For GPU Training (Recommended)
```bash
# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, check SETUP.md GPU section
```

### For Faster Training
Edit `train.py` main():
```python
batch_size = 64        # Increase from 32
num_epochs = 50        # Reduce from 150 for testing
```

### For Better Accuracy
```python
num_epochs = 200       # Train longer
batch_size = 16        # Smaller batches
dropout_rate = 0.4     # More regularization
```

---

## 🧪 Test Your Setup

Before training, verify everything works:

```bash
# Check Python version
python --version

# Check PyTorch installation
python -c "import torch; print(torch.__version__)"

# Check GPU (if using CUDA)
python -c "import torch; print(torch.cuda.is_available())"

# Check OpenCV
python -c "import cv2; print(cv2.__version__)"
```

Expected output:
```
Python 3.8.0 (or higher)
2.0.0+cu118 (or similar)
True (if GPU available)
4.8.0 (or similar)
```

---

## ⚙️ Configuration Hints

**For Different Hardware:**

| Hardware | Recommended Settings |
|----------|----------------------|
| GPU (RTX 2080+) | batch_size=64, epochs=150, lr=0.001 |
| GPU (GTX 1080) | batch_size=32, epochs=100, lr=0.001 |
| GPU (RTX 3060) | batch_size=48, epochs=120, lr=0.001 |
| CPU Intel i7 | batch_size=8, epochs=50, num_workers=0 |

---

## 📊 Monitoring Training

During training, you'll see:

```
Scanning and cleaning data...
-> Removed: 0 missing images, 0 invalid labels.
-> Ready to train: 1000 samples.

--- Starting training on cuda ---
Training samples: 1000

Epoch [  1/150] - Loss: 4.5321 - LR: 0.001000
Epoch [  2/150] - Loss: 3.2145 - LR: 0.001000
Epoch [  3/150] - Loss: 2.8564 - LR: 0.001000
✓ Model saved: ./best_square_ocr_pro.pth (Best Loss: 2.8564)
```

**What to expect:**
- Loss should decrease over time
- After ~50 epochs, loss plateaus
- Model saves when loss improves
- Training time: ~35 min/epoch on GPU

---

## 🔍 Using Your Trained Model

```python
import torch
from model import SquareCRNN
import cv2
import numpy as np

# Load model
model = SquareCRNN(num_classes=36)
model.load_state_dict(torch.load('best_square_ocr_pro.pth'))
model.eval()

# Prepare image
img = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (128, 128))
img = (img.astype(np.float32) / 127.5) - 1.0
img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(img)
    predictions = output.argmax(2)  # Get character indices
    
# Convert to text (example)
chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-."
text = ''.join([chars[p-1] for p in predictions[:, 0] if p > 0])
print(f"Recognized text: {text}")
```

---

## ❓ FAQ Section

### Q: What if training is very slow?
A: You're likely using CPU. Check SETUP.md GPU section to enable CUDA.

### Q: How much data do I need?
A: Start with 100+ samples. Better with 1000+. See DATASET.md for details.

### Q: What does each loss value mean?
A: Loss < 1.0 = Good, Loss < 0.5 = Excellent. See BENCHMARKS.md.

### Q: Can I use different image sizes?
A: Currently requires 128×128. Resize images before training.

### Q: How do I improve accuracy?
A: More data, longer training, clearer images. See TROUBLESHOOTING.md.

### Q: Can I train on CPU only?
A: Yes, but very slow (~10-100x slower). GPU recommended.

### Q: How do I deploy the model?
A: Save weights (done automatically), load in your application (see example above).

---

## 🚨 Common Issues

| Issue | Quick Fix |
|-------|-----------|
| No module named 'torch' | Run: `pip install -r requirements.txt` |
| CUDA out of memory | Reduce batch_size: 32 → 16 |
| Can't find images | Check CSV filenames match image files |
| Loss not decreasing | Check training data quality/quantity |
| Very slow training | Enable GPU in SETUP.md |

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for solutions.

---

## 📚 Full Documentation

After quick start, explore:
1. [README.md](README.md) - Project overview
2. [SETUP.md](SETUP.md) - Detailed setup
3. [DATASET.md](DATASET.md) - Data preparation
4. [API.md](API.md) - Code documentation
5. [BENCHMARKS.md](BENCHMARKS.md) - Performance info
6. [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Problem solving
7. [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute

---

## 💻 Commands Cheat Sheet

```bash
# Setup
git clone https://github.com/yourusername/square-crnn-ocr.git
cd square-crnn-ocr
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Training
python train.py
jupyter notebook train.ipynb

# Testing
python test.py

# Check GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Verify installation
python -c "import torch, cv2, numpy, pandas; print('All good!')"
```

---

## 🎯 Your Next Steps

1. **Right Now:**
   - ✅ Follow installation steps above
   - ✅ Prepare your data in CSV format
   - ✅ Run `python train.py`

2. **While Training:**
   - 📖 Read [BENCHMARKS.md](BENCHMARKS.md) for what to expect
   - 🔧 Fine-tune parameters in `config.py` if needed
   - 📊 Monitor loss values (should decrease)

3. **After Training:**
   - 🧪 Test with `python test.py`
   - 📈 Check accuracy on test images
   - 📤 Deploy to production (share your weights)

4. **Share & Contribute:**
   - 🎁 Share your trained models
   - 🐛 Report issues on GitHub
   - 💡 Suggest improvements

---

## 🎓 Learning Resources

**About OCR:**
- [CRNN Paper](https://arxiv.org/abs/1507.05717)
- [Attention Mechanism](https://arxiv.org/abs/1706.03762)
- [CTC Loss Explained](https://distill.pub/2017/ctc/)

**About PyTorch:**
- [Official Tutorials](https://pytorch.org/tutorials/)
- [Beginner Guide](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

**About Deep Learning:**
- [Fast.ai Courses](https://www.fast.ai/)
- [Andrew Ng's ML Course](https://www.coursera.org/learn/machine-learning)

---

## ✨ You're All Set!

Your training environment is ready. Start with:

```bash
python train.py
```

Questions? Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or [README.md](README.md).

Good luck! 🚀

---

**Last Updated**: March 31, 2026
**Version**: 1.0.0
