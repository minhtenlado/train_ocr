# Troubleshooting Guide

## Installation & Setup Issues

### Issue: Python Version Mismatch
**Symptom**: `pip install -r requirements.txt` fails or shows Python version error

**Solutions**:
```bash
# Check Python version
python --version

# If wrong version, use specific Python
python3.10 -m venv venv

# Or upgrade Python via:
# Windows: Download from python.org
# macOS: brew install python@3.11
# Linux: sudo apt-get install python3.11
```

---

### Issue: CUDA/GPU Not Detected
**Symptom**: `torch.cuda.is_available()` returns False, or NVIDIA driver warnings

**Solutions**:
1. **Verify NVIDIA Driver**
   ```bash
   nvidia-smi  # Should display GPU information
   ```
   If command not found:
   - Windows: Download from [nvidia.com](https://www.nvidia.com/Download/driverDetails.aspx)
   - Linux: `sudo apt-get install nvidia-driver-530`

2. **Reinstall PyTorch with correct CUDA**
   ```bash
   # First, identify your CUDA version
   nvidia-smi  # Look for "CUDA Capability"
   
   # Install correct PyTorch version
   pip uninstall torch torchvision torchaudio -y
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Verify Installation**
   ```bash
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
   ```

---

### Issue: Package Installation Fails
**Symptom**: `ERROR: Could not find a version that satisfies the requirement`

**Solutions**:
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install with no cache
pip install --no-cache-dir -r requirements.txt

# Install packages individually for debugging
pip install torch
pip install torchvision
pip install opencv-python
# ... etc
```

---

## Dataset & Data Issues

### Issue: "No valid training samples found"
**Symptom**: Model exits immediately with this error message

**Causes & Solutions**:
1. **CSV file not found**
   ```bash
   # Check file exists
   ls -la train.csv  # or dir train.csv on Windows
   
   # Fix in train.py:
   csv_path = '/absolute/path/to/train.csv'  # Use full path
   ```

2. **Images directory incorrect**
   ```python
   # In train.py, verify paths match:
   csv_path = '/path/to/train.csv'
   img_dir = '/path/to/images'  # Must match directory structure
   ```

3. **CSV format incorrect**
   ```bash
   # Check CSV format (should be: filename.jpg,LABEL)
   head train.csv
   
   # Fix encoding issues:
   file train.csv  # Check encoding (should be UTF-8)
   ```

4. **Image files missing**
   ```bash
   # Verify images exist in the directory
   ls images/ | wc -l  # Count images
   
   # Check if filenames in CSV match actual files
   head train.csv | cut -d',' -f1 | while read f; do [ ! -f "images/$f" ] && echo "Missing: $f"; done
   ```

---

### Issue: "Error: column index out of range"
**Symptom**: Crash during data loading with column index error

**Solutions**:
1. **Fix CSV format**
   ```bash
   # CSV should have exactly 2 columns: filename,label
   # NO extra commas or spaces
   
   # Check format:
   head -5 train.csv
   
   # Fix using Python:
   import pandas as pd
   df = pd.read_csv('train.csv', names=['filename', 'label'])
   df.to_csv('train_fixed.csv', index=False, header=False)
   ```

2. **Handle multi-line labels**
   ```python
   # If labels contain commas, update the parsing:
   # In train.py, line: parts = line.strip().split(',')
   # Change to:
   parts = line.strip().split(',', 1)  # Split only first comma
   ```

---

### Issue: Image file encoding or reading errors
**Symptom**: "cv2.imread() returned None" or similar warnings

**Solutions**:
1. **Check image files**
   ```bash
   # Verify images are readable
   file images/*.jpg  # Shows file type
   
   # Check file sizes (empty files?)
   ls -lh images/*.jpg | grep "0 B"
   ```

2. **Fix corrupted images**
   ```python
   # Verify image integrity with Python
   import cv2
   for img_file in os.listdir('images/'):
       img = cv2.imread(f'images/{img_file}', cv2.IMREAD_GRAYSCALE)
       if img is None:
           print(f"Corrupted: {img_file}")
           # Remove from CSV
   ```

3. **Image format issues**
   ```bash
   # Convert to standard format if needed
   for f in images/*.bmp; do
       convert "$f" "${f%.bmp}.jpg"
   done
   ```

---

## Training Issues

### Issue: Out of Memory (OOM)
**Symptom**: `RuntimeError: CUDA out of memory` or similar

**Solutions** (in order of least to most drastic):
1. **Reduce batch size**
   ```python
   # In train.py, main():
   batch_size = 16  # Reduce from 32
   loader = DataLoader(dataset, batch_size=batch_size, ...)
   ```

2. **Reduce image size**
   ```python
   # In train.py, OCRDataset():
   size = 64  # Reduce from 128
   # Note: This will reduce accuracy by ~5%
   ```

3. **Reduce hidden size**
   ```python
   # In train.py, main():
   model = SquareCRNN(len(chars), hidden_size=128)  # Reduce from 256
   ```

4. **Use CPU** (slow but works)
   ```python
   # In train.py, main():
   device = torch.device('cpu')  # Force CPU
   ```

5. **Clear GPU cache**
   ```python
   # Add to train loop:
   torch.cuda.empty_cache()
   ```

---

### Issue: High training loss (not decreasing)
**Symptom**: Loss stays high (> 3.0) or doesn't decrease

**Causes & Solutions**:

1. **Dataset quality issues**
   - Check images have good contrast
   - Verify labels are correct
   - Remove blurry/unreadable images
   ```bash
   # Manually inspect sample images
   # Verify they contain clear, readable text
   ```

2. **Character set mismatch**
   ```python
   # Ensure characters in CSV match char_map:
   chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-."
   # Check if your data contains other characters?
   ```

3. **Learning rate too high/low**
   ```python
   # In config.py:
   TRAINING_CONFIG = {
       'learning_rate': 0.0001,  # Try this if too high
   }
   # If very slow, try 0.01
   ```

4. **Model not converging**
   - Train for more epochs: 150 → 200
   - Check if data is actually loading (print sample)
   - Verify loss is computed correctly

---

### Issue: Training very slow
**Symptom**: Each epoch takes hours

**Solutions**:

1. **Using CPU instead of GPU**
   ```python
   print(f"Device: {device}")  # Check output
   # If 'cpu', see Issue: CUDA/GPU Not Detected above
   ```

2. **Data loading bottleneck**
   ```python
   # In DataLoader:
   loader = DataLoader(
       dataset,
       batch_size=32,
       num_workers=4,  # Increase from 2
       pin_memory=True,  # Add this
   )
   ```

3. **Image size too large**
   ```python
   # Reduce from 128×128 to 64×64:
   size = 64
   ```

4. **Profile to find bottleneck**
   ```python
   import time
   
   start = time.time()
   for images, targets, target_lengths in loader:
       pass  # Just loading, no training
   print(f"Data loading time: {time.time() - start:.2f}s")
   ```

---

### Issue: Model overfitting (loss decreases but need better generalization)
**Symptom**: Training loss very low but test accuracy poor

**Solutions**:

1. **Increase dropout**
   ```python
   # In model.py or train.py:
   model = SquareCRNN(num_classes, dropout_rate=0.5)  # Increase from 0.3
   ```

2. **More data augmentation**
   ```python
   # In train.py, augment_image():
   # Increase probability or strength of augmentation
   ```

3. **More training data**
   - Collect more samples
   - Use data augmentation
   - Synthetic data generation

4. **Reduce model size**
   ```python
   # Smaller model = less overfitting
   model = SquareCRNN(num_classes, hidden_size=128)  # Reduce from 256
   ```

---

## Model Inference Issues

### Issue: Low accuracy on test images
**Symptom**: Model predicts wrong characters

**Solutions**:

1. **Verify preprocessing matches training**
   ```python
   # Ensure test images are:
   # - Grayscale or RGB (converted to grayscale)
   # - Resized to 128×128
   # - Normalized to [-1, 1]
   # - Have channel dimension (1, H, W)
   ```

2. **Check model is trained enough**
   ```python
   # Verify model was trained for enough epochs
   # Loss should have plateaued
   # Try with best_square_ocr_pro.pth (best checkpoint)
   ```

3. **Image quality too different from training**
   - If training images are clear, test on clear images
   - If training images have noise, add similar noise to test

4. **Character set mismatch**
   ```python
   # Verify test images only contain:
   chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-."
   ```

---

### Issue: CTC Decode Error
**Symptom**: `decode()` method not working or returning empty

**Solutions**:

1. **Use greedy decoding** (simplest)
   ```python
   # Get predictions
   output = model(image)
   
   # Greedy decode (take argmax per timestep)
   predictions = output.argmax(2)  # (seq_len, batch)
   
   # Convert indices to characters
   decoded = [chars[idx-1] for idx in predictions[:, 0] if idx > 0]
   text = ''.join(decoded)
   ```

2. **Implement CTC beam search**
   ```python
   # Use specialized CTC decode function
   # or install: pip install pyctcdecode
   from pyctcdecode import build_ctcdecoder
   ```

---

## Performance Issues

### Issue: Model size too large for deployment
**Symptom**: 11.2 MB model is too big for edge devices

**Solutions**:

1. **Model quantization** (INT8)
   ```python
   import torch.quantization as quantization
   
   # Quantize model
   quantized_model = torch.quantization.quantize_dynamic(
       model, {torch.nn.LSTM}, dtype=torch.qint8
   )
   torch.save(quantized_model.state_dict(), 'model_quantized.pth')
   # Result: ~2.8 MB, 20% faster
   ```

2. **Knowledge distillation**
   - Train smaller student model with teacher supervision
   - Results in much smaller model

3. **Pruning**
   - Remove unused weights
   - 30-50% size reduction with minimal accuracy loss

---

## Git & Version Control Issues

### Issue: Model weights .pth files not committing
**Symptom**: `*.pth` files ignored by git

**This is intentional** - large binary files shouldn't be in git.

**Solutions**:
1. **Store weights in cloud**
   - Google Drive
   - AWS S3
   - GitHub Releases

2. **Use Git LFS** (Large File Storage)
   ```bash
   git lfs install
   git lfs track "*.pth"
   git add .gitattributes
   git commit -m "Track pytorch models with LFS"
   ```

---

## General Tips

### Debug Mode - Print Everything
Add debug prints to understand flow:

```python
# In train.py, main():
print(f"Dataset size: {len(dataset)}")
print(f"First sample shape: {dataset[0][0].shape}")
print(f"Device: {device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# In training loop:
if epoch % 10 == 0:
    with torch.no_grad():
        test_img, _, _ = dataset[0]
        test_out = model(test_img.unsqueeze(0).to(device))
        print(f"Model output shape: {test_out.shape}")
```

### Save Training Logs
```python
# Add logging to file
import logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# In training loop:
logging.info(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")
```

### Validate Data Integrity
```python
# Create a data validation script
for i in range(len(dataset)):
    img, target, length = dataset[i]
    assert img.shape == (1, 128, 128), f"Sample {i}: Wrong shape"
    assert len(target) == length, f"Sample {i}: Length mismatch"
```

---

## Still Having Issues?

1. **Check documentation**
   - [README.md](README.md) - Overview
   - [SETUP.md](SETUP.md) - Installation
   - [DATASET.md](DATASET.md) - Data format
   - [API.md](API.md) - API reference

2. **Search existing issues**
   - GitHub Issues section
   - Stack Overflow with tags: `pytorch`, `ocr`, `crnn`

3. **Create a detailed issue**
   - Include: Python version, PyTorch version, GPU info
   - Provide: Error message, code snippet, data sample
   - Describe: What you tried, expected behavior

4. **Contact support**
   - GitHub Discussions
   - Project email

---

**Last Updated**: March 31, 2026
**Version**: 1.0.0
