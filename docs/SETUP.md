# Installation and Setup Guide

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 8 GB
- **Disk**: 20 GB available

### Recommended Requirements
- **OS**: Windows 11, macOS 12+, or Ubuntu 20.04+
- **Python**: 3.10 or 3.11
- **GPU**: NVIDIA GPU with CUDA Compute Capability 7.0+ (RTX series recommended)
- **RAM**: 16-32 GB
- **Disk**: 50+ GB SSD (for training data)

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/square-crnn-ocr.git
cd square-crnn-ocr
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Verify activation (should show `(venv)` in terminal):
```bash
python --version  # Should match or exceed 3.8
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

**Common Installation Issues:**

If you encounter issues with PyTorch installation:

**CPU Only (Smaller download, slower training):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**GPU (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**GPU (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Visit [PyTorch.org](https://pytorch.org/get-started/locally/) for the latest CUDA version matching your setup.

### 5. Verify Installation

```bash
python -c "import torch; print(torch.__version__)"
python -c "import cv2; print(cv2.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Expected output:
```
2.0.0+cu118  # or similar version
4.8.0  # or similar OpenCV version
CUDA available: True  # or False if CPU only
```

---

## Configuration

### 1. Configure Dataset Paths

**Option A: Edit `train.py` directly**
```python
# In train.py, main() function
drive_dir = '/path/to/your/data'  # Change this
csv_path = f'{drive_dir}/train.csv'
img_dir = f'{drive_dir}/images'
```

**Option B: Use `config.py` (Recommended)**
```python
# In config.py, DATASET_CONFIG
DATASET_CONFIG = {
    'csv_path': '/your/path/train.csv',
    'image_dir': '/your/path/images',
    ...
}
```

### 2. Configure Training Parameters

Edit `config.py` sections:

```python
# For batch size, epochs, etc.
TRAINING_CONFIG = {
    'batch_size': 32,
    'num_epochs': 150,
    'learning_rate': 0.001,
    ...
}

# For model architecture
MODEL_CONFIG = {
    'num_classes': 36,
    'hidden_size': 256,
    ...
}
```

---

## NVIDIA GPU Setup (Optional but Recommended)

### Windows GPU Setup

1. **Install NVIDIA Driver**
   - Download from [nvidia.com](https://www.nvidia.com/Download/driverDetails.aspx)
   - Choose your GPU model and Windows version
   - Install and restart computer

2. **Verify GPU Availability**
   ```bash
   nvidia-smi  # Should display GPU info
   ```

3. **Install CUDA Toolkit** (if not automatic with driver)
   - Download from [nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
   - Match the CUDA version with PyTorch installation

4. **Test CUDA in Python**
   ```bash
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
   ```

### Linux GPU Setup

```bash
# Ubuntu 20.04 with NVIDIA GPU
sudo apt-get update
sudo apt-get install nvidia-driver-530  # Adjust version as needed
sudo apt-get install nvidia-cuda-toolkit

# Verify
nvidia-smi
```

### Troubleshooting GPU Issues

**CUDA not found:**
```bash
# Check if GPU-enabled PyTorch is installed
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall with correct CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
```

**Driver conflicts:**
```bash
# If issues persist, use CPU version temporarily
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --force-reinstall
```

---

## Running the Project

### 1. Prepare Your Data

Create a CSV file with format:
```csv
image_001.jpg,LABEL1
image_002.jpg,LABEL2
```

Place images in the `images/` directory.

### 2. Start Training

```bash
# Activate virtual environment (if not already activated)
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# Run training
python train.py
```

Expected output:
```
Scanning and cleaning data...
-> Removed: 0 missing images, 0 invalid labels.
-> Ready to train: 1000 samples.

--- Starting training on cuda ---
Training samples: 1000

Epoch [  1/150] - Loss: 4.5321 - LR: 0.001000
Epoch [  2/150] - Loss: 3.2145 - LR: 0.001000
...
```

### 3. Use Jupyter Notebook (Optional)

```bash
jupyter notebook train.ipynb
```

Or use VS Code with Jupyter extension.

### 4. Test Inference

Modify `test.py` with your model path and run:
```bash
python test.py
```

---

## Project Structure After Setup

```
square-crnn-ocr/
├── venv/                          # Virtual environment (gitignored)
├── model.py                       # Model architecture
├── train.py                       # Training script
├── test.py                        # Inference script
├── data.py                        # Data utilities
├── config.py                      # Configuration
├── train.ipynb                    # Jupyter notebook
├── 
├── train.csv                      # Training annotations (gitignored)
├── dataset/
│   └── images/                    # Training images (gitignored)
├── images/                        # Sample images (gitignored)
├── best_square_ocr_pro.pth       # Trained weights (gitignored)
├── __pycache__/                   # Python cache (gitignored)
├──
├── README.md                      # Main documentation
├── DATASET.md                     # Dataset guide
├── BENCHMARKS.md                  # Performance metrics
├── API.md                         # API documentation
├── CONTRIBUTING.md                # Contributing guide
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
├── LICENSE                        # MIT License
└── .github/
    └── ISSUE_TEMPLATE/
        └── bug_report.md          # Bug report template
```

---

## Troubleshooting Setup

### Python Version Issues

```bash
# Check Python version
python --version

# If wrong version, specify explicitly
python3.10 -m venv venv  # Create with Python 3.10
```

### Package Installation Fails

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Try with specific versions
pip install -r requirements.txt --verbose

# Check for conflicting packages
pip check
```

### Virtual Environment Not Activating

**Windows:**
```bash
# If script is disabled, run PowerShell as admin and execute:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then activate normally
venv\Scripts\activate
```

**Linux/macOS:**
```bash
# Make sure script is executable
chmod +x venv/bin/activate

# Activate with bash explicitly
bash venv/bin/activate
```

### CUDA/GPU Not Detected

```bash
# Verify NVIDIA driver
nvidia-smi

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"

# If False, reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Next Steps

After installation:

1. **Read the documentation**
   - Start with [README.md](README.md)
   - Check [DATASET.md](DATASET.md) to prepare your data
   - See [BENCHMARKS.md](BENCHMARKS.md) for performance expectations

2. **Prepare your dataset**
   - Create `train.csv` with image names and labels
   - Place images in `images/` folder

3. **Start training**
   ```bash
   python train.py
   ```

4. **Monitor training**
   - Check loss values (should decrease)
   - Loss plateau indicates training stabilization
   - Check GPU usage with `nvidia-smi` (in another terminal)

5. **Evaluate and deploy**
   - Use `test.py` for inference
   - Save best model weights
   - Deploy to production environment

---

## Uninstall

To remove the project and virtual environment:

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
# Windows:
rmdir /s venv

# Linux/macOS:
rm -rf venv

# Remove project folder
cd ..
rm -rf square-crnn-ocr
```

---

**For more help, see:**
- [API Documentation](API.md)
- [Dataset Guide](DATASET.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Contributing Guide](CONTRIBUTING.md)

Last Updated: 2026
