# API Documentation

## Module Reference

### `model.py`

#### Class: `ResidualBlock`

Residual block module for building deep convolutional networks.

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0)
```

**Parameters:**
- `in_channels` (int): Number of input channels
- `out_channels` (int): Number of output channels
- `stride` (int, optional): Stride for convolution. Default: 1
- `dropout_rate` (float, optional): Dropout probability. Default: 0.0

**Methods:**
- `forward(x)`: Forward pass through the residual block
  - **Input**: torch.Tensor of shape (batch, in_channels, height, width)
  - **Output**: torch.Tensor of shape (batch, out_channels, height, width)

**Example:**
```python
import torch
from model import ResidualBlock

block = ResidualBlock(64, 128, stride=1, dropout_rate=0.3)
x = torch.randn(32, 64, 128, 128)
output = block(x)
print(output.shape)  # (32, 128, 128, 128)
```

---

#### Class: `SimpleAttention`

Self-attention mechanism module.

```python
class SimpleAttention(nn.Module):
    def __init__(self, channel)
```

**Parameters:**
- `channel` (int): Number of input channels

**Methods:**
- `forward(x)`: Apply self-attention
  - **Input**: torch.Tensor of shape (batch, channels, height, width)
  - **Output**: torch.Tensor of same shape with attention applied

**Example:**
```python
from model import SimpleAttention
import torch

attention = SimpleAttention(512)
x = torch.randn(32, 512, 16, 512)  # (batch, channels, height, width)
output = attention(x)
print(output.shape)  # (32, 512, 16, 512)
```

---

#### Class: `SquareCRNN`

Main OCR model combining CNN, Self-Attention, and RNN.

```python
class SquareCRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256, dropout_rate=0.3)
```

**Parameters:**
- `num_classes` (int): Number of character classes (e.g., 36 for digits + letters)
- `hidden_size` (int, optional): LSTM hidden dimension. Default: 256
- `dropout_rate` (float, optional): Dropout rate. Default: 0.3

**Methods:**
- `forward(x)`: Forward pass through the model
  - **Input**: torch.Tensor of shape (batch, 1, 128, 128)
    - Grayscale images, normalized to [-1, 1]
  - **Output**: torch.Tensor of shape (sequence_length, batch, num_classes+1)
    - CTC loss compatible output

**Attributes:**
- `conv1`: Initial convolution layer (1 → 64)
- `layer1-4`: Residual blocks (64 → 128 → 256 → 512 → 512)
- `attention`: Self-attention module
- `adaptive_pool`: Adaptive average pooling
- `rnn`: Bidirectional LSTM
- `fc`: Fully connected classification layer

**Example:**
```python
import torch
from model import SquareCRNN

# Create model
model = SquareCRNN(num_classes=36, hidden_size=256)

# Sample input
x = torch.randn(32, 1, 128, 128)  # 32 images, grayscale, 128x128

# Forward pass
output = model(x)
print(output.shape)  # torch.Size([seq_len, 32, 37])

# Total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")  # ~2.8M
```

---

### `train.py`

#### Class: `OCRDataset`

Custom PyTorch Dataset for loading OCR training data.

```python
class OCRDataset(Dataset):
    def __init__(self, label_file, char_map, img_dir, size=128, is_train=True)
```

**Parameters:**
- `label_file` (str): Path to CSV file (format: `image.jpg,LABEL`)
- `char_map` (dict): Mapping from character to index
- `img_dir` (str): Directory containing image files
- `size` (int, optional): Target image size. Default: 128
- `is_train` (bool, optional): Apply augmentation if True. Default: True

**Attributes:**
- `valid_data` (list): List of valid (image_path, label_indices) tuples

**Methods:**
- `__len__()`: Returns number of samples
- `__getitem__(idx)`: Gets sample at index
  - **Returns**: tuple (image_tensor, target_indices, sequence_length)
- `augment_image(img)`: Apply data augmentation

**Data Validation:**
The class automatically:
- Skips missing images
- Removes empty/invalid labels
- Reports statistics

**Example:**
```python
from train import OCRDataset

char_map = {c: i+1 for i, c in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.")}
dataset = OCRDataset(
    label_file='train.csv',
    char_map=char_map,
    img_dir='images/',
    size=128,
    is_train=True
)

# Access a sample
img, targets, length = dataset[0]
print(f"Image shape: {img.shape}")      # (1, 128, 128)
print(f"Targets: {targets}")             # indices of characters
print(f"Sequence length: {length}")      # number of characters
```

---

#### Function: `collate_fn(batch)`

Collate function for DataLoader with variable-length sequences.

```python
def collate_fn(batch: list) -> tuple
```

**Parameters:**
- `batch` (list): List of samples from OCRDataset

**Returns:**
- `tuple`: (images, targets, target_lengths)
  - `images`: torch.Tensor of shape (batch_size, 1, 128, 128)
  - `targets`: torch.Tensor of concatenated character indices
  - `target_lengths`: torch.Tensor of shape (batch_size,)

**Purpose:**
Handles variable-length sequences for CTC loss by:
- Stacking images into single tensor
- Concatenating character indices with padding
- Creating target length tensor

**Example:**
```python
from torch.utils.data import DataLoader
from train import OCRDataset, collate_fn

dataset = OCRDataset('train.csv', char_map, 'images/')
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

for images, targets, target_lengths in loader:
    print(f"Batch images: {images.shape}")        # (32, 1, 128, 128)
    print(f"Targets: {targets.shape}")            # (total_chars,)
    print(f"Target lengths: {target_lengths.shape}") # (32,)
    break
```

---

#### Function: `main()`

Main training loop.

```python
def main() -> None
```

**Workflow:**
1. Initialize model, optimizer, criterion, and scheduler
2. Load training dataset
3. Create DataLoader with custom collate function
4. Train for specified number of epochs
5. Save best model based on loss

**Key Features:**
- CTC Loss for variable-length recognition
- Gradient clipping for stable training
- ReduceLROnPlateau scheduler
- Model checkpointing

**Configuration:**
Edit these parameters in the function:
- `chars`: Character set (default: 0-9, A-Z, -, .)
- `batch_size`: Batch size (default: 32)
- `num_epochs`: Training epochs (default: 150)
- `learning_rate`: Initial learning rate (default: 0.001)
- `csv_path`: Path to training CSV
- `img_dir`: Path to image directory

---

### `config.py`

#### Function: `get_config(section=None)`

Retrieve configuration dictionary.

```python
def get_config(section: str = None) -> dict
```

**Parameters:**
- `section` (str, optional): Configuration section to retrieve
  - Sections: 'dataset', 'model', 'training', 'optimizer', 'scheduler', etc.
  - If None, returns all configurations

**Returns:**
- Configuration dictionary

**Example:**
```python
from config import get_config

# Get all configurations
all_config = get_config()

# Get specific section
model_config = get_config('model')
print(model_config['num_classes'])  # 36

training_config = get_config('training')
print(training_config['batch_size'])  # 32
```

---

## Complete Training Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SquareCRNN
from train import OCRDataset, collate_fn

# Configuration
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-."
CHAR_MAP = {c: i + 1 for i, c in enumerate(CHARS)}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model
model = SquareCRNN(num_classes=len(CHARS)).to(DEVICE)

# Loss function
criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(DEVICE)

# Create dataset and dataloader
dataset = OCRDataset(
    label_file='train.csv',
    char_map=CHAR_MAP,
    img_dir='images/',
    size=128,
    is_train=True
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2
)

# Training step
for images, targets, target_lengths in loader:
    images = images.to(DEVICE)
    targets = targets.to(DEVICE)
    target_lengths = target_lengths.to(DEVICE)
    
    # Forward pass
    predictions = model(images)
    
    # Compute input lengths for CTC
    input_lengths = torch.full(
        (images.size(0),),
        predictions.size(0),
        dtype=torch.long,
        device=DEVICE
    )
    
    # Compute loss
    loss = criterion(
        predictions.log_softmax(2),
        targets,
        input_lengths,
        target_lengths
    )
    
    print(f"Loss: {loss.item():.4f}")
```

---

## Input/Output Specifications

### Model Input
- **Shape**: (batch_size, 1, 128, 128)
- **Type**: torch.Tensor (float32)
- **Range**: [-1, 1] (normalized)
- **Requirement**: Grayscale images, any resolution resized to 128×128

### Model Output
- **Shape**: (sequence_length, batch_size, num_classes + 1)
  - sequence_length: Variable (depends on image spatial dimensions after CNN)
  - batch_size: Input batch size
  - num_classes + 1: Number of classes + blank token (for CTC)
- **Type**: torch.Tensor (float32)
- **Usage**: Compatible with nn.CTCLoss

### For Inference
```python
model.eval()
with torch.no_grad():
    output = model(images)
    # output shape: (seq_len, batch, num_classes+1)
    predictions = output.argmax(2)  # (seq_len, batch)
```

---

## Error Handling

### Common Issues and Solutions

**1. Shape Mismatch in CTC Loss**
```
Error: input length >= target_length
```
Solution: Ensure images are at least 2×8 pixels after CNN feature extraction.

**2. Out of Memory**
```
Error: CUDA out of memory
```
Solution: Reduce batch_size (32 → 16) or image size (128 → 64)

**3. Dataset Load Error**
```
Error: [Errno 2] No such file or directory
```
Solution: Check CSV path and image directory exist and are accessible

---

**Last Updated**: 2026
**Version**: 1.0.0
