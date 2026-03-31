# Dataset Format and Preparation Guide

## CSV Format

The training dataset should be organized as a CSV file with the following format:

```csv
image_filename.jpg,TEXT_CONTENT
image_filename.png,ANOTHER_TEXT
```

### Example
```csv
ticket_001.jpg,ABC123-45
receipt_002.png,INVOICE-2024
label_003.jpg,PRODUCT.CODE
```

### Requirements

1. **Image Filenames**
   - Column 1: Exact filename (with extension)
   - Extensions supported: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
   - Case-sensitive on Linux/macOS

2. **Text Labels**
   - Column 2+: Text content to recognize
   - Supported characters: `0-9 A-Z - .` (default character set)
   - Case will be converted to uppercase
   - Invalid characters are automatically filtered

3. **File Structure**
```
project_root/
├── train.csv           # CSV with image names and labels
├── images/
│   ├── image_001.jpg
│   ├── image_002.png
│   └── image_003.jpg
└── dataset/            # Alternative directory name
    ├── train_001.jpg
    └── train_002.jpg
```

## Data Validation

The system automatically validates data during loading:

- **Skips missing images**: Images referenced in CSV but not found are skipped
- **Removes empty labels**: Samples with no valid characters are discarded
- **Logs statistics**: Reports invalid images and labels during training setup

Example output:
```
Đang rà soát và làm sạch dữ liệu...
-> Loại bỏ: 5 ảnh rỗng, 3 nhãn sai.
-> Sẵn sàng huấn luyện: 997 mẫu.
```

Translation:
```
Scanning and cleaning data...
-> Removed: 5 missing images, 3 invalid labels.
-> Ready to train: 997 samples.
```

## Image Requirements

### Ideal Format
- **Size**: 128×128 pixels (automatically resized)
  - Minimum: 32×32 (not recommended)
  - Maximum: 512×512 (may cause memory issues)
- **Color**: Grayscale or RGB (converted to grayscale)
- **Format**: Clear with good contrast between text and background
- **Quality**: Acceptable with some noise and blur

### Preprocessing
- Images are automatically:
  - Converted to grayscale (if needed)
  - Resized to 128×128 pixels
  - Normalized to [-1, 1] range
  - Augmented during training (rotation, brightness, noise)

### Not Recommended
❌ Very small text (< 10 pixels height)
❌ Heavily blurred images
❌ Images with multiple orientations
❌ Low contrast (text blends with background)

## Character Set

### Default Characters
```
0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.
```

- **Digits**: 0-9 (10 characters)
- **Letters**: A-Z (26 characters)  
- **Symbols**: `-` (dash), `.` (period)
- **Total**: 36 characters + 1 blank token = 37 classes

### Custom Character Set
To use a different character set:

```python
# In train.py, modify the characters string:
chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-."
```

Or use the config file:
```python
from config import CONFIGURATION
chars = CONFIGURATION['character_set']
```

## Data Augmentation

During training, images are automatically augmented to improve robustness:

1. **Brightness Adjustment**
   - Multiplier: 0.7 to 1.3
   - Bias: -30 to +30
   - Purpose: Simulates varying lighting conditions

2. **Rotation**
   - Angle: ±5 degrees
   - Probability: 50%
   - Purpose: Simulates tilted camera/installation angle
   - Padding: Black (value=0)

3. **Gaussian Noise**
   - Standard deviation: 15
   - Probability: 30%
   - Purpose: Simulates low-quality camera or compression artifacts

## Data Statistics

### Recommended Dataset Sizes

| Dataset Size | Recommendation |
|--------------|-----------------|
| < 100 | Too small, high overfitting risk |
| 100-500 | Minimum viable, use heavy augmentation |
| 500-2000 | Good, train 100+ epochs |
| 2000-10000 | Excellent, train 50-100 epochs |
| > 10000 | Very good, train 20-50 epochs |

### Train/Validation Split
By default, the model uses the entire dataset for training.
To create a validation split, modify `train.py`:

```python
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)
```

## Preparing Your Data

### From Images in a Folder

Use the `name_img.py` script (if available) or follow this process:

1. **Create CSV manually**
   ```bash
   # Linux/macOS
   ls images/ | awk '{print $1",YOUR_LABEL"}' > train.csv
   ```

2. **Or use Python**
   ```python
   import os
   import pandas as pd
   
   image_dir = 'images/'
   images = os.listdir(image_dir)
   
   # Manually add labels or use OCR/manual process
   labels = ['LABEL1', 'LABEL2', ...]  # Your labels
   
   df = pd.DataFrame({'image': images, 'label': labels})
   df.to_csv('train.csv', index=False, header=False)
   ```

3. **Or use data_preprocessing.py** (if provided)
   ```python
   python data.py
   ```

## Troubleshooting

### No valid training samples found
**Cause**: CSV path or image directory incorrect, or all labels are invalid
**Solution**: 
- Check file paths match exactly
- Verify images exist in the directory
- Ensure labels contain valid characters from your character set

### High number of removed samples
**Cause**: Many images missing or invalid labels
**Solution**:
- Double-check image filenames in CSV
- Verify valid character set coverage
- Visually inspect images for readability

### Poor training accuracy
**Cause**: Dataset quality or size issues
**Solution**:
- Increase dataset size (aim for 1000+ samples)
- Ensure images have good contrast
- Check character set matches your data
- Increase training epochs

## Performance Tips

1. **Image Quality**
   - Use high-contrast images (dark text on light background preferred)
   - Ensure consistent lighting
   - Avoid motion blur

2. **Dataset Variety**
   - Include different fonts/styles
   - Vary brightness and contrast
   - Include different image sources

3. **Labeling Accuracy**
   - Verify each label is correct
   - Use uppercase only
   - Double-check special characters (-, .)

4. **Hardware Optimization**
   - Use GPU for faster training (NVIDIA CUDA recommended)
   - Batch size 32-64 for GPU, 8-16 for CPU
   - Use num_workers=4-8 for faster data loading
