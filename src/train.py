"""
Training Script for Square CRNN OCR Model.

This module handles:
    - Data loading and preprocessing
    - Model training with CTC loss
    - Learning rate scheduling
    - Model checkpointing
    - Data augmentation

The script uses PyTorch and OpenCV for image processing.
It supports both CPU and GPU training with automatic device detection.

Usage:
    python train.py
    
    Modify the main() function to adjust:
    - Dataset paths
    - Model architecture parameters
    - Training hyperparameters
    - Batch size and number of epochs

Author: OCR Development Team
License: MIT
"""

import os
import sys
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Ensure package imports work regardless of working directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

try:
    from src.model import SquareCRNN
    from src.config import CHARACTER_SET
except ImportError:
    from model import SquareCRNN
    from config import CHARACTER_SET


class OCRDataset(Dataset):
    """
    Custom Dataset for OCR Training.
    
    Loads images and their corresponding text labels from a CSV file.
    Performs intelligent data validation and augmentation.
    
    CSV Format: image_filename.jpg,TEXT_CONTENT
    
    Features:
        - Automatic data cleaning (removes missing/invalid samples)
        - Smart data augmentation (brightness, rotation, noise)
        - Grayscale image resizing
        - Normalization to [-1, 1] range
        - Training/validation mode support
    
    Args:
        label_file (str): Path to CSV file with image names and labels
        char_map (dict): Mapping from character to index
        img_dir (str): Directory containing image files
        size (int, optional): Target image size (default: 128)
        is_train (bool, optional): Whether to apply augmentation (default: True)
    
    Attributes:
        valid_data (list): List of (image_path, label_indices) tuples
    """
    def __init__(self, label_file, char_map, img_dir, size=128, is_train=True):
        self.size = size
        self.img_dir = img_dir
        self.char_map = char_map
        self.is_train = is_train
        self.valid_data = []

        print("Đang rà soát và làm sạch dữ liệu...")
        missing_imgs, invalid_labels = 0, 0

        with open(label_file, 'r', encoding='utf-8-sig') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    img_name = parts[0]
                    raw_label = ','.join(parts[1:])
                    img_path = os.path.join(self.img_dir, img_name)

                    if not os.path.exists(img_path):
                        missing_imgs += 1
                        continue

                    clean_label = [self.char_map[c] for c in raw_label.upper() if c in self.char_map]

                    if len(clean_label) == 0:
                        invalid_labels += 1
                        continue

                    self.valid_data.append((img_path, clean_label))

        print(f"-> Loại bỏ: {missing_imgs} ảnh rỗng, {invalid_labels} nhãn sai.")
        print(f"-> Sẵn sàng huấn luyện: {len(self.valid_data)} mẫu.")

    def augment_image(self, img):
        """
        Apply intelligent data augmentation to image.
        
        Mimics real-world scenarios of poor image quality:
        - Brightness/contrast variation (simulates lighting conditions)
        - Slight rotation (simulates tilted camera)
        - Gaussian noise (simulates low-quality camera/compression)
        
        Args:
            img (np.ndarray): Input grayscale image
            
        Returns:
            np.ndarray: Augmented image
        """
        alpha = random.uniform(0.7, 1.3)
        beta = random.randint(-30, 30)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        if random.random() > 0.5:
            rows, cols = img.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), random.uniform(-5, 5), 1)
            img = cv2.warpAffine(img, M, (cols, rows), borderValue=(0))

        if random.random() > 0.7:
            gauss = np.random.normal(0, 15, img.size).reshape(img.shape).astype('uint8')
            img = cv2.add(img, gauss)

        return img

    def __len__(self):
        """Get total number of valid samples in dataset."""
        return len(self.valid_data)

    def __getitem__(self, idx):
        """Load and preprocess image and label at given index."""
        img_path, target = self.valid_data[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            img = np.zeros((self.size, self.size), dtype=np.uint8)
            target = [1]

        if self.is_train:
            img = self.augment_image(img)

        img = cv2.resize(img, (self.size, self.size))
        img = (img.astype(np.float32) / 127.5) - 1.0
        img = torch.from_numpy(img).unsqueeze(0)

        return img, torch.tensor(target, dtype=torch.long), len(target)


def collate_fn(batch):
    """
    Collate function for DataLoader to handle variable-length sequences.
    
    Args:
        batch (list): List of (image, target, length) tuples from OCRDataset
        
    Returns:
        tuple: (images, targets, target_lengths)
    """
    imgs, targets, target_lengths = zip(*batch)
    imgs = torch.stack(imgs)
    targets = torch.cat(targets)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    return imgs, targets, target_lengths


def main():
    """
    Main training loop for Square CRNN OCR model.
    """
    chars = CHARACTER_SET
    char_map = {c: i + 1 for i, c in enumerate(chars)}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SquareCRNN(len(chars)).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5, min_lr=1e-6)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(device)

    # Resolve CSV and images paths flexibly
    csv_path = os.path.join(base_dir, 'train.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(base_dir, 'data', 'train.csv')

    img_dir = os.path.join(base_dir, 'images')
    if not os.path.exists(img_dir):
        img_dir = os.path.join(base_dir, 'data', 'dataset')

    if not os.path.exists(csv_path):
        print(f"ERROR: CSV File not found at {csv_path}. Please check data path.")
        return

    print(f"[+] Loading dataset from CSV: {csv_path}")
    print(f"[+] Image directory: {img_dir}")

    dataset = OCRDataset(csv_path, char_map, img_dir, is_train=True)

    if len(dataset) == 0:
        print("ERROR: No valid training samples found. Check data files and paths.")
        return

    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=2)

    print(f"\n--- Starting training on {device} ---")
    print(f"Training samples: {len(dataset)}")

    epochs = 150
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for imgs, targets, target_lengths in loader:
            imgs, targets, target_lengths = imgs.to(device), targets.to(device), target_lengths.to(device)

            optimizer.zero_grad()
            preds = model(imgs)

            input_lengths = torch.full(size=(imgs.size(0),), fill_value=preds.size(0), dtype=torch.long, device=device)

            loss = criterion(preds.log_softmax(2), targets, input_lengths, target_lengths)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)

        print(f"Epoch [{epoch:3d}/{epochs}] - Loss: {avg_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            models_dir = os.path.join(base_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            save_path = os.path.join(models_dir, 'best_square_ocr_pro.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Model saved: {save_path} (Best Loss: {best_loss:.4f})")


if __name__ == "__main__":
    main()