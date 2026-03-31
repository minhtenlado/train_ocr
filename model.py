"""
Square CRNN Model - Advanced OCR Architecture

This module defines the deep learning architecture for Optical Character Recognition (OCR).
It implements a CRNN (Convolutional Recurrent Neural Network) with:
    - Residual blocks for deep feature extraction
    - Self-attention mechanism for capturing long-range dependencies
    - Bidirectional LSTM for sequence modeling
    - CTC loss compatible output

The model is optimized for recognizing alphanumeric characters (0-9, A-Z, -, .)
in images with various lighting conditions and rotations.

References:
    - ResNet: https://arxiv.org/abs/1512.03385
    - CRNN: https://arxiv.org/abs/1507.05717
    - Attention: https://arxiv.org/abs/1706.03762

Author: OCR Development Team
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual Block with optional dropout.
    
    Implements a residual connection (skip connection) that helps train very deep networks
    by allowing gradients to flow directly through the network.
    
    Architecture:
        Conv2d -> BatchNorm -> ReLU -> Dropout
        Conv2d -> BatchNorm
        + skip connection
        ReLU
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int, optional): Stride for the first convolution. Default: 1
        dropout_rate (float, optional): Dropout rate for regularization. Default: 0.0
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class SimpleAttention(nn.Module):
    """
    Self-Attention Mechanism for capturing feature relationships.
    
    Uses scaled dot-product attention to allow the network to focus on
    important spatial regions in the feature map. This helps identify
    crucial character features.
    
    Architecture:
        - Query, Key, Value projections reduce dimensionality
        - Compute attention weights using scaled dot-product
        - Apply attention to values
        - Residual connection with learnable weight (gamma)
    
    Args:
        channel (int): Number of input channels
    """
    def __init__(self, channel):
        super(SimpleAttention, self).__init__()
        self.query = nn.Conv2d(channel, channel // 8, 1)
        self.key = nn.Conv2d(channel, channel // 8, 1)
        self.value = nn.Conv2d(channel, channel, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()

        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)

        proj_value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x
        return out


class SquareCRNN(nn.Module):
    """
    Square CRNN - Deep Learning Model for OCR.
    
    A sophisticated Convolutional Recurrent Neural Network designed for
    optical character recognition. Combines CNN for feature extraction
    with LSTM for sequence modeling, enhanced with self-attention.
    
    Architecture Layers:
        1. Initial Conv Block: 1 → 64 channels
        2. Residual Blocks: 64 → 128 → 256 → 512 → 512
        3. Self-Attention: 512 channels
        4. Adaptive Avg Pooling: Compress height to 1
        5. Bidirectional LSTM: 512 → hidden_size (×2 layers)
        6. Classification FC: hidden_size*2 → num_classes+1 (with blank)
    
    Input: 
        Grayscale images of shape (batch, 1, height, width)
        Expected: 128×128 pixels, normalized to [-1, 1]
    
    Output:
        Character predictions of shape (sequence_length, batch, num_classes+1)
        Compatible with CTC loss for variable-length sequences
    
    Args:
        num_classes (int): Number of character classes (e.g., 36 for 0-9 + A-Z)
        hidden_size (int, optional): LSTM hidden state dimension. Default: 256
        dropout_rate (float, optional): Dropout rate for regularization. Default: 0.3
    
    Example:
        >>> model = SquareCRNN(num_classes=37)  # 36 chars + blank
        >>> x = torch.randn(32, 1, 128, 128)  # batch of 32 images
        >>> output = model(x)
        >>> output.shape  # (seq_len, batch, 37)
    """
        super(SquareCRNN, self).__init__()

        # 1. Trích xuất đặc trưng ban đầu
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2. Khối Residual
        self.layer1 = ResidualBlock(64, 128, stride=1, dropout_rate=0.1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer2 = ResidualBlock(128, 256, stride=1, dropout_rate=dropout_rate)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1))

        self.layer3 = ResidualBlock(256, 512, stride=1, dropout_rate=dropout_rate)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1))

        self.layer4 = ResidualBlock(512, 512, stride=1, dropout_rate=dropout_rate)

        # --- NÂNG CẤP: Gắn Attention trước khi ép kích thước ---
        self.attention = SimpleAttention(512)

        # 3. Ép chiều cao về 1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

        # 4. Sequence Modeling (RNN)
        self.rnn = nn.LSTM(512, hidden_size, bidirectional=True, num_layers=2, dropout=0.4) # Tăng dropout RNN để chống nhiễu chuỗi

        # 5. Phân loại
        self.fc = nn.Linear(hidden_size * 2, num_classes + 1)

        # Khởi tạo trọng số thông minh
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.maxpool2(x)

        x = self.layer2(x)
        x = self.maxpool3(x)

        x = self.layer3(x)
        x = self.maxpool4(x)

        x = self.layer4(x)

        # Gắn Attention giúp mô hình tìm ra các nét chữ quan trọng nhất
        x = self.attention(x)

        x = self.adaptive_pool(x)

        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        x = x.permute(2, 0, 1)

        x, _ = self.rnn(x)
        x = self.fc(x)
        return x