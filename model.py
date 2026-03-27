import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Thêm Dropout2d để chống overfitting, giúp mô hình bắt đặc trưng cục bộ tốt hơn
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

class SquareCRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256, dropout_rate=0.2):
        super(SquareCRNN, self).__init__()
        
        # 1. Lớp trích xuất đặc trưng đầu vào
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 128x128 -> 64x64
        
        # 2. Các khối Residual kết hợp MaxPool (giảm chiều H nhanh hơn chiều W)
        self.layer1 = ResidualBlock(64, 128, stride=1, dropout_rate=0.1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 64x64 -> 32x32
        
        self.layer2 = ResidualBlock(128, 256, stride=1, dropout_rate=dropout_rate)
        # Giảm chiều cao đi 2 lần, giữ nguyên chiều rộng (Sequence Length)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)) # 32x32 -> 16x32
        
        self.layer3 = ResidualBlock(256, 512, stride=1, dropout_rate=dropout_rate)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)) # 16x32 -> 8x32
        
        self.layer4 = ResidualBlock(512, 512, stride=1, dropout_rate=dropout_rate)
        
        # 3. Adaptive Pooling: Chìa khóa để giảm tải LSTM
        # Ép chiều cao (Height) về đúng 1 pixel, giữ nguyên chiều rộng (Width)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None)) 
        
        # 4. Sequence Modeling (RNN)
        # Input size giờ chỉ còn 512 (thay vì 512 * 8 như trước)
        self.rnn = nn.LSTM(512, hidden_size, bidirectional=True, num_layers=2, dropout=0.3)

        # 5. Phân loại
        self.fc = nn.Linear(hidden_size * 2, num_classes + 1)

    def forward(self, x):
        # Đi qua CNN
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        
        x = self.layer1(x)
        x = self.maxpool2(x)
        
        x = self.layer2(x)
        x = self.maxpool3(x)
        
        x = self.layer3(x)
        x = self.maxpool4(x)
        
        x = self.layer4(x)
        
        # Pooling ép chiều cao về 1
        x = self.adaptive_pool(x) # Output: [B, 512, 1, W]
        
        # Chuẩn bị dữ liệu cho RNN
        b, c, h, w = x.size()
        x = x.view(b, c * h, w) # c*h lúc này chính xác là 512 * 1 = 512
        x = x.permute(2, 0, 1)  # Đảo chiều thành [W, B, C] cho chuẩn đầu vào LSTM
        
        # Đi qua RNN
        x, _ = self.rnn(x)
        
        # Đưa ra dự đoán
        x = self.fc(x)
        return x