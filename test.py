import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
from ultralytics import YOLO

# ==========================================
# 1. ĐỊNH NGHĨA LẠI MÔ HÌNH OCR (BẢN PRO CÓ ATTENTION)
# ==========================================
class ResidualBlock(nn.Module):
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
    def __init__(self, num_classes, hidden_size=256, dropout_rate=0.3):
        super(SquareCRNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.layer1 = ResidualBlock(64, 128, stride=1, dropout_rate=0.1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.layer2 = ResidualBlock(128, 256, stride=1, dropout_rate=dropout_rate)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)) 
        self.layer3 = ResidualBlock(256, 512, stride=1, dropout_rate=dropout_rate)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)) 
        self.layer4 = ResidualBlock(512, 512, stride=1, dropout_rate=dropout_rate)
        self.attention = SimpleAttention(512)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None)) 
        self.rnn = nn.LSTM(512, hidden_size, bidirectional=True, num_layers=2, dropout=0.4)
        self.fc = nn.Linear(hidden_size * 2, num_classes + 1)

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
        x = self.attention(x)
        x = self.adaptive_pool(x) 
        b, c, h, w = x.size()
        x = x.view(b, c * h, w) 
        x = x.permute(2, 0, 1)  
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# ==========================================
# 2. HÀM GIẢI MÃ OCR
# ==========================================
def decode_predictions(preds, idx_to_char):
    _, preds = preds.max(2) 
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds = preds.cpu().numpy()
    decoded = []
    for i in range(len(preds)):
        if preds[i] != 0 and (not (i > 0 and preds[i - 1] == preds[i])):
            decoded.append(idx_to_char[preds[i]])
    return "".join(decoded)

# ==========================================
# 3. LUỒNG XỬ LÝ CHÍNH (PIPELINE YOLO + OCR)
# ==========================================
def main():
    # 1. Thiết lập đường dẫn
    yolo_model_path = "best.pt" 
    ocr_model_path = "best_square_ocr_pro.pth" 
    image_path = r"C:\2026\Du_an_ky_thuat_nang_cao\train_ocr\image.png"

    if not os.path.exists(yolo_model_path):
        print(f"Lỗi: Không tìm thấy file YOLO tại {yolo_model_path}")
        return
    if not os.path.exists(ocr_model_path):
        print(f"Lỗi: Không tìm thấy file OCR tại {ocr_model_path}")
        return
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy file ảnh tại {image_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang chạy Pipeline trên: {device}")

    # 2. Khởi tạo mô hình YOLO & OCR
    print("Đang tải mô hình YOLO...")
    yolo_model = YOLO(yolo_model_path)

    print("Đang tải mô hình OCR (Pro Version)...")
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-."
    idx_to_char = {i + 1: c for i, c in enumerate(chars)}
    idx_to_char[0] = ""
    
    ocr_model = SquareCRNN(len(chars)).to(device)
    ocr_model.load_state_dict(torch.load(ocr_model_path, map_location=device))
    ocr_model.eval()

    # 4. Đọc ảnh gốc
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("Không thể đọc ảnh đầu vào.")
        return

    # 5. DÙNG YOLO ĐỂ CẮT BIỂN SỐ
    results = yolo_model.predict(source=img_bgr, conf=0.5, save=False, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy() 
    
    if len(boxes) == 0:
        print("YOLO không tìm thấy biển số nào.")
        return

    # 6. CẮT ẢNH VÀ ĐƯA VÀO OCR (GIỮ NGUYÊN KHUNG, KHÔNG CẮT DÒNG)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Cắt đúng vùng YOLO khoanh
        cropped_plate = img_bgr[y1:y2, x1:x2]
        gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        
        # Tiền xử lý đưa thẳng vào OCR
        resized_plate = cv2.resize(gray_plate, (128, 128))
        normalized_plate = (resized_plate.astype(np.float32) / 127.5) - 1.0 
        img_tensor = torch.from_numpy(normalized_plate).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds = ocr_model(img_tensor) 
        
        text_result = decode_predictions(preds, idx_to_char)
           
        print(f"Biển số {i+1}: {text_result}")
        
        # Vẽ khung và in chữ lên ảnh gốc
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_bgr, text_result, (x1, max(y1 - 10, 0)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Hiện ảnh cắt ra (để bạn xem YOLO khoanh có chuẩn không)
        cv2.imshow(f"Bien so bi cat {i+1}", cropped_plate)

    cv2.imshow("Ket Qua Pipeline YOLO + OCR", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()