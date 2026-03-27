import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import random
from model import SquareCRNN

class OCRDataset(Dataset):
    def __init__(self, label_file, char_map, img_dir, size=128, is_train=True):
        self.size = size
        self.img_dir = img_dir
        self.char_map = char_map
        self.is_train = is_train
        self.valid_data = []

        print("Đang rà soát và làm sạch dữ liệu...")
        missing_imgs = 0
        invalid_labels = 0

        # Đọc và lọc dữ liệu thông minh ngay từ đầu
        with open(label_file, 'r', encoding='utf-8-sig') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    img_name = parts[0]
                    raw_label = ','.join(parts[1:]) 
                    img_path = os.path.join(self.img_dir, img_name)

                    # 1. Kiểm tra ảnh có tồn tại thực sự không (Khắc phục cảnh báo OpenCV)
                    if not os.path.exists(img_path):
                        missing_imgs += 1
                        continue
                    
                    # 2. Làm sạch nhãn: Chỉ giữ lại các ký tự có trong từ điển
                    clean_label = [self.char_map[c] for c in raw_label.upper() if c in self.char_map]
                    
                    # Nếu nhãn rỗng sau khi làm sạch thì bỏ qua
                    if len(clean_label) == 0:
                        invalid_labels += 1
                        continue
                        
                    self.valid_data.append((img_path, clean_label))

        print(f"-> Đã loại bỏ {missing_imgs} ảnh không tồn tại.")
        print(f"-> Đã loại bỏ {invalid_labels} nhãn không hợp lệ.")
        print(f"-> Tổng số dữ liệu sạch sẵn sàng huấn luyện: {len(self.valid_data)} mẫu.")

    def augment_image(self, img):
        """Thêm nhiễu sáng/tương phản nhẹ để model học thông minh hơn"""
        alpha = random.uniform(0.8, 1.2) # Độ tương phản
        beta = random.randint(-20, 20)   # Độ sáng
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return img

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        img_path, target = self.valid_data[idx]
        
        # Đọc ảnh (lúc này chắc chắn 99% ảnh tồn tại)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Xử lý ngoại lệ nếu file bị hỏng (corrupted)
        if img is None:
            img = np.zeros((self.size, self.size), dtype=np.uint8)
            target = [1] # Nhãn giả định để không lỗi batch

        # Áp dụng Augmentation nếu đang huấn luyện
        if self.is_train and random.random() > 0.5:
            img = self.augment_image(img)

        img = cv2.resize(img, (self.size, self.size))
        img = (img.astype(np.float32) / 127.5) - 1.0 
        img = torch.from_numpy(img).unsqueeze(0)
        
        return img, torch.tensor(target, dtype=torch.long), len(target)

def collate_fn(batch):
    imgs, targets, target_lengths = zip(*batch)
    imgs = torch.stack(imgs)
    targets = torch.cat(targets)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long) 
    return imgs, targets, target_lengths

def main():
    # 1. Khai báo từ điển
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-."
    char_map = {c: i + 1 for i, c in enumerate(chars)}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SquareCRNN(len(chars)).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    # Tinh chỉnh lại scheduler để giảm LR mượt hơn
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, min_lr=1e-6)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(device)

    # Đảm bảo đường dẫn chính xác
    csv_path = 'train.csv' 
    img_dir = 'images'
    
    if not os.path.exists(csv_path):
        print(f"Lỗi rớt mạng: Không tìm thấy file {csv_path}!")
        return
        
    # Khởi tạo dataset với chế độ is_train=True để bật Data Augmentation
    dataset = OCRDataset(csv_path, char_map, img_dir, is_train=True)
    
    if len(dataset) == 0:
        print("Không có dữ liệu hợp lệ để huấn luyện. Vui lòng kiểm tra lại CSV và thư mục ảnh!")
        return

    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=0)

    print(f"--- Bắt đầu huấn luyện trên: {device} ---")
    
    epochs = 100
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for imgs, targets, target_lengths in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            optimizer.zero_grad()
            preds = model(imgs) # [W, B, C]
            
            batch_size = imgs.size(0)
            input_lengths = torch.full(size=(batch_size,), fill_value=preds.size(0), dtype=torch.long, device=device)
            
            loss = criterion(preds.log_softmax(2), targets, input_lengths, target_lengths)
            loss.backward()
            
            # Gradient Clipping: Thông minh chặn lại các lỗi đột biến từ dữ liệu nhiễu
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        
        print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_square_ocr.pth")
            print(f"  -> Lưu mô hình đỉnh nhất! (Loss: {best_loss:.4f})")

if __name__ == "__main__":
    main()