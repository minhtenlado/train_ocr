import pandas as pd

# 1. Đọc dữ liệu từ file train.txt (giả sử dùng dấu Tab để phân tách)
# Nếu file của bạn dùng khoảng trắng, hãy thay sep='\t' thành sep='\s+'
df = pd.read_csv('train.txt', sep='\t', names=['Full_Path', 'Nội dung bức ảnh'], header=None)

# 2. Tách lấy tên bức ảnh (phần sau dấu '/' cuối cùng)
df['Tên bức ảnh'] = df['Full_Path'].str.split('/').str[-1]

# 3. Chỉ lấy 2 cột bạn cần
df_final = df[['Tên bức ảnh', 'Nội dung bức ảnh']]

# 4. Xuất ra file CSV
df_final.to_csv('train_final.csv', index=False, encoding='utf-8-sig')

print("Chuyển đổi hoàn tất! File 'train_final.csv' đã sẵn sàng.")
print(df_final.head())