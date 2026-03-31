import pandas as pd
import os

# Get base directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')

# 1. Read data from train.txt file (using Tab separator)
input_file = os.path.join(data_dir, 'train.txt')
output_file = os.path.join(data_dir, 'train.csv')

df = pd.read_csv(input_file, sep='\t', names=['Full_Path', 'Nội dung bức ảnh'], header=None)

# 2. Extract image names (part after last '/')
df['Tên bức ảnh'] = df['Full_Path'].str.split('/').str[-1]

# 3. Keep only 2 needed columns
df_final = df[['Tên bức ảnh', 'Nội dung bức ảnh']]

# 4. Export to CSV
df_final.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"Chuyển đổi hoàn tất! File '{output_file}' đã sẵn sàng.")
print(df_final.head())