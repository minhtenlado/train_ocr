import os

def rename_images(folder_path, start_index=676, prefix="img_"):
    # Các định dạng ảnh phổ biến để code chỉ đổi tên ảnh, bỏ qua file khác
    valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
    
    try:
        # Lấy danh sách tất cả các file trong thư mục
        files = os.listdir(folder_path)
        
        # Lọc ra các file ảnh và sắp xếp để đổi tên theo thứ tự
        images = [f for f in files if f.lower().endswith(valid_extensions)]
        images.sort() 
        
        current_index = start_index
        count = 0
        
        for filename in images:
            # Lấy đường dẫn đầy đủ của file cũ
            old_file_path = os.path.join(folder_path, filename)
            
            # Lấy phần mở rộng của file (ví dụ: .jpg, .png)
            file_extension = os.path.splitext(filename)[1]
            
            # Tạo tên file mới (ví dụ: img_675.jpg)
            new_filename = f"{prefix}{current_index}{file_extension}"
            new_file_path = os.path.join(folder_path, new_filename)
            
            # Kiểm tra xem tên mới đã tồn tại chưa để tránh ghi đè lỗi
            if not os.path.exists(new_file_path):
                os.rename(old_file_path, new_file_path)
                print(f"Đã đổi tên: {filename}  ->  {new_filename}")
                count += 1
            else:
                print(f"Bỏ qua: {new_filename} đã tồn tại.")
            
            # Tăng chỉ số cho ảnh tiếp theo
            current_index += 1
            
        print(f"\nHoàn thành! Đã đổi tên thành công {count} file ảnh.")
        
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy thư mục. Vui lòng kiểm tra lại đường dẫn.")
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")

# ==========================================
# CÁCH SỬ DỤNG
# ==========================================
# Thay thế chuỗi bên dưới bằng đường dẫn tới thư mục ảnh của bạn.
# Lưu ý: Thêm chữ 'r' trước dấu ngoặc kép trên Windows để tránh lỗi đường dẫn.
thu_muc_anh = r"C:\2026\Du_an_ky_thuat_nang_cao\train_ocr\images" 

rename_images(thu_muc_anh, start_index=676, prefix="img_")