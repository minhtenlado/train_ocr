import os

def rename_images(folder_path, start_index=676, prefix="img_"):
    """
    Rename all images in a folder with sequential naming.
    
    Args:
        folder_path (str): Path to folder containing images
        start_index (int): Starting number for image names
        prefix (str): Prefix for image names
    """
    # Valid image extensions
    valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
    
    try:
        # Get list of all files in folder
        files = os.listdir(folder_path)
        
        # Filter image files and sort
        images = [f for f in files if f.lower().endswith(valid_extensions)]
        images.sort() 
        
        current_index = start_index
        count = 0
        
        for filename in images:
            # Get full path of old file
            old_file_path = os.path.join(folder_path, filename)
            
            # Get file extension
            file_extension = os.path.splitext(filename)[1]
            
            # Create new filename
            new_filename = f"{prefix}{current_index}{file_extension}"
            new_file_path = os.path.join(folder_path, new_filename)
            
            # Check if new name already exists
            if not os.path.exists(new_file_path):
                os.rename(old_file_path, new_file_path)
                print(f"Đã đổi tên: {filename}  ->  {new_filename}")
                count += 1
            else:
                print(f"Bỏ qua: {new_filename} đã tồn tại.")
            
            current_index += 1
            
        print(f"\nHoàn thành! Đã đổi tên thành công {count} file ảnh.")
        
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy thư mục. Vui lòng kiểm tra lại đường dẫn.")
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")


if __name__ == "__main__":
    # Use relative paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images_dir = os.path.join(base_dir, 'images')
    
    rename_images(images_dir, start_index=676, prefix="img_")