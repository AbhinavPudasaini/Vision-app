from image_processor import process_and_store_image
import os

if __name__ == "__main__":
    folder_path = None  # Simulated user upload folder
    file_path = input("Enter the path to the image file or folder (leave empty for simulated upload): ")
    if folder_path is not None:
        for file in os.listdir(folder_path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(folder_path, file)
                print(f"Processing image: {img_path}")
                process_and_store_image(img_path)
    if file_path:
        if os.path.isfile(file_path) and file_path.lower().endswith((".jpg", ".jpeg", ".png")):
            print(f"Processing single image: {file_path}")
            process_and_store_image(file_path)
        else:
            print("Invalid file path or unsupported file type. Please provide a valid image file.")
