import os
import shutil
from grouping import find_images_by_face_clustering
from grouping import group_images_by_clip_dbscan_with_text
from image_processor import load_image
import requests
from PIL import Image
from io import BytesIO

def display_image(path_or_url):
    try:
        if path_or_url.startswith("http"):
            response = requests.get(path_or_url)
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(path_or_url)
        img.show()  # Opens with default viewer
    except Exception as e:
        print(f"Failed to display image: {e}")


def save_groups_to_folders(groups, base_dir="grouped_results"):
    import os
    import shutil

    os.makedirs(base_dir, exist_ok=True)

    for group_id, images in groups.items():
        group_folder = os.path.join(base_dir, f"group_{group_id}")
        os.makedirs(group_folder, exist_ok=True)

        for img_path in images:
            try:
                shutil.copy(img_path, group_folder)
            except Exception as e:
                print(f"⚠️ Failed to copy {img_path}: {e}")

if __name__ == "__main__":
    print("📂 Group Images By:")
    print("[1] Faces in Provided Images")
    print("[2] Visual Similarity to a Reference Image")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        img_paths = input("Enter comma-separated face image paths: ").split(",")
        img_paths = [p.strip() for p in img_paths if p.strip()]
        matched = find_images_by_face_clustering(img_paths)
        print(f"\n✅ Found {len(matched)} images with matching faces.")

    elif choice == "2":
        prompt = input("Enter your prompt")
        matched = group_images_by_clip_dbscan_with_text(
                    text_prompt=prompt,
                    )
        print(f"\n✅ Found {len(matched)} visually similar images.")
        for group_id, images in matched.items():
            print(f"\nGroup {group_id}:")
            for img_path in images:
                print(f" - {img_path}")
                # display_image(img_path)

    else:
        print("❌ Invalid choice.")


# input_faces = {
#     "faces/bride.jpg": 1,
#     "faces/groom.jpg": 1,
#     "faces/father.jpg": 2,
#     "faces/mother.jpg": 2,
#     "faces/friend.jpg": 3,
# }

# result = group_by_person_importance(input_faces)