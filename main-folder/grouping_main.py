import os
import shutil
from grouping import find_images_by_face_clustering
from grouping import group_images_by_clip_dbscan_with_text
from image_processor import load_image

# def save_results(image_paths, save_dir="grouped_results"):
#     os.makedirs(save_dir, exist_ok=True)
#     for path in image_paths:
#         try:
#             shutil.copy(path, save_dir)
#         except Exception as e:
#             print(f"❌ Error copying {path}: {e}")
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
        # save_results(matched, save_dir="grouped_by_faces_krisma")

    elif choice == "2":
        # img_path = input("Enter reference image path: ").strip()
        # top_k = int(input("How many similar images to retrieve? (default 10): ") or "10")
        matched = group_images_by_clip_dbscan_with_text(
    text_prompt="people wearing red jackets",
    eps=0.18,
    similarity_threshold=0.23)
        print(f"\n✅ Found {len(matched)} visually similar images.")
        save_groups_to_folders(matched)

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