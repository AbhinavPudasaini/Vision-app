from search import build_faiss_index, search_top_k
from PIL import Image
import requests
from io import BytesIO
import os
from search import build_faiss_index, search_top_k, search_face_image,search_face_then_clip, search_multiple_faces_from_images
import time


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

if __name__ == "__main__":
    print("🔎 Building FAISS index...")
    build_faiss_index()

    while True:
        mode = input("\n[1] Text Search\n[2] Face Image Search\n[3] Face plus text\n[4] Multiple faces\n[5] Exit\nChoose: ").strip()

        if mode == "1":
            query = input("Enter your search query: ")
            results = search_top_k(query, k=2)
            print("\nTop 5 Results:")
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['image']} (Score: {res['score']:.4f})")
                display_image(res['image'])


        elif mode == "2":
            img_path = input("Enter image file path or URL: ")
            results = search_face_image(img_path, k=2)
            if not results:
                print("❌ No face detected.")
            else:
                print("\nTop 5 Face Matches:")
                for i, res in enumerate(results, 1):
                    print(f"{i}. {res['image']} (Score: {res['score']:.4f}) BBox: {res['bbox']}")
                display_image(res['image']) 

        elif mode == "3":
            face_path = input("Enter face image path or URL: ")
            text_prompt = input("Enter text description (e.g. 'wearing red shirt'): ")
            results = search_face_then_clip(face_path, text_prompt, k=2)

            if not results:
                print("❌ No match found.")
            else:
                print("\nTop Combined Matches:")
                for i, res in enumerate(results, 1):
                    # print(f"{i}. {res['image']} (Score: {res['score']:.4f})")
                    print(f"{i}. {res['image']} | Face Score: {res['face_score']:.4f}, CLIP Score: {res['clip_score']:.4f}")

                    display_image(res['image'])

        elif mode == "4":
            print("Enter image paths or URLs (comma separated):")
            input_paths = input("Paths: ").strip().split(",")
            image_paths = [path.strip() for path in input_paths if path.strip()]
            print("image_paths:", image_paths)

            results = search_multiple_faces_from_images(image_paths, k=5)
            print("\nTop Group Matches Based on Multiple Input Images:")
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['image']} | Score: {res['score']:.4f} | Faces Matched: {res['matched_faces']}")
                display_image(res['image'])
        elif mode == "5":
            print("Exiting search mode.")
            break
        else:
            print("❌ Invalid option.")