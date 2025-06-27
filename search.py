import faiss
import numpy as np
from pymongo import MongoClient
from model_loader import get_clip_model, get_face_app
from PIL import Image
import requests
from io import BytesIO
import torch
from image_processor import load_image, compute_color_histogram

# MongoDB setup
client = MongoClient("mongodb://localhost:27017")
collection = client["photo_manager"]["images"]

# Store global FAISS indexes
clip_index = None
face_index = None
hist_index = None
clip_id_map = []
face_id_map = []
hist_id_map = []

def build_faiss_index():
    global clip_index, clip_id_map, face_index, face_id_map, hist_index, hist_id_map

    clip_model, clip_processor = get_clip_model()
    face_app = get_face_app()

    # CLIP Index
    clip_index = faiss.IndexHNSWFlat(768, 32)
    clip_index.hnsw.efSearch = 64
    clip_index.hnsw.efConstruction = 80
    clip_id_map.clear()

    # Face Index
    face_index = faiss.IndexHNSWFlat(512, 32)
    face_index.hnsw.efSearch = 64
    face_index.hnsw.efConstruction = 80
    face_id_map.clear()

    # Histogram Index
    hist_index = faiss.IndexHNSWFlat(32 * 32 * 32, 32)
    hist_index.hnsw.efSearch = 64
    hist_index.hnsw.efConstruction = 80
    hist_id_map.clear()

    for doc in collection.find():
        # CLIP Embeddings
        if 'clip_embedding' in doc:
            vec = np.array(doc['clip_embedding']).astype(np.float32)
            vec /= np.linalg.norm(vec)
            clip_index.add(vec[np.newaxis, :])
            clip_id_map.append(doc)

        # Face Embeddings
        for face in doc.get("faces", []):
            vec = np.array(face["embedding"]).astype(np.float32)
            vec /= np.linalg.norm(vec)
            face_index.add(vec[np.newaxis, :])
            face_id_map.append({
                "image": doc["source"],
                "bbox": face["bbox"]
            })

        # Color Histograms
        if 'color_histogram' in doc:
            hist_vec = np.array(doc['color_histogram'], dtype='float32')
            hist_index.add(hist_vec[np.newaxis, :])
            hist_id_map.append({
                "image": doc["source"],
                "color_histogram": doc['color_histogram']
            })
# def search_top_k(query, k=5):
#     clip_model, clip_processor = get_clip_model()
#     inputs = clip_processor(text=[query], return_tensors="pt", padding=True)
#     with torch.no_grad():
#         emb = clip_model.get_text_features(**inputs)
#         emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
#     emb = emb.cpu().numpy().astype(np.float32)

#     scores, indices = clip_index.search(emb, k)
#     return [{
#         "image": clip_id_map[idx]['source'],
#         "score": float(scores[0][i])
#     } for i, idx in enumerate(indices[0])]

def search_top_k(query, k=5):
    clip_model, clip_processor = get_clip_model()
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = clip_model.get_text_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    emb = emb.cpu().numpy().astype(np.float32)

    scores, indices = clip_index.search(emb, k)
    return [{
        "image": clip_id_map[idx]['source'],
        "score": float(scores[0][i])
    } for i, idx in enumerate(indices[0])]

def search_face_image(image_path, k=5):
    face_app = get_face_app()

    # Load image
    if image_path.startswith("http"):
        img = Image.open(BytesIO(requests.get(image_path).content)).convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")

    faces = face_app.get(np.array(img))
    if not faces:
        return []

    face_emb = np.array(faces[0].embedding).astype(np.float32)
    face_emb /= np.linalg.norm(face_emb)

    scores, indices = face_index.search(np.expand_dims(face_emb, axis=0), k)
    # find the mean of score and filter out the faces with score less than 0.2
    # mean_score = np.mean(scores[0])- 0.1
    # indices = [idx for i, idx in enumerate(indices[0]) if scores[0][i] >= mean_score]
    # # get the scores of filtered indices
    # scores = [scores[0][i] for i, idx in enumerate(indices)]

    # return [{
    #     "image": face_id_map[idx]['image'],
    #     "bbox": face_id_map[idx]['bbox'],
    #     "score": float(scores[i])
    # } for i, idx in enumerate(indices)]

    return [{
        "image": face_id_map[idx]['image'],
        "bbox": face_id_map[idx]['bbox'],
        "score": float(scores[0][i])
    } for i, idx in enumerate(indices[0])]

# def search_face_and_text(face_image_path, text_query, k=4, face_threshold=0.2):
#     clip_model, clip_processor = get_clip_model()
#     face_app = get_face_app()

#     db_images = list(collection.find({}))

#     # Step 1: Face Embedding
#     face_img = load_image(face_image_path)
#     np_img = np.array(face_img)
#     faces = face_app.get(np_img)

#     if not faces:
#         print("❌ No face found in input image.")
#         return []

#     input_face_emb = faces[0].embedding
#     input_face_emb = input_face_emb / np.linalg.norm(input_face_emb)  # Normalize for cosine similarity

#     # Step 2: Text embedding
#     text_inputs = clip_processor(text=[text_query], return_tensors="pt", padding=True)
#     with torch.no_grad():
#         text_emb = clip_model.get_text_features(**text_inputs)
#         text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
#         text_emb = text_emb.squeeze().cpu().numpy()

#     results = []

#     for entry in db_images:
#         best_face_score = -1  # cosine similarity ranges [-1, 1]

#         for face_data in entry.get("faces", []):
#             db_face_emb = np.array(face_data["embedding"])
#             db_face_emb = db_face_emb / np.linalg.norm(db_face_emb)  # Normalize

#             face_score = np.dot(input_face_emb, db_face_emb)
#             best_face_score = max(best_face_score, face_score)

#         if best_face_score >= face_threshold:
#             # Combine with CLIP score
#             clip_emb = np.array(entry["clip_embedding"])
#             clip_score = np.dot(text_emb, clip_emb)

#             # Final score: weighted sum (adjust weights as needed)
#             combined_score = (clip_score * 1 + best_face_score * 0.25)

#             results.append({
#                 "image": entry["source"],
#                 "score": combined_score,
#                 "clip_score": clip_score,
#                 "face_score": best_face_score
#             })

#     # Sort by combined score
#     results = sorted(results, key=lambda x: x["score"], reverse=True)

#     # results = sorted(results, key=lambda x: x["clip_score"], reverse=True)
#     return results[:k]

# def search_face_then_clip(face_image_path, text_query, k=5, face_threshold=0.25):
#     clip_model, clip_processor = get_clip_model()
#     face_app = get_face_app()

#     db_images = list(collection.find({}))
#     face_img = load_image(face_image_path)
#     np_img = np.array(face_img)
#     faces = face_app.get(np_img)

#     if not faces:
#         print("❌ No face found.")
#         return []

#     input_face_emb = faces[0].embedding
#     input_face_emb = input_face_emb / np.linalg.norm(input_face_emb)

#     filtered_entries = []

#     # Stage 1: Filter by face similarity
#     for entry in db_images:
#         for face_data in entry.get("faces", []):
#             db_face_emb = np.array(face_data["embedding"])
#             db_face_emb = db_face_emb / np.linalg.norm(db_face_emb)

#             face_score = np.dot(input_face_emb, db_face_emb)
#             if face_score >= face_threshold:
#                 filtered_entries.append({
#                     "entry": entry,
#                     "face_score": face_score
#                 })
#                 break  # Only keep best face match per image

#     # Stage 2: Rank by CLIP score
#     text_inputs = clip_processor(text=[text_query], return_tensors="pt", padding=True)
#     with torch.no_grad():
#         text_emb = clip_model.get_text_features(**text_inputs)
#         text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
#         text_emb = text_emb.squeeze().cpu().numpy()

#     final_results = []
#     for item in filtered_entries:
#         entry = item["entry"]
#         clip_emb = np.array(entry["clip_embedding"])
#         clip_score = np.dot(text_emb, clip_emb)

#         final_results.append({
#             "image": entry["source"],
#             "clip_score": clip_score,
#             "face_score": item["face_score"],
#             "combined_score": clip_score  # final ranking by clip only
#         })

#     final_results = sorted(final_results, key=lambda x: x["clip_score"], reverse=True)
#     return final_results[:k]

def search_face_then_clip(face_image_path, text_query, k=5, top_n_faces=10):
    clip_model, clip_processor = get_clip_model()
    face_app = get_face_app()

    db_images = list(collection.find({}))
    face_img = load_image(face_image_path)
    np_img = np.array(face_img)
    faces = face_app.get(np_img)

    if not faces:
        print("❌ No face found.")
        return []

    input_face_emb = faces[0].embedding
    input_face_emb = input_face_emb / np.linalg.norm(input_face_emb)

    face_matches = []

    # Stage 1: Find top-N similar faces across DB
    for entry in db_images:
        for face_data in entry.get("faces", []):
            db_face_emb = np.array(face_data["embedding"])
            db_face_emb = db_face_emb / np.linalg.norm(db_face_emb)

            face_score = np.dot(input_face_emb, db_face_emb)

            face_matches.append({
                "entry": entry,
                "face_score": face_score
            })
            
    # Sort and take top N face matches
    top_matches = sorted(face_matches, key=lambda x: x["face_score"]> (np.mean(face_matches["face_score"])- 0.1), reverse=True)[:top_n_faces]

    # Step 2: Rank by CLIP (text) similarity
    text_inputs = clip_processor(text=[text_query], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_emb = clip_model.get_text_features(**text_inputs)
        text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
        text_emb = text_emb.squeeze().cpu().numpy()

    final_results = []
    for item in top_matches:
        entry = item["entry"]
        clip_emb = np.array(entry["clip_embedding"])
        clip_score = np.dot(text_emb, clip_emb)

        final_results.append({
            "image": entry["source"],
            "clip_score": clip_score,
            "face_score": item["face_score"],
            "combined_score": clip_score  # you can modify weight here
        })

    final_results = sorted(final_results, key=lambda x: x["clip_score"], reverse=True)
    return final_results[:k]


def search_multiple_faces_from_images(image_paths, k=5, threshold=0.45):
    face_app = get_face_app()

    query_embeddings = []

    for path in image_paths:
        image = load_image(path)
        faces = face_app.get(np.array(image))

        if not faces:
            print(f"⚠️ No faces found in image: {path}")
            continue
        for face in faces:
            # if not hasattr(face, 'embedding'):
            #     print(f"⚠️ No embedding found for face in image: {path}")
            #     continue
            emb = face.embedding
            emb = emb / np.linalg.norm(emb)  # Normalize
            query_embeddings.append(emb)

    # if len(query_embeddings) < 2:
    #     print("❌ Need at least two face embeddings for meaningful group search.")
    #     return []

    db_images = list(collection.find({}))
    results = []

    # Step 2: Search in DB
    for entry in db_images:
        db_faces = entry.get("faces", [])
        db_embeddings = [
            np.array(f["embedding"]) / np.linalg.norm(f["embedding"])
            for f in db_faces
        ]

        matched_scores = []

        for query_emb in query_embeddings:
            similarities = [np.dot(query_emb, db_emb) for db_emb in db_embeddings]
            # best_score = sorted(final_results, key=lambda x: x["clip_score"], reverse=True)
            best_score = max(similarities) if similarities else 0

            if best_score >= threshold:
                matched_scores.append(best_score)
            else:
                matched_scores = []  # One face not found → discard
                break
                
        # for query in query_embeddings:
            
        if matched_scores:
            avg_score = np.mean(matched_scores)
            results.append({
                "image": entry["source"],
                "score": avg_score,
                "matched_faces": len(matched_scores)
            })
    # Sort results by average score

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:k]

import faiss
import numpy as np

# def search_histogram_similarity(query_img_path, k=5):
#     query_hist = compute_color_histogram(query_img_path)

#     if len(hist_id_map) == 0:
#         print("❌ No histograms in FAISS index.")
#         return []

#     query_hist = np.array(query_hist, dtype='float32')
#     distances, indices = hist_index.search(query_hist.reshape(1, -1), k)

#     results = []
#     for i, dist in zip(indices[0], distances[0]):
#         results.append({
#             "image": hist_id_map[i]["image"],
#             "score": dist
#         })
#     return results

import faiss
import numpy as np

# def search_similar_images(query_img_path, k=5):
#     query_emb = get_dino_embedding(query_img_path).reshape(1, -1)
#     distances, indices = index.search(query_emb, k)

#     results = []
#     for i, dist in zip(indices[0], distances[0]):
#         results.append({
#             "image": valid_paths[i],
#             "score": float(dist)
#         })
#     return results

def get_clip_image_embedding(image_path, clip_model, clip_processor):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(clip_model.device)

    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)

    return features.cpu().numpy().astype("float32")

def search_similar_images_clip(query_img_path, k=5):
    clip_model, clip_processor = get_clip_model()
    query_emb = get_clip_image_embedding(query_img_path, clip_model, clip_processor)
    
    distances, indices = clip_index.search(query_emb, k)

    results = []
    for i, dist in zip(indices[0], distances[0]):
        results.append({
            "image": clip_id_map[i]['source'],
            "score": float(dist)
        })
    return results





# def search_histogram_similarity(query_img_path, k=5):
#     query_hist = compute_color_histogram(query_img_path)
#     mongo_images = list(collection.find({}, {"source": 1, "color_histogram": 1}))

#     # Prepare FAISS index
#     image_paths = [img["source"] for img in mongo_images]
#     histogram_list = [np.array(img["color_histogram"], dtype='float32') for img in mongo_images]

#     if len(histogram_list) == 0:
#         print("❌ No histograms in database.")
#         return []

#     dim = len(histogram_list[0])
#     index = faiss.IndexFlatL2(dim)
#     index.add(np.array(histogram_list))

#     distances, indices = index.search(np.array(query_hist).reshape(1, -1), k)

#     results = []
#     for i, dist in zip(indices[0], distances[0]):
#         results.append({
#             "image": image_paths[i],
#             "score": dist
#         })
#     return results

# def fuse_search_results(query_img_path, text_query=None, alpha=0.5, k=5):
#     # alpha: weighting factor (0 = only hist, 1 = only CLIP)
#     results = {}

#     # Histogram search
#     hist_results = search_histogram_similarity(query_img_path, k=k)
#     for r in hist_results:
#         results[r["image"]] = {"hist_score": r["score"], "clip_score": 0.0}

#     # CLIP search (if text_query is given)
#     if text_query:
#         clip_results = search_top_k(text_query, k=k)
#         for r in clip_results:
#             if r["image"] not in results:
#                 results[r["image"]] = {"hist_score": 0.0, "clip_score": r["score"]}
#             else:
#                 results[r["image"]]["clip_score"] = r["score"]

#     # Normalize and combine scores
#     for r in results.values():
#         hist_s = r["hist_score"]
#         clip_s = r["clip_score"]
#         r["combined"] = alpha * clip_s + (1 - alpha) * hist_s

#     # Sort by combined score
#     sorted_results = sorted(results.items(), key=lambda x: x[1]["combined"], reverse=True)
#     return [{"image": img, "score": vals["combined"]} for img, vals in sorted_results[:k]]

import numpy as np

def search_clip_histogram_combined(query_img_path, k=10):
    # Load CLIP model + processor
    clip_model, clip_processor = get_clip_model()
    
    # Step 1: CLIP image embedding and similarity
    query_clip_emb = get_clip_image_embedding(query_img_path, clip_model, clip_processor)
    distances_clip, indices_clip = clip_index.search(query_clip_emb, k)

    # Step 2: Get top-k image paths from CLIP
    top_clip_paths = [clip_id_map[i]['source'] for i in indices_clip[0]]
  

    # Step 3: Compute histogram of the query image
    query_hist = compute_color_histogram(query_img_path)
    query_hist = np.array(query_hist, dtype='float32').reshape(1, -1)

    # Step 4: Get histograms of top-k CLIP images from DB
    mongo_images = list(collection.find(
        {"source": {"$in": top_clip_paths}}, {"source": 1, "color_histogram": 1}
    ))

    # Map source to histogram
    hist_map = {img["source"]: np.array(img["color_histogram"], dtype='float32') for img in mongo_images}

    # Step 5: Build FAISS histogram index
    hist_paths = list(hist_map.keys())
    hist_vectors = list(hist_map.values())

    if len(hist_vectors) == 0:
        print("❌ No histograms found.")
        return []

    dim = len(hist_vectors[0])
    hist_index = faiss.IndexFlatL2(dim)
    hist_index.add(np.vstack(hist_vectors))

    distances_hist, indices_hist = hist_index.search(query_hist, len(hist_vectors))

    # Step 6: Combine Scores
    combined_results = []
    for rank, (i, hist_score) in enumerate(zip(indices_hist[0], distances_hist[0])):
        path = hist_paths[i]
        try:
            clip_score = float(distances_clip[0][top_clip_paths.index(path)])
        except ValueError:
            clip_score = 1.0  # fallback if not found (shouldn't happen)

        combined_score = 0.25 * clip_score + 0.75 * hist_score
        if combined_score < 1.57:


            combined_results.append({
                "image": path,
                "clip_score": clip_score,
                "histogram_score": hist_score,
                "score": combined_score
            })

    # Sort by combined score (lower is better for L2 distance)
    combined_results.sort(key=lambda x: x["score"])
    return combined_results

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

from itertools import combinations

def find_near_duplicate_images(threshold=0.90):
    """
    embeddings: List or array of shape (N, D)
    image_paths: Corresponding image file paths
    threshold: Cosine similarity threshold to mark as near-duplicate
    """
    duplicates = []
    images = collection.find({}, {"source": 1, "clip_embedding": 1})
    # inputs = clip_processor(images=image, return_tensors="pt").to(clip_model.device)
    
    # find all pairs of images and compute cosine similarity
    for img1, img2 in combinations(images, 2):
        emb1 = np.array(img1["clip_embedding"]).astype(np.float32)
        emb2 = np.array(img2["clip_embedding"]).astype(np.float32)

        sim = cosine_similarity(emb1, emb2)

        if sim >= threshold:
            duplicates.append({
                "image1": img1["source"],
                "image2": img2["source"],
                "similarity": sim
            })
    return duplicates


# if __name__ == "__main__":
#     duplicates = find_near_duplicate_images()
#     if duplicates:
#         print("Found near-duplicate images:")
#         for dup in duplicates:
#             print(f"{dup['image1']} <-> {dup['image2']} (Similarity: {dup['similarity']:.4f})")
#             display_image(dup['image1'])
#             display_image(dup['image2'])

