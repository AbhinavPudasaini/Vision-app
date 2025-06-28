import numpy as np
from model_loader import get_face_app
from db import collection
from image_processor import load_image
import torch
from model_loader import get_clip_model

import numpy as np
from sklearn.cluster import DBSCAN
from db import collection

def find_images_by_face_clustering(face_image_paths, eps=0.62, min_samples=1):
    face_app = get_face_app()
    query_embeddings = []

    # Step 1: Extract query embeddings
    for path in face_image_paths:
        image = load_image(path)
        faces = face_app.get(np.array(image))
        if not faces:
            print(f"⚠️ No face found in {path}")
            continue
        for face in faces:
            emb = face.embedding / np.linalg.norm(face.embedding)
            query_embeddings.append(emb)

    if not query_embeddings:
        print("❌ No valid face embeddings.")
        return []

    query_embeddings = np.array(query_embeddings)

    # Step 2: Extract all DB face embeddings
    all_embeddings = []
    image_refs = []

    for entry in collection.find({}):
        for face_data in entry.get("faces", []):
            emb = np.array(face_data["embedding"])
            norm_emb = emb / np.linalg.norm(emb)
            all_embeddings.append(norm_emb)
            image_refs.append(entry["source"])

    if not all_embeddings:
        print("❌ No face data in DB.")
        return []

    all_embeddings = np.array(all_embeddings)

    # Step 3: Combine and Cluster
    combined = np.vstack([query_embeddings, all_embeddings])
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(combined)

    # Step 4: Find cluster(s) that query face(s) belong to
    query_cluster_ids = set(clustering.labels_[:len(query_embeddings)])
    if -1 in query_cluster_ids:
        query_cluster_ids.remove(-1)  # remove noise cluster

    print(f"🔍 Matched clusters: {query_cluster_ids}")

    # Step 5: Find DB image indices with same cluster
    matched_images = set()
    for idx, label in enumerate(clustering.labels_[len(query_embeddings):]):
        if label in query_cluster_ids:
            matched_images.add(image_refs[idx])

    return list(matched_images)





# def find_similar_images_by_clip(reference_image_path, k=10):
#     model, processor = get_clip_model()
#     image = load_image(reference_image_path).convert("RGB")

#     inputs = processor(images=image, return_tensors="pt")
#     with torch.no_grad():
#         query_emb = model.get_image_features(**inputs)
#         query_emb = query_emb / query_emb.norm(p=2, dim=-1, keepdim=True)
#         query_emb = query_emb.squeeze().cpu().numpy()

#     db_entries = list(collection.find({}))
#     results = []

#     for entry in db_entries:
#         db_emb = np.array(entry["clip_embedding"])
#         score = np.dot(query_emb, db_emb)
#         results.append((entry["source"], score))

#     results.sort(key=lambda x: x[1], reverse=True)
#     return [path for path, _ in results[:k]]

import numpy as np
from sklearn.cluster import KMeans
from pymongo import MongoClient
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image
import os

# MongoDB setup
# client = MongoClient("mongodb://localhost:27017/")
# db = client["your_db_name"]
# collection = db["your_collection_name"]

# # Load CLIP embeddings and image paths
# def load_embeddings():
#     entries = list(collection.find({}))
#     embeddings = []
#     image_paths = []

#     for entry in entries:
#         if "clip_embedding" in entry:
#             embeddings.append(np.array(entry["clip_embedding"]))
#             image_paths.append(entry["source"])
    
#     return np.array(embeddings), image_paths

# Cluster images using KMeans
import numpy as np
from sklearn.cluster import KMeans
from db import collection

# def group_images_by_clip(k=10):
#     clip_embeddings = []
#     image_sources = []

#     for entry in collection.find({}):
#         clip_embeddings.append(entry["clip_embedding"])
#         image_sources.append(entry["source"])

#     if not clip_embeddings:
#         print("❌ No clip embeddings.")
#         return {}

#     clip_embeddings = np.array(clip_embeddings)

#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(clip_embeddings)

#     grouped = {}
#     for label, path in zip(labels, image_sources):
#         grouped.setdefault(label, []).append(path)

    # return grouped

# import numpy as np
# from sklearn.cluster import DBSCAN
# from pymongo import MongoClient
# from collections import defaultdict
# from db import collection
# from model_loader import get_clip_model

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from collections import defaultdict
# from models import get_clip_model
from db import collection  # Your MongoDB setup
from transformers import CLIPProcessor

def group_images_by_clip_dbscan_with_text(
    text_prompt,
    eps=0.27,
    min_samples=3,
    similarity_threshold=0.2
):
    clip_model, clip_processor = get_clip_model()

    # Step 1: Convert text to CLIP embedding
    inputs = clip_processor(text=[text_prompt], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_emb = clip_model.get_text_features(**inputs)
        text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
        text_emb = text_emb.squeeze().cpu().numpy()

    # Step 2: Collect image embeddings + sources
    clip_embeddings = []
    image_sources = []

    for entry in collection.find({}):
        if "clip_embedding" in entry:
            emb = np.array(entry["clip_embedding"])
            emb_norm = emb / np.linalg.norm(emb)
            score = np.dot(emb_norm, text_emb)

            if score >= similarity_threshold:
                clip_embeddings.append(emb_norm)
                image_sources.append(entry["source"])

    if not clip_embeddings:
        print("❌ No images match the text prompt.")
        return {}

    clip_embeddings = np.array(clip_embeddings)

    # Step 3: DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = dbscan.fit_predict(clip_embeddings)

    # Step 4: Grouping
    grouped = defaultdict(list)
    for label, path in zip(labels, image_sources):
        grouped[label].append(path)

    return dict(grouped)



import numpy as np
from collections import defaultdict
# from models import get_face_app
from db import collection
# from utils import load_image
import random

def get_importance_percentage(level):
    # Define % selection by importance level (you can tweak)
    return {
        1: 0.9,
        2: 0.7,
        3: 0.5,
        4: 0.3,
        5: 0.1
    }.get(level, 0.1)  # Default 10% for unknown levels

def group_by_person_importance(face_image_paths_with_level, threshold=0.5):
    face_app = get_face_app()
    db_entries = list(collection.find({}))
    image_matches = defaultdict(list)  # {level: [images]}

    for face_path, level in face_image_paths_with_level.items():
        img = load_image(face_path)
        np_img = np.array(img)
        faces = face_app.get(np_img)

        if not faces:
            print(f"❌ No face found in {face_path}")
            continue

        query_emb = faces[0].embedding
        query_emb = query_emb / np.linalg.norm(query_emb)

        matched_images = []

        for entry in db_entries:
            for face_data in entry.get("faces", []):
                db_emb = np.array(face_data["embedding"])
                db_emb = db_emb / np.linalg.norm(db_emb)

                sim = np.dot(query_emb, db_emb)
                if sim >= threshold:
                    matched_images.append(entry["source"])
                    break

        # Deduplicate and shuffle
        matched_images = list(set(matched_images))
        random.shuffle(matched_images)

        # Sample based on level
        sample_percent = get_importance_percentage(level)
        sample_size = int(len(matched_images) * sample_percent)
        selected = matched_images[:sample_size]

        image_matches[level].extend(selected)

    return image_matches

# --------------------------------------------------------------------

# def get_clip_image_embedding(image_path, clip_model, clip_processor):
#     image = Image.open(image_path).convert("RGB")
#     inputs = clip_processor(images=image, return_tensors="pt").to(clip_model.device)

#     with torch.no_grad():
#         features = clip_model.get_image_features(**inputs)
#         features = features / features.norm(p=2, dim=-1, keepdim=True)

#     return features.cpu().numpy().astype("float32")

# def search_similar_images_clip(query_img_path, k=5):
#     clip_model, clip_processor = get_clip_model()
#     query_emb = get_clip_image_embedding(query_img_path, clip_model, clip_processor)
    
#     distances, indices = clip_index.search(query_emb, k)

#     results = []
#     for i, dist in zip(indices[0], distances[0]):
#         results.append({
#             "image": clip_id_map[i]['source'],
#             "score": float(dist)
#         })
#     return results

# def search_clip_histogram_combined(query_img_path, k=15):
#     # Load CLIP model + processor
#     clip_model, clip_processor = get_clip_model()
    
#     # Step 1: CLIP image embedding and similarity
#     query_clip_emb = get_clip_image_embedding(query_img_path, clip_model, clip_processor)
#     distances_clip, indices_clip = clip_index.search(query_clip_emb, k)

#     # Step 2: Get top-k image paths from CLIP
#     top_clip_paths = [clip_id_map[i]['source'] for i in indices_clip[0]]

#     # Step 3: Compute histogram of the query image
#     query_hist = compute_color_histogram(query_img_path)
#     query_hist = np.array(query_hist, dtype='float32').reshape(1, -1)

#     # Step 4: Get histograms of top-k CLIP images from DB
#     mongo_images = list(collection.find(
#         {"source": {"$in": top_clip_paths}}, {"source": 1, "color_histogram": 1}
#     ))

#     # Map source to histogram
#     hist_map = {img["source"]: np.array(img["color_histogram"], dtype='float32') for img in mongo_images}

#     # Step 5: Build FAISS histogram index
#     hist_paths = list(hist_map.keys())
#     hist_vectors = list(hist_map.values())

#     if len(hist_vectors) == 0:
#         print("❌ No histograms found.")
#         return []

#     dim = len(hist_vectors[0])
#     hist_index = faiss.IndexFlatL2(dim)
#     hist_index.add(np.vstack(hist_vectors))

#     distances_hist, indices_hist = hist_index.search(query_hist, len(hist_vectors))

#     # Step 6: Combine Scores
#     combined_results = []
#     for rank, (i, hist_score) in enumerate(zip(indices_hist[0], distances_hist[0])):
#         path = hist_paths[i]
#         try:
#             clip_score = float(distances_clip[0][top_clip_paths.index(path)])
#         except ValueError:
#             clip_score = 1.0  # fallback if not found (shouldn't happen)

#         combined_score = 0.5 * clip_score + 0.5 * hist_score

#         combined_results.append({
#             "image": path,
#             "clip_score": clip_score,
#             "histogram_score": hist_score,
#             "score": combined_score
#         })

#     # Sort by combined score (lower is better for L2 distance)
#     combined_results.sort(key=lambda x: x["score"])
#     return combined_results