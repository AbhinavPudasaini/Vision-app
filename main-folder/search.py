import faiss
import numpy as np
from pymongo import MongoClient
from model_loader import get_clip_model, get_face_app
from PIL import Image
import requests
from io import BytesIO
import torch
from image_processor import load_image

# MongoDB setup
client = MongoClient("mongodb://localhost:27017")
collection = client["photo_manager"]["images"]

# Store global FAISS indexes
clip_index = None
face_index = None
clip_id_map = []
face_id_map = []

def build_faiss_index():
    global clip_index, clip_id_map, face_index, face_id_map

    clip_model, clip_processor = get_clip_model()
    face_app = get_face_app()

    # Clip index
    clip_index = faiss.IndexFlatIP(768)  # instead of 512
    clip_id_map.clear()

    # Face index
    face_index = faiss.IndexFlatIP(512)
    face_id_map.clear()

    for doc in collection.find():
        if 'clip_embedding' in doc:
            vec = np.array(doc['clip_embedding']).astype(np.float32)
            vec /= np.linalg.norm(vec)
            
            clip_index.add(np.expand_dims(vec, axis=0))
            clip_id_map.append(doc)

        for face in doc.get("faces", []):
            vec = np.array(face["embedding"]).astype(np.float32)
            vec /= np.linalg.norm(vec)
            face_index.add(np.expand_dims(vec, axis=0))
            face_id_map.append({
                "image": doc["source"],
                "bbox": face["bbox"]
            })

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
    top_matches = sorted(face_matches, key=lambda x: x["face_score"], reverse=True)[:top_n_faces]

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
