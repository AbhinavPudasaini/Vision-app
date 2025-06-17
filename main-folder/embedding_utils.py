from PIL import Image
import torch
import numpy as np
from model_loader import get_clip_model, get_face_app

clip_model, clip_processor = get_clip_model()
face_app = get_face_app()

def get_clip_embeddings_batch(images):
    inputs = clip_processor(images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
    return embeddings.cpu().numpy()

def get_face_embeddings_batch(images):
    all_embeddings = []
    for image in images:
        faces = face_app.get(np.array(image))
        emb_list = []
        for face in faces:
            emb_list.append({
                "bbox": face.bbox.tolist(),
                "embedding": face.embedding.tolist()
            })
        all_embeddings.append(emb_list)
    return all_embeddings
