import torch
from transformers import CLIPProcessor, CLIPModel
from insightface.app import FaceAnalysis

print("[🔄] Loading models...")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

print("[✅] Models loaded.")

def get_clip_model():
    return clip_model, clip_processor

def get_face_app():
    return face_app
