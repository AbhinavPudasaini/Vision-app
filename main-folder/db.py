from pymongo import MongoClient
from config import MONGO_URI

client = MongoClient(MONGO_URI)
db = client["photo_manager"]
collection = db["images"]
