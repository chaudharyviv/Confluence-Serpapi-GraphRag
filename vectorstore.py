import os
import json
import chromadb
from chromadb.config import Settings
from config import VECTOR_DB_DIR, COLLECTION_NAME, VERSION_FILE

os.makedirs("data", exist_ok=True)

client = chromadb.Client(
    Settings(persist_directory=VECTOR_DB_DIR)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

def load_versions():
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE) as f:
            return json.load(f)
    return {}

def save_versions(v):
    with open(VERSION_FILE, "w") as f:
        json.dump(v, f, indent=2)
