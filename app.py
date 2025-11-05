import os
import json
import numpy as np
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
from sentence_transformers import SentenceTransformer

# ---------------- Cache fix for HuggingFace ----------------
os.environ["HF_HOME"] = "/tmp"
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/tmp"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INDEX_PATH = "/tmp/api_index.faiss"
IDS_PATH = "/tmp/api_ids.npy"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
model = None
index = None
api_ids = []

# ---------- Load model ----------
def get_model():
    global model
    if model is None:
        model = SentenceTransformer(MODEL_NAME, cache_folder="/tmp")
    return model

# ---------- Embeddings ----------
def embed(texts: List[str]):
    m = get_model()
    vectors = m.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vectors.astype("float32")

# ---------- Load / Save Index ----------
def save_state():
    global index, api_ids
    if index is not None:
        faiss.write_index(index, INDEX_PATH)
    np.save(IDS_PATH, np.array(api_ids, dtype=object))

def load_state():
    global index, api_ids
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
    else:
        dim = get_model().get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)

    if os.path.exists(IDS_PATH):
        api_ids = np.load(IDS_PATH, allow_pickle=True).tolist()
    else:
        api_ids = []

load_state()

# ---------- Request Models ----------
class UpsertItem(BaseModel):
    id: str
    description: str

class UpsertRequest(BaseModel):
    items: List[UpsertItem]
    rebuild_if_exists: bool = True

class SearchRequest(BaseModel):
    api_description: str
    closest_k_apis: int = 5

# ---------- Routes ----------

#---------- Health Check ----------
@app.get("/health")
def health():
    return {"status": "ok", "vectors": len(api_ids), "model": MODEL_NAME}

#---------- Upsert APIs ----------
@app.post("/upsert")
def upsert(req: UpsertRequest):
    global index, api_ids

    existing = set(api_ids)
    payload_ids = [i.id for i in req.items]
    any_exists = any(i in existing for i in payload_ids)

    if any_exists and req.rebuild_if_exists:
        all_items_map = {it.id: it.description for it in req.items}
        for id in existing:
            all_items_map.setdefault(id, "")
        pairs = [{"id": k, "description": v} for k, v in all_items_map.items()]
        descs = [p["description"] for p in pairs]
        embs = embed(descs)
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs)
        api_ids = [p["id"] for p in pairs]
        save_state()
        return {"status": "rebuild", "total": len(api_ids)}

    new_items = [it for it in req.items if it.id not in existing]
    if not new_items:
        return {"status": "no-op"}

    descs = [it.description for it in new_items]
    embs = embed(descs)
    index.add(embs)
    api_ids.extend([it.id for it in new_items])
    save_state()
    return {"status": "upsert", "added": len(new_items)}

#---------- Search APIs ----------
@app.post("/search")
def search(req: SearchRequest):
    if len(api_ids) == 0:
        return {"results": []}

    k = min(req.closest_k_apis, len(api_ids))
    q = embed([req.api_description])
    scores, idx = index.search(q, k)

    results = [
        {"api_id": api_ids[i], "semantic_score": round(float(scores[0][j]))}
        for j, i in enumerate(idx[0])
    ]

    return {"results": results}

#---------- Stats ----------
@app.get("/stats")
def stats():
    return {
        "total_apis": len(api_ids),
        "model": MODEL_NAME,
        "index_path": INDEX_PATH,
        "ids_path": IDS_PATH,
    }

