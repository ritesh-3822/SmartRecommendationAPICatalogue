code = r'''
import os
import json
import uuid
from typing import List, Optional

import numpy as np
import faiss
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# ---------------------------
# Config & Paths
# ---------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "/content/api_index.faiss"
IDS_PATH   = "/content/api_ids.npy"
CFG_PATH   = "/content/index_config.json"

# ---------------------------
# Globals
# ---------------------------
app = FastAPI(title="Semantic Search Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

_model: Optional[SentenceTransformer] = None
_index: Optional[faiss.Index] = None
_api_ids: List[str] = []

# ---------------------------
# Utilities
# ---------------------------
def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def l2_normalize(x: np.ndarray) -> np.ndarray:
    # x: (n, d)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / norms).astype("float32")

def embed_texts(texts: List[str]) -> np.ndarray:
    emb = get_model().encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False  # we normalize ourselves
    )
    return l2_normalize(emb.astype("float32"))

def save_ids():
    np.save(IDS_PATH, np.array(_api_ids, dtype=object), allow_pickle=True)

def load_ids():
    global _api_ids
    if os.path.exists(IDS_PATH):
        _api_ids = np.load(IDS_PATH, allow_pickle=True).tolist()
    else:
        _api_ids = []

def save_index():
    if _index is not None:
        faiss.write_index(_index, INDEX_PATH)
    with open(CFG_PATH, "w") as f:
        json.dump({"model": MODEL_NAME, "metric": "cosine_ip_on_l2_norm"}, f)

def load_index():
    global _index
    if os.path.exists(INDEX_PATH):
        _index = faiss.read_index(INDEX_PATH)
    else:
        # Create an empty index with correct dimension from model
        dim = get_model().get_sentence_embedding_dimension()
        _index = faiss.IndexFlatIP(dim)  # inner product over normalized vectors == cosine
    load_ids()

def rebuild_index_from_pairs(pairs: List[dict]):
    """
    pairs: list of {"id": str, "description": str}
    Rebuilds FAISS and replaces _api_ids in the given order.
    """
    global _index, _api_ids
    descs = [(p.get("description") or "") for p in pairs]
    embs = embed_texts(descs)
    dim = embs.shape[1]
    _index = faiss.IndexFlatIP(dim)
    _index.add(embs)
    _api_ids = [p["id"] for p in pairs]
    save_index()
    save_ids()

# ---------------------------
# Pydantic models
# ---------------------------
class UpsertItem(BaseModel):
    id: str
    description: str

class UpsertRequest(BaseModel):
    items: List[UpsertItem] = Field(..., description="List of {id, description}")
    rebuild_if_exists: bool = Field(True, description="If any ID exists, rebuild full index for correctness")

class NearestRequest(BaseModel):
    api_description: str = Field(..., description="Natural language description of the requirement")
    closest_k_apis: int = Field(10, ge=1, le=10, description="Max 10")

class NearestItem(BaseModel):
    api_id: str
    semantic_score: float

class NearestResponse(BaseModel):
    results: List[NearestItem]

# ---------------------------
# Startup
# ---------------------------
load_index()  # loads _index and _api_ids (creates empty if missing)

# ---------------------------
# Routes
# ---------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "vectors": int(_index.ntotal) if _index is not None else 0,
        "ids": len(_api_ids),
    }

@app.get("/stats")
def stats():
    return {
        "index_ntotal": int(_index.ntotal) if _index is not None else 0,
        "id_count": len(_api_ids),
        "dim": get_model().get_sentence_embedding_dimension(),
    }

@app.post("/upsert")
def upsert(req: UpsertRequest):
    """
    Spring Boot pushes catalog items.
    Contract: { "items": [ { "id": "...", "description": "..." }, ... ] }
    Behavior:
      - If rebuild_if_exists=True and any ID already present -> rebuild full index using
        the union of old+new (new overwrites duplicates).
      - Else, append only new IDs.
    """
    global _api_ids, _index

    if not req.items:
        return {"upserted": 0, "rebuild": False}

    existing_ids = set(_api_ids)
    payload_ids = [it.id for it in req.items]
    any_exists = any(i in existing_ids for i in payload_ids)

    if any_exists and req.rebuild_if_exists:
        # Build a dict of current id->description from existing IDs.
        # We do not persist descriptions; so we expect Spring Boot to send the full current snapshot
        # when rebuilding. If a description is missing, we fall back to empty string.
        id_to_desc = {it.id: it.description for it in req.items}

        # Add previously known IDs that aren't in the payload, keeping their description empty
        # (Spring Boot should ideally include them in the payload on rebuild).
        pairs = [{"id": i, "description": id_to_desc.get(i, "")} for i in existing_ids]

        # Add brand-new IDs at the end
        for it in req.items:
            if it.id not in existing_ids:
                pairs.append({"id": it.id, "description": it.description})

        rebuild_index_from_pairs(pairs)
        return {"upserted": len(req.items), "rebuild": True}

    # Fast path: append only new IDs
    new_items = [it for it in req.items if it.id not in existing_ids]
    if not new_items:
        return {"upserted": 0, "rebuild": False}

    descs = [it.description for it in new_items]
    embs = embed_texts(descs)
    _index.add(embs)
    _api_ids.extend([it.id for it in new_items])
    save_index()
    save_ids()
    return {"upserted": len(new_items), "rebuild": False}

@app.post("/search", response_model=NearestResponse)
def search(req: NearestRequest):
    """
    Input:
    {
      "api_description": "Some string description",
      "closest_k_apis": 5
    }
    Output:
    {
      "results": [
        {"api_id": "...", "semantic_score": 0.87},
        ...
      ]
    }
    """
    if _index is None or _index.ntotal == 0 or len(_api_ids) == 0:
        return {"results": []}

    k = min(int(req.closest_k_apis or 10), 10)
    q_emb = embed_texts([req.api_description])
    scores, idxs = _index.search(q_emb, k)

    results: List[NearestItem] = []
    for i, s in zip(idxs[0], scores[0]):
        if i < 0:
            continue
        results.append(NearestItem(api_id=_api_ids[int(i)], semantic_score=float(s)))
    return NearestResponse(results=results)

@app.post("/duplicate-check", response_model=NearestResponse)
def duplicate_check(req: NearestRequest):
    # Same logic/contract as /search (alias for clarity in your backend)
    return search(req)
'''
with open("/content/semantic_service.py", "w") as f:
    f.write(code)
