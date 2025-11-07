# step3_full_pipeline_faiss.py
"""
Pairwise matching pipeline using faiss.IndexFlatIP (inner product) for ANN.
- Normalise les embeddings pour que IP == cosine similarity.
- Cache les embeddings en emb_cache/<source>_embs.npz
- Rerank via score_pair (champ-par-champ) sur candidates retournés par faiss.
"""

import os
import json
import re
import math
import hashlib
import struct
import random
import unicodedata
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import pathlib
import time

# try import faiss and show clear message if missing
try:
    import faiss
except Exception as e:
    raise ImportError("faiss is required for this script. Install with `pip install faiss-cpu` (or faiss-gpu).") from e

# ---------------------------
# Config (change paths/names if needed)
# ---------------------------
EMBED_FILES = {
    "linkedin": "linkedin_profiles_normalized.json",
    "twitter":  "twitter_data_cleaned.json",
    "github":   "github_cleaned.json"
}
OUT_TEMPLATE = "results4/matches_{a}_{b}_mapped_top1.json"

FIELD_WEIGHTS = {
    "username": 0.60,
    "name": 0.60,
    "bio": 0.20,
    "repo_names": 0.10,
    "repo_descriptions": 0.10,
    "location": 0.05,
    "_default": 0.05
}

PAIR_CONFIGS = {
    ("linkedin", "github"): {"threshold": 0.75, "top_k": 5, "field_weights": FIELD_WEIGHTS},
    ("linkedin", "twitter"): {"threshold": 0.75, "top_k": 3, "field_weights": FIELD_WEIGHTS},
    ("twitter", "github"): {"threshold": 0.75, "top_k": 5, "field_weights": FIELD_WEIGHTS},
    ("twitter", "linkedin"): {"threshold": 0.75, "top_k": 5, "field_weights": FIELD_WEIGHTS},
    ("github", "twitter"): {"threshold": 0.75, "top_k": 4, "field_weights": FIELD_WEIGHTS},
    ("github", "linkedin"): {"threshold": 0.75, "top_k": 4, "field_weights": FIELD_WEIGHTS},
}

PAIR_FIELDS = {
    ("linkedin", "github"): ["username", "name", "bio", "repo_names", "repo_descriptions"],
    ("linkedin", "twitter"): ["username", "name", "bio"],
    ("github" , "linkedin"): ["username", "name", "bio", "repo_names", "repo_descriptions"],
    ("github", "twitter"): ["username", "name", "bio"],
    ("twitter", "github"): ["username", "name","bio"],
    ("twitter" , "linkedin"): ["username", "name", "bio"],
}

# Model & dims
SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
THRESHOLD = 0.6
TOP_K = 5
CANON_DIM = 384  # fallback dim

# ---------------------------
# Utilities: IO + normalization
# ---------------------------
_non_alnum = re.compile(r"[^0-9a-z ]+")

def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    try:
        s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    except Exception:
        pass
    s = s.replace("\n", " ").replace("\r", " ")
    s = _non_alnum.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------------------------
# String similarity utilities (same as before)
# ---------------------------
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost)
        prev = cur
    return prev[-1]

def normalized_levenshtein(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    dist = levenshtein(a, b)
    denom = max(len(a), len(b))
    return 1.0 - (dist / denom) if denom > 0 else 0.0

def jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    s1_len, s2_len = len(s1), len(s2)
    match_distance = max((max(s1_len, s2_len) // 2) - 1, 0)
    s1_matches = [False]*s1_len
    s2_matches = [False]*s2_len
    matches = 0
    transpositions = 0
    for i in range(s1_len):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, s2_len)
        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    k = 0
    for i in range(s1_len):
        if s1_matches[i]:
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
    transpositions /= 2
    jaro = ((matches / s1_len) + (matches / s2_len) + ((matches - transpositions) / matches)) / 3.0
    prefix = 0
    for i in range(min(4, s1_len, s2_len)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    jw = jaro + (prefix * p * (1 - jaro))
    return jw

# ---------------------------
# Embedding helpers
# ---------------------------
def cosine_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def deterministic_hash_vector(s: str, dim: int = CANON_DIM, seedfold: int = 0) -> np.ndarray:
    if s is None:
        s = ""
    h = hashlib.sha256(s.encode("utf-8") + struct.pack("I", seedfold)).digest()
    seed = int.from_bytes(h[:8], "big")
    rnd = random.Random(seed)
    vec = np.fromiter((rnd.gauss(0,1) for _ in range(dim)), dtype=np.float32, count=dim)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-12) if norm > 0 else vec

def extract_embedding_from_field(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, list) and value and all(isinstance(x, (int, float)) for x in value):
        return np.array(value, dtype=np.float32)
    if isinstance(value, list) and value and isinstance(value[0], list):
        arrs = []
        for el in value:
            if el is None:
                continue
            if all(isinstance(x, (int, float)) for x in el):
                arrs.append(np.array(el, dtype=np.float32))
        if arrs:
            return np.vstack(arrs).mean(axis=0)
    return None

# ---------------------------
# Efficient batch encoding (encode only non-empty texts)
# ---------------------------
def batch_encode_texts_efficient(texts: List[str], batch_size: int = 64) -> List[Optional[np.ndarray]]:
    out = [None] * len(texts)
    idxs = [i for i, t in enumerate(texts) if t and str(t).strip() != ""]
    if not idxs:
        return out
    non_empty_texts = [texts[i] for i in idxs]
    for i in range(0, len(non_empty_texts), batch_size):
        batch = non_empty_texts[i:i+batch_size]
        encoded = SBERT_MODEL.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        for j, arr in enumerate(encoded):
            out[idxs[i+j]] = np.array(arr, dtype=np.float32)
    return out

# ---------------------------
# Field-level similarity (same as before)
# ---------------------------
def username_similarity(a_raw: Dict[str,Any], b_raw: Dict[str,Any]) -> float:
    a_user = (a_raw.get("profile_id") or a_raw.get("username") or a_raw.get("user_id") or a_raw.get("Username"))
    b_user = (b_raw.get("profile_id") or b_raw.get("username") or b_raw.get("user_id") or b_raw.get("Username"))
    if a_user is None or b_user is None:
        return 0.0
    def to_str(u):
        if isinstance(u, list):
            return " ".join(map(str,u))
        return str(u)
    a_s = normalize_text(to_str(a_user))
    b_s = normalize_text(to_str(b_user))
    if not a_s or not b_s:
        return 0.0
    if a_s == b_s:
        return 1.0
    jw = jaro_winkler(a_s, b_s)
    return jw

def name_similarity(a_raw: Dict[str,Any], b_raw: Dict[str,Any]) -> float:
    a_name = a_raw.get("full_name") or a_raw.get("fullName") or a_raw.get("name")
    b_name = b_raw.get("name") or b_raw.get("full_name") or b_raw.get("fullName")
    if a_name is None or b_name is None:
        return 0.0
    def to_str(x):
        if isinstance(x, list):
            return " ".join(map(str,x))
        return str(x)
    a_s = normalize_text(to_str(a_name))
    b_s = normalize_text(to_str(b_name))
    if not a_s or not b_s:
        return 0.0
    jw = jaro_winkler(a_s, b_s)
    lv = normalized_levenshtein(a_s, b_s)
    a_tokens = set(a_s.split())
    b_tokens = set(b_s.split())
    jacc = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
    return (0.5 * jw) + (0.3 * lv) + (0.2 * jacc)

def location_similarity(a_raw: Dict[str,Any], b_raw: Dict[str,Any]) -> float:
    a_loc = a_raw.get("location")
    b_loc = b_raw.get("location")
    if not a_loc or not b_loc:
        return 0.0
    def to_str(x):
        if isinstance(x, list):
            return " ".join(map(str,x))
        return str(x)
    a_s = normalize_text(to_str(a_loc))
    b_s = normalize_text(to_str(b_loc))
    if a_s == b_s:
        return 1.0
    jw = jaro_winkler(a_s, b_s)
    return jw

def semantic_similarity_field(a_emb: Optional[np.ndarray], b_emb: Optional[np.ndarray]) -> float:
    return cosine_sim(a_emb, b_emb)

# ---------------------------
# Prepare profiles (same as previous version)
# ---------------------------
def prepare_profiles(raw_profiles: List[Dict[str,Any]], source_name: str, canonical_dim: int) -> List[Dict[str,Any]]:
    prepared = []
    bio_texts = []
    repo_names_texts = []
    repo_desc_texts = []
    projects_texts = []
    name_texts = []
    idx_to_item = []

    for p in raw_profiles:
        item = {"__orig__": p, "source": source_name}
        item["norm"] = {}
        username_candidates = (p.get("profile_id") or p.get("username") or p.get("user_id") or p.get("Username"))
        if username_candidates is not None:
            if isinstance(username_candidates, list):
                username_candidates = " ".join(map(str, username_candidates))
            item["norm"]["username"] = normalize_text(str(username_candidates))
        else:
            item["norm"]["username"] = ""
        name_val = p.get("full_name") or p.get("fullName") or p.get("name")
        if name_val is not None:
            if isinstance(name_val, list):
                name_val = " ".join(map(str, name_val))
            item["norm"]["name"] = normalize_text(str(name_val))
        else:
            item["norm"]["name"] = ""
        loc = p.get("location")
        if loc is not None:
            if isinstance(loc, list):
                loc = " ".join(map(str, loc))
            item["norm"]["location"] = normalize_text(str(loc))
        else:
            item["norm"]["location"] = ""
        bio_fields = []
        for k in ("headline", "about", "bio", "description"):
            v = p.get(k)
            if v:
                if isinstance(v, list):
                    bio_fields.append(" ".join(map(str, v)))
                else:
                    bio_fields.append(str(v))
        if p.get("company"):
            bio_fields.append(str(p.get("company")))
        if p.get("repo_descriptions") and isinstance(p.get("repo_descriptions"), list):
            joined = " ".join(str(x) for x in p.get("repo_descriptions") if isinstance(x, (str, int, float)))
            if joined:
                bio_fields.append(joined)
        item["norm"]["bio_text"] = normalize_text(" ".join(bio_fields)) if bio_fields else ""
        if p.get("projects") and isinstance(p.get("projects"), list):
            proj_texts = []
            for proj in p.get("projects"):
                if proj is None:
                    continue
                if isinstance(proj, dict):
                    title = proj.get("name") or proj.get("title") or proj.get("project_name") or ""
                    desc  = proj.get("description") or proj.get("descriptions") or proj.get("details") or ""
                    if isinstance(title, list):
                        title = " ".join(str(x) for x in title if x)
                    if isinstance(desc, list):
                        desc = " ".join(str(x) for x in desc if x)
                    if title:
                        proj_texts.append(str(title))
                    if desc:
                        proj_texts.append(str(desc))
                elif isinstance(proj, (str, int, float)):
                    proj_texts.append(str(proj))
            if proj_texts:
                joined_projects = " ".join(proj_texts)
                item["norm"]["projects_text"] = normalize_text(joined_projects)
            else:
                item["norm"]["projects_text"] = ""
        else:
            item["norm"]["projects_text"] = ""
        item["emb"] = {"global": None, "bio": None, "repo_names": None, "repo_descriptions": None, "projects": None, "name": None}
        if "bio" in p:
            existing = extract_embedding_from_field(p.get("bio"))
            if existing is not None:
                item["emb"]["bio"] = existing
        if "repo_descriptions" in p:
            existing = extract_embedding_from_field(p.get("repo_descriptions"))
            if existing is not None:
                item["emb"]["repo_descriptions"] = existing
        rn = p.get("repo_names")
        if rn:
            if isinstance(rn, list):
                joined = " ".join(str(x) for x in rn if isinstance(x, (str, int, float)))
            else:
                joined = str(rn)
            item["norm"].setdefault("repo_names_text", normalize_text(joined) if joined else "")
        else:
            item["norm"].setdefault("repo_names_text", "")
        bio_texts.append(item["norm"]["bio_text"] or "")
        repo_names_texts.append(item["norm"].get("repo_names_text","") or "")
        repo_desc_texts.append(item["norm"]["bio_text"] or "")
        projects_texts.append(item["norm"].get("projects_text","") or "")
        name_texts.append(item["norm"].get("name","") or "")
        idx_to_item.append(item)
        prepared.append(item)

    bio_embs = batch_encode_texts_efficient(bio_texts, batch_size=64)
    repo_names_embs = batch_encode_texts_efficient(repo_names_texts, batch_size=64)
    projects_embs = batch_encode_texts_efficient(projects_texts, batch_size=64)
    name_embs = batch_encode_texts_efficient(name_texts, batch_size=64)

    for idx, item in enumerate(prepared):
        p = item["__orig__"]
        if item["emb"].get("bio") is None:
            emb = bio_embs[idx]
            if emb is not None:
                item["emb"]["bio"] = emb
            else:
                item["emb"]["bio"] = deterministic_hash_vector(item["norm"].get("bio_text",""), dim=canonical_dim) if item["norm"].get("bio_text") else None
        if item["emb"].get("repo_names") is None:
            emb = repo_names_embs[idx]
            if emb is not None and emb.size>0:
                item["emb"]["repo_names"] = emb
            else:
                rn_text = item["norm"].get("repo_names_text","")
                item["emb"]["repo_names"] = deterministic_hash_vector(rn_text, dim=canonical_dim) if rn_text else None
        if item["emb"].get("repo_descriptions") is None:
            if item["emb"].get("repo_names") is not None:
                item["emb"]["repo_descriptions"] = item["emb"].get("repo_names")
            elif item["emb"].get("bio") is not None:
                item["emb"]["repo_descriptions"] = item["emb"].get("bio")
            else:
                item["emb"]["repo_descriptions"] = None
        if item["emb"].get("projects") is None:
            emb = projects_embs[idx]
            if emb is not None:
                item["emb"]["projects"] = emb
            else:
                proj_text = item["norm"].get("projects_text","")
                item["emb"]["projects"] = deterministic_hash_vector(proj_text, dim=canonical_dim) if proj_text else None
        if item["emb"].get("name") is None:
            emb = name_embs[idx]
            if emb is not None:
                item["emb"]["name"] = emb
            else:
                nm = item["norm"].get("name","")
                item["emb"]["name"] = deterministic_hash_vector(nm, dim=canonical_dim) if nm else None

    return prepared

# ---------------------------
# Compute global embedding (weighted mean)
# ---------------------------
def compute_global_embedding(item: Dict[str,Any], field_weights: Dict[str,float], dim: int = CANON_DIM):
    keys = ["name", "username", "bio", "repo_names", "repo_descriptions", "projects"]
    vecs = []
    ws = []
    for k in keys:
        emb = item["emb"].get(k)
        if emb is None:
            continue
        w = field_weights.get(k, field_weights.get("_default", 0.05))
        vecs.append(emb.astype(np.float32))
        ws.append(w)
    if not vecs:
        return None
    mat = np.vstack(vecs)
    ws = np.array(ws, dtype=np.float32)[:, None]
    weighted = (mat * ws).sum(axis=0) / (ws.sum() + 1e-12)
    norm = np.linalg.norm(weighted)
    return weighted / (norm + 1e-12) if norm > 0 else weighted

# ---------------------------
# Embedding cache (same)
# ---------------------------
def embeddings_cache_path(source_name: str):
    p = pathlib.Path("emb_cache")
    p.mkdir(exist_ok=True)
    return p / f"{source_name}_embs.npz"

def save_embeddings_cache(prepared: List[Dict[str,Any]], source_name: str, dim: int = CANON_DIM):
    arrs = {}
    keys = ("global","bio","repo_names","repo_descriptions","projects","name")
    mats = {k: [] for k in keys}
    for p in prepared:
        for k in keys:
            e = p["emb"].get(k)
            if e is None:
                mats[k].append(np.zeros((dim,), dtype=np.float32))
            else:
                mats[k].append(np.asarray(e, dtype=np.float32))
    for k in keys:
        arrs[k] = np.vstack(mats[k])
    np.savez_compressed(embeddings_cache_path(source_name), **arrs)

def load_embeddings_cache_if_exists(prepared: List[Dict[str,Any]], source_name: str):
    p = embeddings_cache_path(source_name)
    if not p.exists():
        return False
    data = np.load(p)
    keys = ("global","bio","repo_names","repo_descriptions","projects","name")
    for i, item in enumerate(prepared):
        for k in keys:
            arr = data[k][i]
            item["emb"][k] = arr.astype(np.float32)
    return True

# ---------------------------
# Scoring pair (same)
# ---------------------------
def score_pair(a_prep: Dict[str,Any], b_prep: Dict[str,Any], field_weights: Dict[str,float], fields_to_use: Optional[List[str]] = None) -> Dict[str,Any]:
    per_field = {}
    total_weight = 0.0
    weighted_sum = 0.0
    if fields_to_use is None:
        fields_to_use = ["username", "name", "bio", "repo_names", "repo_descriptions", "location"]
    for field in fields_to_use:
        a_emb = a_prep["emb"].get(field)
        b_emb = b_prep["emb"].get(field)
        if field in ("bio","projects","repo_names","repo_descriptions") and (a_emb is None or b_emb is None):
            s = 0.0
        else:
            if field == "username":
                s = username_similarity(a_prep["__orig__"], b_prep["__orig__"])
            elif field == "name":
                s = name_similarity(a_prep["__orig__"], b_prep["__orig__"])
            elif field == "bio":
                s = semantic_similarity_field(a_prep["emb"].get("bio"), b_prep["emb"].get("bio"))
            elif field == "projects":
                s = semantic_similarity_field(a_prep["emb"].get("projects"), b_prep["emb"].get("projects"))
            elif field == "repo_names":
                s = semantic_similarity_field(a_prep["emb"].get("repo_names"), b_prep["emb"].get("repo_names"))
            elif field == "repo_descriptions":
                s = semantic_similarity_field(a_prep["emb"].get("repo_descriptions"), b_prep["emb"].get("repo_descriptions"))
            elif field == "location":
                s = location_similarity(a_prep["__orig__"], b_prep["__orig__"])
            else:
                s = 0.0
        w = field_weights.get(field, field_weights.get("_default", 0.0))
        per_field[field] = {"score": s, "weight": w}
        weighted_sum += s * w
        total_weight += w
    final_score = float(weighted_sum / total_weight) if total_weight > 0 else 0.0
    return {"score": final_score, "per_field": per_field}

# ---------------------------
# FAISS index build & query
# ---------------------------
def build_faiss_index_from_prepared(B_prepared, dim):
    rows = []
    for p in B_prepared:
        g = p["emb"].get("global")
        if g is None:
            g = p["emb"].get("bio") or p["emb"].get("name")
            if g is None:
                g = np.zeros((dim,), dtype=np.float32)
        rows.append(np.asarray(g, dtype=np.float32))
    B_matrix = np.vstack(rows)  # shape (n_b, dim)
    # normalize rows to unit length (so inner product == cosine)
    norms = np.linalg.norm(B_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    B_norm = B_matrix / norms
    # faiss wants contiguous float32
    B_norm = np.ascontiguousarray(B_norm.astype(np.float32))
    dim = B_norm.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product index
    index.add(B_norm)  # add vectors
    return index, B_norm

def query_faiss(index, B_norm, queries, top_k=50, batch_size=512):
    """
    queries: np.ndarray shape (n_q, dim) (not necessarily normalised)
    Returns: list of length n_q, each element list of (idx, sim)
    """
    n_q = queries.shape[0]
    results = [None] * n_q
    for i in range(0, n_q, batch_size):
        stop = min(n_q, i + batch_size)
        batch = queries[i:stop].astype(np.float32)
        # normalize queries rows
        q_norms = np.linalg.norm(batch, axis=1, keepdims=True)
        q_norms[q_norms == 0] = 1e-12
        qn = batch / q_norms
        qn = np.ascontiguousarray(qn.astype(np.float32))
        # distances = inner product (since both normalized => cosine)
        D, I = index.search(qn, min(top_k, B_norm.shape[0]))
        # D shape (batch_size, top_k)
        for bi in range(qn.shape[0]):
            idxs = I[bi].tolist()
            sims = D[bi].tolist()
            results[i + bi] = list(zip(idxs, sims))
    return results

# ---------------------------
# Main pairwise function using FAISS + rerank (with safe global assignment)
# ---------------------------
def pairwise_match_faiss(a_name: str, b_name: str,
                         selected_fields: Optional[List[str]] = None,
                         threshold: float = THRESHOLD,
                         top_k: int = TOP_K,
                         field_weights: Dict[str,float] = FIELD_WEIGHTS,
                         faiss_k: int = 50,
                         use_cache: bool = True):
    t0 = time.perf_counter()
    rawA = load_json(EMBED_FILES[a_name])
    rawB = load_json(EMBED_FILES[b_name])
    print(f"[info] loaded {len(rawA)} A, {len(rawB)} B")

    # detect embedding dim if present
    detected_dim = None
    for p in rawB:
        for k,v in p.items():
            emb = extract_embedding_from_field(v)
            if emb is not None:
                detected_dim = int(emb.shape[0])
                break
        if detected_dim:
            break
    canonical_dim = detected_dim if detected_dim else CANON_DIM
    print(f"[info] canonical_dim = {canonical_dim}")

    A = prepare_profiles(rawA, a_name, canonical_dim)
    B = prepare_profiles(rawB, b_name, canonical_dim)
    print(f"[info] prepared A={len(A)} B={len(B)}")

    # compute or load global embeddings (safe assignment)
    if use_cache and load_embeddings_cache_if_exists(A, a_name):
        print(f"[cache] loaded embeddings cache for {a_name}")
    else:
        for p in A:
            if p["emb"].get("global") is None:
                g = compute_global_embedding(p, field_weights, dim=canonical_dim)
                if g is None:
                    p["emb"]["global"] = np.zeros((canonical_dim,), dtype=np.float32)
                else:
                    p["emb"]["global"] = np.asarray(g, dtype=np.float32)
        save_embeddings_cache(A, a_name, dim=canonical_dim)
        print(f"[cache] saved embeddings for {a_name}")

    if use_cache and load_embeddings_cache_if_exists(B, b_name):
        print(f"[cache] loaded embeddings cache for {b_name}")
    else:
        for p in B:
            if p["emb"].get("global") is None:
                g = compute_global_embedding(p, field_weights, dim=canonical_dim)
                if g is None:
                    p["emb"]["global"] = np.zeros((canonical_dim,), dtype=np.float32)
                else:
                    p["emb"]["global"] = np.asarray(g, dtype=np.float32)
        save_embeddings_cache(B, b_name, dim=canonical_dim)
        print(f"[cache] saved embeddings for {b_name}")

    # build faiss index
    index, B_norm = build_faiss_index_from_prepared(B, canonical_dim)
    print(f"[info] faiss index built; B_norm shape = {B_norm.shape}")

    A_globals = np.vstack([p["emb"]["global"] for p in A]).astype(np.float32)
    # query faiss for candidates
    candidate_lists = query_faiss(index, B_norm, A_globals, top_k=faiss_k, batch_size=512)

    raw_matches = []
    for i, cand_list in enumerate(candidate_lists):
        if not cand_list:
            continue
        cand_idxs = [c[0] for c in cand_list]
        a_item = A[i]
        scored = []
        for j in cand_idxs:
            b_item = B[j]
            sc = score_pair(a_item, b_item, field_weights, fields_to_use=selected_fields)
            if sc["score"] >= threshold:
                scored.append((sc["score"], j, sc))
        scored.sort(key=lambda x: x[0], reverse=True)
        for score, j, sc in scored[:top_k]:
            raw_matches.append({
                "profileA_index": i,
                "profileA_id": (A[i]["__orig__"].get("profile_id") or A[i]["__orig__"].get("username") or A[i]["__orig__"].get("user_id")),
                "profileB_index": j,
                "profileB_id": (B[j]["__orig__"].get("profile_id") or B[j]["__orig__"].get("username") or B[j]["__orig__"].get("user_id")),
                "score": float(score),
                "per_field": sc["per_field"]
            })

    # dedupe top1 per A
    best_per_A = {}
    for m in raw_matches:
        ida = m["profileA_id"] if m["profileA_id"] is not None else m["profileA_index"]
        if ida not in best_per_A or m["score"] > best_per_A[ida]["score"]:
            best_per_A[ida] = m
    final_matches = list(best_per_A.values())

    out = OUT_TEMPLATE.format(a=a_name, b=b_name)
    save_json(final_matches, out)
    t1 = time.perf_counter()
    print(f"[done] saved {len(final_matches)} matches to {out} (time {t1-t0:.2f}s)")
    return final_matches

# ---------------------------
# Script entrypoint
# ---------------------------
if __name__ == "__main__":
    pairs_to_match = [
        ("linkedin", "github"),
        ("linkedin", "twitter"),
        ("github", "twitter"),
        ("github", "linkedin"),
        ("twitter", "github"),
        ("twitter", "linkedin"),
    ]

    for a, b in pairs_to_match:
        cfg = PAIR_CONFIGS.get((a,b), {})
        threshold = cfg.get("threshold", THRESHOLD)
        top_k = cfg.get("top_k", TOP_K)
        field_weights = cfg.get("field_weights", FIELD_WEIGHTS)
        selected_fields = PAIR_FIELDS.get((a,b), None)
        print(f"\n=== Matching {a} -> {b} (threshold={threshold}, top_k={top_k}) ===")
        pairwise_match_faiss(a, b, selected_fields=selected_fields, threshold=threshold, top_k=top_k, field_weights=field_weights, faiss_k=100)
