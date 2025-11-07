# step3_full_pipeline.py
import json
import re
import os
import math
import hashlib
import struct
import random
from typing import List, Dict, Any, Optional
import numpy as np

# -------- user config (change file names if needed) ----------
EMBED_FILES = {
    "linkedin": "linkedin_profiles_normalized.json",
    "twitter":  "twitter_data_cleaned.json",
    "github":   "github_cleaned.json"
}
OUT_TEMPLATE = "results/matches_{a}_{b}_mapped_top1.json"

# field weights (tuneable)
FIELD_WEIGHTS = {
    "username": 0.60,
    "name": 0.60,
    "bio": 0.20,               # headline/bio semantic
    "repo_names": 0.1,
    "repo_descriptions": 0.1,
    "location": 0.05,
    "_default": 0.05
}

PAIR_CONFIGS = {
    ("linkedin", "github"): {"threshold": 0.7, "top_k": 5, "field_weights": FIELD_WEIGHTS},
    ("linkedin", "twitter"): {"threshold": 0.7, "top_k": 5, "field_weights": FIELD_WEIGHTS},
    ("github", "twitter"): {"threshold": 0.7, "top_k": 5, "field_weights": FIELD_WEIGHTS},
}

PAIR_FIELDS = {
    ("linkedin", "github"): ["username", "name", "bio", "repo_names", "repo_descriptions"],
    ("linkedin", "twitter"): ["username", "name", "bio"],
    ("github", "twitter"): ["username", "name", "bio"],
}


THRESHOLD = 0.6  # final weighted score threshold to consider a match (tweak)
TOP_K = 5         # keep top-K candidates before selecting top1 per A

# canonical embedding dim used when deterministic vectors are generated.
CANON_DIM = 384

# fields present in your data (adjust if different)
LIKELY_FIELDS = {
    "linkedin": ["profile_id", "username", "full_name", "headline", "about", "projects", "location"],
    "github":   ["user_id", "username", "name", "bio", "company", "email", "location", "repo_names", "repo_descriptions"],
    "twitter":  ["Username", "username", "name", "bio", "location"]
}

# ---------------------------
# Utilities: IO + text normalization
# ---------------------------
def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

_non_alnum = re.compile(r"[^0-9a-z ]+")

def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.strip().lower()
    # remove accents using a simple transliteration (keep ascii only)
    try:
        import unicodedata
        s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    except Exception:
        pass
    s = s.replace("\n", " ").replace("\r", " ")
    s = _non_alnum.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------------------------
# String similarity functions
# ---------------------------
def levenshtein(a: str, b: str) -> int:
    # classic Levenshtein (iterative, memory-optimized)
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
    # pure python Jaro-Winkler implementation
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
    # find matches
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
    # count transpositions
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
    # Jaro-Winkler
    # common prefix up to 4 chars
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
    if a.size == 0 or b.size == 0:
        return 0.0
    if a.shape != b.shape:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def deterministic_hash_vector(s: str, dim: int = CANON_DIM, seedfold: int = 0) -> np.ndarray:
    if s is None:
        s = ""
    # stable seed from sha256
    h = hashlib.sha256(s.encode("utf-8") + struct.pack("I", seedfold)).digest()
    seed = int.from_bytes(h[:8], "big")
    rnd = random.Random(seed)
    vec = np.fromiter((rnd.gauss(0,1) for _ in range(dim)), dtype=np.float32, count=dim)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-12) if norm > 0 else vec

def extract_embedding_from_field(value: Any) -> Optional[np.ndarray]:
    """
    The raw JSON may contain embedding lists or lists of lists.
    This function tries to detect and convert numeric lists into numpy arrays.
    """
    if value is None:
        return None
    # if value is already numeric list
    if isinstance(value, list) and value and all(isinstance(x, (int, float)) for x in value):
        return np.array(value, dtype=np.float32)
    # if value is a list of numeric lists, compute mean
    if isinstance(value, list) and value and isinstance(value[0], list) and all(isinstance(el, (list, tuple, np.ndarray)) for el in value):
        arrs = []
        for el in value:
            if el is None:
                continue
            if all(isinstance(x, (int, float)) for x in el):
                arrs.append(np.array(el, dtype=np.float32))
        if arrs:
            return np.vstack(arrs).mean(axis=0)
    # not an embedding
    return None

# ---------------------------
# Field-level similarity computation
# ---------------------------
def username_similarity(a_raw: Dict[str,Any], b_raw: Dict[str,Any]) -> float:
    # try canonical fields
    a_user = (a_raw.get("profile_id") or a_raw.get("username") or a_raw.get("user_id") or a_raw.get("Username"))
    b_user = (b_raw.get("profile_id") or b_raw.get("username") or b_raw.get("user_id") or b_raw.get("Username"))
    if a_user is None or b_user is None:
        return 0.0
    # both might be lists -> join
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
    # strong: jaro_winkler on normalized username
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
    # try Jaro-Winkler and normalized Levenshtein and average them
    jw = jaro_winkler(a_s, b_s)
    lv = normalized_levenshtein(a_s, b_s)
    # also account for token overlap (same family name)
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
    # safe cosine - if dims mismatch returns 0
    return cosine_sim(a_emb, b_emb)

# ---------------------------
# Build canonical per-profile representation (normalized strings + embeddings)
# ---------------------------
def prepare_profiles(raw_profiles: List[Dict[str,Any]], source_name: str, canonical_dim: int) -> List[Dict[str,Any]]:
    prepared = []
    for p in raw_profiles:
        item = {"__orig__": p, "source": source_name}
        # normalized strings
        item["norm"] = {}
        # username/name/location as normalized strings
        username_candidates = (p.get("profile_id") or p.get("username") or p.get("user_id") or p.get("Username"))
        if username_candidates is not None:
            if isinstance(username_candidates, list):
                username_candidates = " ".join(map(str, username_candidates))
            item["norm"]["username"] = normalize_text(str(username_candidates))
        # name
        name_val = p.get("full_name") or p.get("fullName") or p.get("name")
        if name_val is not None:
            if isinstance(name_val, list):
                name_val = " ".join(map(str, name_val))
            item["norm"]["name"] = normalize_text(str(name_val))
        # location
        loc = p.get("location")
        if loc is not None:
            if isinstance(loc, list):
                loc = " ".join(map(str, loc))
            item["norm"]["location"] = normalize_text(str(loc))
        # headline/bio/raw long text (concat fields that might be semantically relevant)
        bio_fields = []
        for k in ("headline", "about", "bio", "description"):
            v = p.get(k)
            if v:
                if isinstance(v, list):
                    bio_fields.append(" ".join(map(str, v)))
                else:
                    bio_fields.append(str(v))
        # also include company/company description if present
        if p.get("company"):
            bio_fields.append(str(p.get("company")))
        if p.get("repo_descriptions") and isinstance(p.get("repo_descriptions"), list):
            # join short list of repo descriptions, but only if they're strings/numeric
            joined = " ".join(str(x) for x in p.get("repo_descriptions") if isinstance(x, (str, int, float)))
            if joined:
                bio_fields.append(joined)
        # --- projects: robust extraction (list of dicts or list of strings) ---
        # --- inside prepare_profiles, after projects_text ---
        if p.get("projects") and isinstance(p.get("projects"), list):
            proj_texts = []
            for proj in p.get("projects"):
                if proj is None:
                    continue
                # project may be a dict like {"name": "...", "description": "..."}
                if isinstance(proj, dict):
                    # flexible keys
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
                # sometimes projects is already a list of strings
                elif isinstance(proj, (str, int, float)):
                    proj_texts.append(str(proj))
                # ignore other weird types
            if proj_texts:
                joined_projects = " ".join(proj_texts)
                bio_fields.append(joined_projects)
                # also add a dedicated projects normalized field (useful later)
                item["norm"]["projects_text"] = normalize_text(joined_projects)
            else:
                item["norm"]["projects_text"] = ""
            proj_text = item["norm"]["projects_text"]
            if "emb" not in item:
                item["emb"] = {}
            item["emb"]["projects"] = deterministic_hash_vector(proj_text, dim=canonical_dim) if proj_text else None

        else:
            item["norm"]["projects_text"] = ""
        
        item["norm"]["bio_text"] = normalize_text(" ".join(bio_fields)) if bio_fields else ""

        # embeddings: try to extract numeric embeddings from fields, fallback to deterministic
        item["emb"] = {}
        # check common possible embedding fields in the raw record
        # (user might already have precomputed fields like 'embedding', 'bio_embedding', etc.)
        extracted = None
        for candidate_key in ("embedding","emb","bio_embedding","vector","vectors"):
            if candidate_key in p:
                extracted = extract_embedding_from_field(p[candidate_key])
                if extracted is not None:
                    item["emb"]["global"] = extracted
                    break
        # field-level extraction
        # repo_names/repo_descriptions could be lists of strings; if not numeric embeddings, we will create deterministic vectors
        # attempt to extract field embeddings if present in JSON
        item["emb"]["bio"] = None
        if "bio" in p:
            emb = extract_embedding_from_field(p.get("bio"))
            if emb is not None:
                item["emb"]["bio"] = emb
        # try repo_descriptions
        if "repo_descriptions" in p:
            emb = extract_embedding_from_field(p.get("repo_descriptions"))
            if emb is not None:
                item["emb"]["repo_descriptions"] = emb
        # names: rarely embedded, so will use deterministic hashing
        # now ensure we have deterministic vectors for fields we will compare semantically
        if item["norm"].get("bio_text"):
            if item["emb"].get("bio") is None:
                item["emb"]["bio"] = deterministic_hash_vector(item["norm"]["bio_text"], dim=canonical_dim)
        else:
            item["emb"]["bio"] = None

        # repo_names: if present as strings -> deterministic
        rn = p.get("repo_names")
        if rn:
            if isinstance(rn, list):
                joined = " ".join(str(x) for x in rn if isinstance(x, (str, int, float)))
            else:
                joined = str(rn)
            item["emb"]["repo_names"] = deterministic_hash_vector(normalize_text(joined), dim=canonical_dim) if joined else None
        else:
            item["emb"]["repo_names"] = None

        # repo_descriptions
        if p.get("repo_descriptions"):
            # if numeric embeddings present, extract_embedding_from_field handles it above
            if item["emb"].get("repo_descriptions") is None:
                joined = " ".join(str(x) for x in p.get("repo_descriptions") if isinstance(x, (str, int, float)))
                item["emb"]["repo_descriptions"] = deterministic_hash_vector(normalize_text(joined), dim=canonical_dim) if joined else None

        # name deterministic embedding (optional)
        if item["norm"].get("name"):
            item["emb"]["name"] = deterministic_hash_vector(item["norm"]["name"], dim=canonical_dim)
        else:
            item["emb"]["name"] = None

        # keep raw original too
        prepared.append(item)
    return prepared

# ---------------------------
# Per-pair scoring using the recommended hybrid approach
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
            continue

        w = field_weights.get(field, field_weights.get("_default", 0.0))
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
        per_field[field] = {"score": s, "weight": w}
        weighted_sum += s * w
        total_weight += w

    final_score = float(weighted_sum / total_weight) if total_weight > 0 else 0.0
    return {"score": final_score, "per_field": per_field}

# ---------------------------
# Blocking / Candidate selection - simple but effective
# ---------------------------
def build_index(prepared: List[Dict[str,Any]], key_field: str = "username") -> Dict[str, List[int]]:
    """
    Build a dict mapping a blocking key -> list of indices.
    Here we use first two characters of username or name token as a block key.
    """
    idx = {}
    for i, p in enumerate(prepared):
        key = ""
        if p["norm"].get(key_field):
            key = p["norm"][key_field][:2]  # first two chars
        else:
            # fallback to name
            name = p["norm"].get("name","")
            key = name[:2] if name else ""
        idx.setdefault(key, []).append(i)
    return idx

# ---------------------------
# Main pairwise function
# ---------------------------
def pairwise_match(a_name: str, b_name: str,
                   selected_fields: Optional[List[str]] = None,
                   threshold: float = THRESHOLD,
                   top_k: int = TOP_K,
                   field_weights: Dict[str,float] = FIELD_WEIGHTS,
                   show_diag: bool = True):
    a_path = EMBED_FILES[a_name]
    b_path = EMBED_FILES[b_name]
    rawA = load_json(a_path)
    rawB = load_json(b_path)
    if show_diag:
        print(f"Loaded {len(rawA)} from {a_name}, {len(rawB)} from {b_name}")
        print("sample raw keys A:", {k: type(v).__name__ for k,v in (rawA[0].items() if rawA else [])})
        if rawB:
            print("sample raw keys B:", {k: type(v).__name__ for k,v in rawB[0].items()})
    if selected_fields is None:
        selected_fields = PAIR_FIELDS.get((a_name, b_name), None)

    # quick attempt to detect canonical dim from any numeric embedding present in B
    detected_dim = None
    for p in rawB:
        for k,v in p.items():
            emb = extract_embedding_from_field(v)
            if emb is not None:
                detected_dim = emb.shape[0]
                break
        if detected_dim:
            break
    canonical_dim = detected_dim if detected_dim else CANON_DIM
    if show_diag:
        print(f"[diag] canonical embedding dim = {canonical_dim}")

    A = prepare_profiles(rawA, a_name, canonical_dim)
    B = prepare_profiles(rawB, b_name, canonical_dim)
    if show_diag and A:
        print(f"[diag] example prepared A fields: {list(A[0]['norm'].keys())}, emb keys: {list(A[0]['emb'].keys())}")
    if show_diag and B:
        print(f"[diag] example prepared B fields: {list(B[0]['norm'].keys())}, emb keys: {list(B[0]['emb'].keys())}")

    # build blocking index on B
    b_index = build_index(B, key_field="username")

    raw_matches = []
    for i, a in enumerate(A):
        # skip if no useful content
        if not (a["norm"].get("username") or a["norm"].get("name") or a["norm"].get("bio_text")):
            continue

        block_key = (a["norm"].get("username") or a["norm"].get("name") or "")[:2]
        candidates_idx = set()
        if block_key in b_index:
            candidates_idx.update(b_index[block_key])
        # also attempt more relaxed blocks: first char or empty-key
        if block_key and block_key[:1] in b_index:
            candidates_idx.update(b_index[block_key[:1]])
        # if still empty, fallback to full scan (but we try to avoid it)
        if not candidates_idx:
            candidates_idx = set(range(len(B)))

        scored = []
        for j in candidates_idx:
            b = B[j]
            sc = score_pair(a, b, field_weights, fields_to_use=selected_fields)
            if sc["score"] >= threshold:
                scored.append((sc["score"], j, sc))

        # keep top_k
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

    # keep only top match per A
    best_per_A = {}
    for m in raw_matches:
        ida = m["profileA_id"] if m["profileA_id"] is not None else m["profileA_index"]
        if ida not in best_per_A or m["score"] > best_per_A[ida]["score"]:
            best_per_A[ida] = m
    final_matches = list(best_per_A.values())

    if show_diag:
        print(f"[diag] matches considered: {len(raw_matches)}; final top1: {len(final_matches)}")

    out = OUT_TEMPLATE.format(a=a_name, b=b_name)
    save_json(final_matches, out)
    if show_diag:
        print(f"Saved {len(final_matches)} matches to {out}")
    return final_matches


# ---------------------------
# run as script
# ---------------------------
if __name__ == "__main__":
    pairs_to_match = [
        ("linkedin", "github"),
        ("linkedin", "twitter"),
        ("github", "twitter"),
    ]

    for a, b in pairs_to_match:
        config = PAIR_CONFIGS.get((a,b), {})
        threshold = config.get("threshold", THRESHOLD)
        top_k = config.get("top_k", TOP_K)
        field_weights = config.get("field_weights", FIELD_WEIGHTS)
        fields_to_use = PAIR_FIELDS.get((a,b), None)

        pairwise_match(a, b, threshold=threshold, top_k=top_k, field_weights=field_weights, selected_fields=fields_to_use)