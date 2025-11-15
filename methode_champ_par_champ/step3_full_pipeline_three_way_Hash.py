# step3_three_way_only.py
import json
import re
import os
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
THREE_OUT_TEMPLATE = "matches_{a}_{b}_{c}_top_cosine.json"

# ensure results dir exists
os.makedirs("results", exist_ok=True)

# field weights (tuneable)
FIELD_WEIGHTS = {
    "username": 0.60,
    "name": 0.60,
    "bio": 0.20,               # headline/bio semantic
    "repo_names": 0.10,
    "repo_descriptions": 0.10,
    "location": 0.05,
    "_default": 0.05
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
    if isinstance(value, list) and value and isinstance(value[0], list) and all(isinstance(el, (list, tuple, np.ndarray)) for el in value):
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
# Field-level similarity computation
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
# Build canonical per-profile representation (normalized strings + embeddings)
# ---------------------------
def prepare_profiles(raw_profiles: List[Dict[str,Any]], source_name: str, canonical_dim: int) -> List[Dict[str,Any]]:
    prepared = []
    for p in raw_profiles:
        item = {"__orig__": p, "source": source_name}
        item["norm"] = {}
        username_candidates = (p.get("profile_id") or p.get("username") or p.get("user_id") or p.get("Username"))
        if username_candidates is not None:
            if isinstance(username_candidates, list):
                username_candidates = " ".join(map(str, username_candidates))
            item["norm"]["username"] = normalize_text(str(username_candidates))
        name_val = p.get("full_name") or p.get("fullName") or p.get("name")
        if name_val is not None:
            if isinstance(name_val, list):
                name_val = " ".join(map(str, name_val))
            item["norm"]["name"] = normalize_text(str(name_val))
        loc = p.get("location")
        if loc is not None:
            if isinstance(loc, list):
                loc = " ".join(map(str, loc))
            item["norm"]["location"] = normalize_text(str(loc))

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
                bio_fields.append(joined_projects)
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

        item["emb"] = {}
        extracted = None
        for candidate_key in ("embedding","emb","bio_embedding","vector","vectors"):
            if candidate_key in p:
                extracted = extract_embedding_from_field(p[candidate_key])
                if extracted is not None:
                    item["emb"]["global"] = extracted
                    break

        item["emb"]["bio"] = None
        if "bio" in p:
            emb = extract_embedding_from_field(p.get("bio"))
            if emb is not None:
                item["emb"]["bio"] = emb
        if "repo_descriptions" in p:
            emb = extract_embedding_from_field(p.get("repo_descriptions"))
            if emb is not None:
                item["emb"]["repo_descriptions"] = emb

        if item["norm"].get("bio_text"):
            if item["emb"].get("bio") is None:
                item["emb"]["bio"] = deterministic_hash_vector(item["norm"]["bio_text"], dim=canonical_dim)
        else:
            item["emb"]["bio"] = None

        rn = p.get("repo_names")
        if rn:
            if isinstance(rn, list):
                joined = " ".join(str(x) for x in rn if isinstance(x, (str, int, float)))
            else:
                joined = str(rn)
            item["emb"]["repo_names"] = deterministic_hash_vector(normalize_text(joined), dim=canonical_dim) if joined else None
        else:
            item["emb"]["repo_names"] = None

        if p.get("repo_descriptions"):
            if item["emb"].get("repo_descriptions") is None:
                joined = " ".join(str(x) for x in p.get("repo_descriptions") if isinstance(x, (str, int, float)))
                item["emb"]["repo_descriptions"] = deterministic_hash_vector(normalize_text(joined), dim=canonical_dim) if joined else None

        if item["norm"].get("name"):
            item["emb"]["name"] = deterministic_hash_vector(item["norm"]["name"], dim=canonical_dim)
        else:
            item["emb"]["name"] = None

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
    idx = {}
    for i, p in enumerate(prepared):
        key = ""
        if p["norm"].get(key_field):
            key = p["norm"][key_field][:2]
        else:
            name = p["norm"].get("name","")
            key = name[:2] if name else ""
        idx.setdefault(key, []).append(i)
    return idx

# ---------------------------
# Three-way matching (triplets) ONLY
# ---------------------------
def three_way_match(a_name: str, b_name: str, c_name: str,
                    selected_fields: Optional[List[str]] = None,
                    threshold: float = THRESHOLD,
                    top_k: int = TOP_K,
                    field_weights: Dict[str,float] = FIELD_WEIGHTS,
                    max_candidates_per_side: int = 50,
                    show_diag: bool = True):
    """
    Perform 3-way matching between datasets A, B, C.
    Combined triplet score = mean(score_ab, score_ac, score_bc).
    Only this function is executed by the script.
    """

    a_path = EMBED_FILES[a_name]
    b_path = EMBED_FILES[b_name]
    c_path = EMBED_FILES[c_name]

    rawA = load_json(a_path)
    rawB = load_json(b_path)
    rawC = load_json(c_path)
    if show_diag:
        print(f"Loaded {len(rawA)} from {a_name}, {len(rawB)} from {b_name}, {len(rawC)} from {c_name}")

    if selected_fields is None:
        selected_fields = list(set(
            (PAIR_FIELDS.get((a_name,b_name), []) or []) +
            (PAIR_FIELDS.get((a_name,c_name), []) or []) +
            (PAIR_FIELDS.get((b_name,c_name), []) or [])
        ))

    # detect canonical dim (try B then C)
    detected_dim = None
    for p in rawB + rawC:
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
    C = prepare_profiles(rawC, c_name, canonical_dim)

    if show_diag and A:
        print(f"[diag] example prepared A fields: {list(A[0]['norm'].keys())}, emb keys: {list(A[0]['emb'].keys())}")

    # build blocking indices
    b_index = build_index(B, key_field="username")
    c_index = build_index(C, key_field="username")

    triplet_results = []

    for i, a in enumerate(A):
        if not (a["norm"].get("username") or a["norm"].get("name") or a["norm"].get("bio_text")):
            continue

        block_key = (a["norm"].get("username") or a["norm"].get("name") or "")[:2]
        candidates_b = set()
        candidates_c = set()
        if block_key in b_index:
            candidates_b.update(b_index[block_key])
        if block_key and block_key[:1] in b_index:
            candidates_b.update(b_index[block_key[:1]])
        if block_key in c_index:
            candidates_c.update(c_index[block_key])
        if block_key and block_key[:1] in c_index:
            candidates_c.update(c_index[block_key[:1]])
        if not candidates_b:
            candidates_b = set(range(min(len(B), max_candidates_per_side)))
        if not candidates_c:
            candidates_c = set(range(min(len(C), max_candidates_per_side)))

        # preliminary scores a-b and a-c, keep top-N per side
        scored_b = []
        for j in candidates_b:
            sc_ab = score_pair(a, B[j], field_weights, fields_to_use=selected_fields)
            scored_b.append((sc_ab["score"], j, sc_ab))
        scored_b.sort(key=lambda x: x[0], reverse=True)
        top_b = scored_b[:max_candidates_per_side]

        scored_c = []
        for k in candidates_c:
            sc_ac = score_pair(a, C[k], field_weights, fields_to_use=selected_fields)
            scored_c.append((sc_ac["score"], k, sc_ac))
        scored_c.sort(key=lambda x: x[0], reverse=True)
        top_c = scored_c[:max_candidates_per_side]

        # cross product but limited by top_k on each side to avoid blowup
        for score_ab, j, sc_ab in top_b[:top_k]:
            for score_ac, k, sc_ac in top_c[:top_k]:
                sc_bc = score_pair(B[j], C[k], field_weights, fields_to_use=selected_fields)
                score_bc = sc_bc["score"]
                combined_score = float((score_ab + score_ac + score_bc) / 3.0)

                if combined_score >= threshold:
                    per_field_combined = {}
                    fields_all = set()
                    fields_all.update(sc_ab["per_field"].keys())
                    fields_all.update(sc_ac["per_field"].keys())
                    fields_all.update(sc_bc["per_field"].keys())
                    for f in fields_all:
                        vals = []
                        if f in sc_ab["per_field"]:
                            vals.append(sc_ab["per_field"][f]["score"])
                        if f in sc_ac["per_field"]:
                            vals.append(sc_ac["per_field"][f]["score"])
                        if f in sc_bc["per_field"]:
                            vals.append(sc_bc["per_field"][f]["score"])
                        per_field_combined[f] = {
                            "avg_score": sum(vals) / len(vals) if vals else 0.0,
                            "sources": {
                                "a-b": sc_ab["per_field"].get(f),
                                "a-c": sc_ac["per_field"].get(f),
                                "b-c": sc_bc["per_field"].get(f)
                            }
                        }

                    triplet_results.append({
                        "profileA_index": i,
                        "profileA_id": (A[i]["__orig__"].get("profile_id") or A[i]["__orig__"].get("username") or A[i]["__orig__"].get("user_id")),
                        "profileB_index": j,
                        "profileB_id": (B[j]["__orig__"].get("profile_id") or B[j]["__orig__"].get("username") or B[j]["__orig__"].get("user_id")),
                        "profileC_index": k,
                        "profileC_id": (C[k]["__orig__"].get("profile_id") or C[k]["__orig__"].get("username") or C[k]["__orig__"].get("user_id")),
                        "score_ab": float(score_ab),
                        "score_ac": float(score_ac),
                        "score_bc": float(score_bc),
                        "combined_score": float(combined_score),
                        "per_field_combined": per_field_combined
                    })

    # keep top1 triplet per A by combined_score
    best_per_A = {}
    for t in triplet_results:
        ida = t["profileA_id"] if t["profileA_id"] is not None else t["profileA_index"]
        if ida not in best_per_A or t["combined_score"] > best_per_A[ida]["combined_score"]:
            best_per_A[ida] = t
    final_triplets = list(best_per_A.values())

    if show_diag:
        print(f"[diag] triplets considered: {len(triplet_results)}; final top1 per A: {len(final_triplets)}")

    out = THREE_OUT_TEMPLATE.format(a=a_name, b=b_name, c=c_name)
    save_json(final_triplets, out)
    if show_diag:
        print(f"Saved {len(final_triplets)} triplets to {out}")
    return final_triplets

# ---------------------------
# run as script (three-way only)
# ---------------------------
if __name__ == "__main__":
    # Example: run only three-way matching between linkedin, github, twitter
    try:
        three_way_match("linkedin", "github", "twitter",
                        threshold=0.7,
                        top_k=5,
                        field_weights=FIELD_WEIGHTS,
                        max_candidates_per_side=100,
                        show_diag=True)
    except Exception as e:
        print(f"Error running three-way match: {e}")
