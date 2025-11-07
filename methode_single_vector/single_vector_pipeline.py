# single_vector_pipeline.py
"""
Pipeline: encode each profile as ONE canonical vector (SBERT) then match by cosine similarity.

Outputs:
 - cached embeddings: embeddings/{source}_embeddings.npy
 - id mapping: embeddings/{source}_ids.json
 - matches: results/singlevec_matches_{A}_{B}.json
 - summary stats printed to console
"""

import os
import json
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

# -------- user config ----------
EMBED_FILES = {
    "linkedin": "../final_pipline/linkedin_profiles_normalized.json",
    "twitter":  "../final_pipline/twitter_data_cleaned.json",
    "github":   "../final_pipline/github_cleaned.json"
}

EMBED_CACHE_DIR = "embeddings"
RESULTS_DIR = "results_singlevec"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
TOP_K = 5
THRESHOLD = 0.65  # cosine threshold (0..1)

os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# initialize model
MODEL = SentenceTransformer(MODEL_NAME)

# ---------------- IO ----------------
def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ---------------- normalize and build canonical text ----------------
_non_alnum = __import__("re").compile(r"[^0-9a-z ]+")

def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    try:
        import unicodedata
        s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    except Exception:
        pass
    s = s.replace("\n", " ").replace("\r", " ")
    s = _non_alnum.sub(" ", s)
    s = __import__("re").sub(r"\s+", " ", s).strip()
    return s

def build_canonical_text(profile: Dict[str, Any]) -> str:
    """
    Choose and concatenate relevant fields into one canonical text:
    username, name, headline/bio/about, company, projects, repo_names, repo_descriptions, location, email.
    """
    parts = []
    # username / id
    for k in ("profile_id","user_id","username","Username"):
        v = profile.get(k)
        if v:
            if isinstance(v, list):
                v = " ".join(map(str,v))
            parts.append(str(v))
            break
    # name
    for k in ("full_name","fullName","name"):
        v = profile.get(k)
        if v:
            parts.append(str(v))
            break
    # headline / bio / about / description
    for k in ("headline","about","bio","description"):
        v = profile.get(k)
        if v:
            if isinstance(v, list):
                parts.append(" ".join(map(str,v)))
            else:
                parts.append(str(v))
    # company / organization
    if profile.get("company"):
        parts.append(str(profile.get("company")))
    # email
    if profile.get("email"):
        parts.append(str(profile.get("email")))
    # projects (list of dicts or strings)
    if profile.get("projects"):
        pr = profile.get("projects")
        if isinstance(pr, list):
            tmp = []
            for item in pr:
                if isinstance(item, dict):
                    title = item.get("name") or item.get("title") or item.get("project_name") or ""
                    desc  = item.get("description") or item.get("details") or ""
                    if title: tmp.append(str(title))
                    if desc: tmp.append(str(desc))
                elif isinstance(item, (str,int,float)):
                    tmp.append(str(item))
            if tmp:
                parts.append(" ".join(tmp))
    # repo names / descriptions
    if profile.get("repo_names"):
        rn = profile.get("repo_names")
        if isinstance(rn, list):
            parts.append(" ".join(str(x) for x in rn if x))
        else:
            parts.append(str(rn))
    if profile.get("repo_descriptions"):
        rd = profile.get("repo_descriptions")
        if isinstance(rd, list):
            parts.append(" ".join(str(x) for x in rd if x))
        else:
            parts.append(str(rd))
    # location
    if profile.get("location"):
        loc = profile.get("location")
        if isinstance(loc, list):
            parts.append(" ".join(str(x) for x in loc))
        else:
            parts.append(str(loc))
    # fallback: profile_url or external_links
    if profile.get("profile_url"):
        parts.append(str(profile.get("profile_url")))
    if profile.get("external_links"):
        try:
            if isinstance(profile["external_links"], dict):
                parts.append(" ".join(str(x) for x in profile["external_links"].values()))
            else:
                parts.append(str(profile["external_links"]))
        except Exception:
            pass

    combined = " ".join(parts)
    return normalize_text(combined)

# ---------------- batch encode / caching ----------------
def embed_profiles_once(source: str, force: bool = False) -> Tuple[np.ndarray, List[str]]:
    """
    For a source name (linkedin/twitter/github), compute or load cached embeddings.
    Returns (emb_array (N x D), ids_list)
    """
    src_file = EMBED_FILES[source]
    cache_vec_file = os.path.join(EMBED_CACHE_DIR, f"{source}_embeddings.npy")
    cache_ids_file = os.path.join(EMBED_CACHE_DIR, f"{source}_ids.json")

    # if cached and not force -> load
    if (not force) and os.path.exists(cache_vec_file) and os.path.exists(cache_ids_file):
        print(f"Loading cached embeddings for {source} from {cache_vec_file}")
        embs = np.load(cache_vec_file)
        ids = json.load(open(cache_ids_file, "r", encoding="utf-8"))
        return embs, ids

    print(f"Encoding profiles for {source} and caching to {cache_vec_file}")
    profiles = load_json(src_file)
    texts = []
    ids = []
    for p in profiles:
        # build canonical id (prefer profile_id/user_id/username)
        id_val = p.get("profile_id") or p.get("user_id") or p.get("username") or p.get("Username")
        if isinstance(id_val, list):
            id_val = " ".join(map(str, id_val))
        if id_val is None:
            # fallback to index-based id
            id_val = f"idx_{len(ids)}"
        ids.append(str(id_val))
        texts.append(build_canonical_text(p))

    # batch encode using SBERT
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        encoded = MODEL.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(encoded.astype(np.float32))
    if embeddings:
        embs = np.vstack(embeddings)
    else:
        embs = np.zeros((0, MODEL.get_sentence_embedding_dimension()), dtype=np.float32)

    # normalize vectors to unit length for cosine via dot product
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms

    # save cache
    np.save(cache_vec_file, embs)
    json.dump(ids, open(cache_ids_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Saved {len(ids)} embeddings for {source}")
    return embs, ids

# ---------------- matching (matrix) ----------------
def match_A_to_B(sourceA: str, sourceB: str, top_k: int = TOP_K, threshold: float = THRESHOLD, force_embed: bool = False) -> List[Dict[str,Any]]:
    A_embs, A_ids = embed_profiles_once(sourceA, force=force_embed)
    B_embs, B_ids = embed_profiles_once(sourceB, force=force_embed)

    if A_embs.shape[0] == 0 or B_embs.shape[0] == 0:
        return []

    # dot product matrix = cosine because vectors normalized
    # compute in chunks if memory heavy
    results = []
    CHUNK = 512
    for i in range(0, A_embs.shape[0], CHUNK):
        A_chunk = A_embs[i:i+CHUNK]  # (m, d)
        sims = np.dot(A_chunk, B_embs.T)  # (m, n)
        # for each row, get top_k indices and scores
        for row_idx in range(sims.shape[0]):
            sims_row = sims[row_idx]
            # find indices above threshold
            cand_idx = np.where(sims_row >= threshold)[0]
            if cand_idx.size == 0:
                continue
            # sort top by score
            top_idx_sorted = cand_idx[np.argsort(sims_row[cand_idx])[::-1]]
            top_idx_sorted = top_idx_sorted[:top_k]
            for j in top_idx_sorted:
                score = float(sims_row[j])
                results.append({
                    "profileA_index": i + row_idx,
                    "profileA_id": A_ids[i + row_idx],
                    "profileB_index": int(j),
                    "profileB_id": B_ids[j],
                    "score": score
                })
    # keep only best per A (highest score)
    best = {}
    for r in results:
        ida = r["profileA_id"]
        if ida not in best or r["score"] > best[ida]["score"]:
            best[ida] = r
    final = list(best.values())

    out_file = os.path.join(RESULTS_DIR, f"singlevec_matches_{sourceA}_{sourceB}.json")
    save_json(final, out_file)
    print(f"Saved {len(final)} matches to {out_file} (threshold={threshold}, top_k={top_k})")
    return final

# ---------------- utilities and stats ----------------
def stats_for_matchlist(matchlist: List[Dict[str,Any]], total_in_A: int) -> Dict[str, Any]:
    matched = len(matchlist)
    return {
        "matched_count": matched,
        "total_in_A": total_in_A,
        "perc_matched": round(100.0 * matched / total_in_A, 2) if total_in_A > 0 else 0.0
    }

def analyze_pair(sourceA: str, sourceB: str, top_k: int = TOP_K, threshold: float = THRESHOLD) -> Dict[str,Any]:
    rawA = load_json(EMBED_FILES[sourceA])
    rawB = load_json(EMBED_FILES[sourceB])
    matchesAB = match_A_to_B(sourceA, sourceB, top_k=top_k, threshold=threshold)
    matchesBA = match_A_to_B(sourceB, sourceA, top_k=top_k, threshold=threshold)  # for reverse comparison

    statsAB = stats_for_matchlist(matchesAB, len(rawA))
    statsBA = stats_for_matchlist(matchesBA, len(rawB))

    # sets for cross-check
    A_ids_AB = set(m["profileA_id"] for m in matchesAB)
    B_ids_AB = set(m["profileB_id"] for m in matchesAB)
    A_ids_BA = set(m["profileB_id"] for m in matchesBA)  # careful: matchesBA stores profileA_id from B, profileB_id from A
    B_ids_BA = set(m["profileA_id"] for m in matchesBA)

    # count common where A->B and B->A agree (i.e., same pair appears in either direction)
    # Build unordered pair keys for AB and for BA
    ab_pairs = set(( (m["profileA_id"], m["profileB_id"]) for m in matchesAB ))
    ba_pairs = set(( (m["profileB_id"], m["profileA_id"]) for m in matchesBA ))  # flipped
    mutual_pairs = ab_pairs & ba_pairs

    return {
        "A": sourceA, "B": sourceB,
        "statsA_to_B": statsAB,
        "statsB_to_A": statsBA,
        "mutual_pair_count": len(mutual_pairs),
        "example_mutual_pairs": list(mutual_pairs)[:10]
    }

# ---------------- main entry ----------------
def run_all_pairs(pairs: Optional[List[Tuple[str,str]]] = None):
    if pairs is None:
        pairs = [
            ("linkedin","github"),
            ("linkedin","twitter"),
            ("github","twitter"),
        ]
    summary = {}
    for a,b in pairs:
        print("\n=== Analyzing pair:", a, "->", b, "===\n")
        res = analyze_pair(a,b, top_k=TOP_K, threshold=THRESHOLD)
        summary[(a,b)] = res
        # basic print
        print(f"{a} -> {b}: matched {res['statsA_to_B']['matched_count']}/{res['statsA_to_B']['total_in_A']} ({res['statsA_to_B']['perc_matched']}%)")
        print(f"{b} -> {a}: matched {res['statsB_to_A']['matched_count']}/{res['statsB_to_A']['total_in_A']} ({res['statsB_to_A']['perc_matched']}%)")
        print(f"Mutual (A->B and B->A) pair count: {res['mutual_pair_count']}")
        if res['example_mutual_pairs']:
            print("Example mutual pairs:", res['example_mutual_pairs'][:5])
    # save summary
    summary_out = os.path.join(RESULTS_DIR, "singlevec_summary.json")
    # convert keys to string for JSON
    summary_jsonifiable = {f"{a}->{b}": v for (a,b), v in summary.items()}
    save_json(summary_jsonifiable, summary_out)
    print("\nSaved summary to", summary_out)
    return summary

if __name__ == "__main__":
    run_all_pairs()
