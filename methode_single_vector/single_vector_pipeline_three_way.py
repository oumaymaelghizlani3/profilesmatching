# single_vector_pipeline_three_way.py
"""
Single-vector SBERT pipeline extended to three-way matching (triplets).
Produces:
 - cached embeddings: embeddings/{source}_embeddings.npy
 - id mapping: embeddings/{source}_ids.json
 - pairwise results (optional): results_singlevec/singlevec_matches_{A}_{B}.json
 - three-way results: results_singlevec/threeway_matches_{A}_{B}_{C}.json
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
THRESHOLD = 0.65        # threshold for considering a candidate (cosine)
MAX_CANDIDATES_PER_SIDE = 200  # max candidates to keep from each side before cross-product
TRIPLET_THRESHOLD = 0.60  # threshold on combined triplet score to keep result (tweak)

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
    # projects
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
    for idx, p in enumerate(profiles):
        id_val = p.get("profile_id") or p.get("user_id") or p.get("username") or p.get("Username")
        if isinstance(id_val, list):
            id_val = " ".join(map(str, id_val))
        if id_val is None:
            id_val = f"idx_{idx}"
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

# ---------------- utilities ----------------
def save_results(path: str, data: Any):
    save_json(data, path)

# ---------------- three-way matching ----------------
def three_way_match_singlevec(a_name: str, b_name: str, c_name: str,
                              threshold: float = THRESHOLD,
                              top_k: int = TOP_K,
                              max_candidates_per_side: int = MAX_CANDIDATES_PER_SIDE,
                              triplet_threshold: float = TRIPLET_THRESHOLD,
                              chunk_size: int = 512,
                              force_embed: bool = False,
                              show_diag: bool = True) -> List[Dict[str, Any]]:
    """
    Perform three-way matching using single canonical SBERT vector per profile.
    For each A:
      - find candidate set in B (similarity >= threshold) and keep top-N (limited)
      - find candidate set in C (similarity >= threshold) and keep top-N
      - compute pairwise score BC between every (b,c) in those candidate sets
      - combined_score = mean(score_ab, score_ac, score_bc)
      - keep triplets with combined_score >= triplet_threshold
    Return: best triplet per A (highest combined_score).
    """
    # load embeddings (cache)
    A_embs, A_ids = embed_profiles_once(a_name, force=force_embed)
    B_embs, B_ids = embed_profiles_once(b_name, force=force_embed)
    C_embs, C_ids = embed_profiles_once(c_name, force=force_embed)

    if show_diag:
        print(f"[diag] sizes -> A:{A_embs.shape}, B:{B_embs.shape}, C:{C_embs.shape}")

    if A_embs.shape[0] == 0 or B_embs.shape[0] == 0 or C_embs.shape[0] == 0:
        print("[warn] one of the sources has zero embeddings; aborting three-way match.")
        return []

    # Precompute B x C similarity matrix? careful memory: B x C could be large.
    # We'll compute BC similarities on the fly but can compute small blocks when candidate sets are small.

    triplet_results = []
    # process A in chunks to limit memory
    for a_start in range(0, A_embs.shape[0], chunk_size):
        A_chunk = A_embs[a_start:a_start+chunk_size]  # (m, d)
        # dot with B and C
        sims_B = np.dot(A_chunk, B_embs.T)  # (m, nB)
        sims_C = np.dot(A_chunk, C_embs.T)  # (m, nC)

        for local_idx in range(A_chunk.shape[0]):
            global_a_idx = a_start + local_idx
            a_id = A_ids[global_a_idx]
            rowB = sims_B[local_idx]
            rowC = sims_C[local_idx]

            # find candidate indices in B and C above threshold
            candB = np.where(rowB >= threshold)[0]
            candC = np.where(rowC >= threshold)[0]

            if candB.size == 0 or candC.size == 0:
                # if either side has no candidate above threshold, we might still consider top-k heuristics:
                # keep top_k even if below threshold (optional). For now skip.
                continue

            # sort candidates by score desc and limit
            candB_sorted = candB[np.argsort(rowB[candB])[::-1]]
            candC_sorted = candC[np.argsort(rowC[candC])[::-1]]

            candB_sorted = candB_sorted[:max_candidates_per_side]
            candC_sorted = candC_sorted[:max_candidates_per_side]

            # further limit for cross-product to top_k each for combinatorial control
            candB_for_cross = candB_sorted[:top_k]
            candC_for_cross = candC_sorted[:top_k]

            best_for_a = None  # hold best triplet for this A

            # compute BC scores for the small candidate sets by dot product
            # build small matrices
            B_small = B_embs[candB_for_cross]  # (kB, d)
            C_small = C_embs[candC_for_cross]  # (kC, d)
            # compute BC_sim matrix (kB x kC)
            BC_sim = np.dot(B_small, C_small.T) if (B_small.size and C_small.size) else np.zeros((B_small.shape[0], C_small.shape[0]))

            for ib_idx, b_idx in enumerate(candB_for_cross):
                score_ab = float(rowB[b_idx])
                for ic_idx, c_idx in enumerate(candC_for_cross):
                    score_ac = float(rowC[c_idx])
                    score_bc = float(BC_sim[ib_idx, ic_idx]) if BC_sim.size else 0.0
                    combined = (score_ab + score_ac + score_bc) / 3.0
                    if combined >= triplet_threshold:
                        candidate_trip = {
                            "profileA_index": int(global_a_idx),
                            "profileA_id": a_id,
                            "profileB_index": int(b_idx),
                            "profileB_id": B_ids[int(b_idx)],
                            "profileC_index": int(c_idx),
                            "profileC_id": C_ids[int(c_idx)],
                            "score_ab": score_ab,
                            "score_ac": score_ac,
                            "score_bc": score_bc,
                            "combined_score": float(combined)
                        }
                        # keep best combined for this A
                        if best_for_a is None or candidate_trip["combined_score"] > best_for_a["combined_score"]:
                            best_for_a = candidate_trip

            if best_for_a is not None:
                triplet_results.append(best_for_a)

    # Optional: if multiple triplets for same A (shouldn't happen by construction), keep top1
    best_per_A = {}
    for t in triplet_results:
        ida = t["profileA_id"]
        if ida not in best_per_A or t["combined_score"] > best_per_A[ida]["combined_score"]:
            best_per_A[ida] = t
    final_triplets = list(best_per_A.values())

    out_file = os.path.join(RESULTS_DIR, f"threeway_matches_{a_name}_{b_name}_{c_name}.json")
    save_results(out_file, final_triplets)
    print(f"Saved {len(final_triplets)} three-way matches to {out_file} (triplet_threshold={triplet_threshold})")
    return final_triplets

# ---------------- optional pairwise matcher (kept for debugging) ----------------
def match_A_to_B(sourceA: str, sourceB: str, top_k: int = TOP_K, threshold: float = THRESHOLD, force_embed: bool = False) -> List[Dict[str,Any]]:
    A_embs, A_ids = embed_profiles_once(sourceA, force=force_embed)
    B_embs, B_ids = embed_profiles_once(sourceB, force=force_embed)
    if A_embs.shape[0] == 0 or B_embs.shape[0] == 0:
        return []
    results = []
    CHUNK = 512
    for i in range(0, A_embs.shape[0], CHUNK):
        A_chunk = A_embs[i:i+CHUNK]
        sims = np.dot(A_chunk, B_embs.T)
        for row_idx in range(sims.shape[0]):
            sims_row = sims[row_idx]
            cand_idx = np.where(sims_row >= threshold)[0]
            if cand_idx.size == 0:
                continue
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
    best = {}
    for r in results:
        ida = r["profileA_id"]
        if ida not in best or r["score"] > best[ida]["score"]:
            best[ida] = r
    final = list(best.values())
    out_file = os.path.join(RESULTS_DIR, f"singlevec_matches_{sourceA}_{sourceB}.json")
    save_results(out_file, final)
    print(f"Saved {len(final)} pairwise matches to {out_file} (threshold={threshold})")
    return final

# ---------------- main ----------------
if __name__ == "__main__":
    # encode caches (optional: force=True to re-encode)
    print("Ensuring embeddings cached for all sources...")
    for s in EMBED_FILES.keys():
        embed_profiles_once(s, force=False)

    # Run three-way matching for linkedin, github, twitter (example)
    three_way_match_singlevec("linkedin", "github", "twitter",
                              threshold=THRESHOLD,
                              top_k=TOP_K,
                              max_candidates_per_side=MAX_CANDIDATES_PER_SIDE,
                              triplet_threshold=TRIPLET_THRESHOLD,
                              chunk_size=512,
                              force_embed=False,
                              show_diag=True)
