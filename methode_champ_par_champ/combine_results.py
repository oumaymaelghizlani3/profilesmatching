import json
import os
from collections import defaultdict, Counter

RESULTS_DIR = "results"
OUT_FILE = "stats/consolidated_matches.json"

# ajuster vos fichiers originaux pour compter les profils
EMBED_FILES = {
    "linkedin": "linkedin_profiles_normalized.json",
    "twitter":  "twitter_data_cleaned.json",
    "github":   "github_cleaned.json"
}

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def consolidate_matches(results_dir=RESULTS_DIR):
    matches_by_sourceA = defaultdict(dict)  # {sourceA_id: {sourceB: idB, sourceC: idC, ...}}
    seen_pairs = set()
    consolidated = []

    files = [f for f in os.listdir(results_dir) if f.endswith(".json") and f.startswith("matches_")]

    for f in files:
        parts = f.replace(".json","").split("_")
        try:
            idx = parts.index("matches")
            sourceA = parts[idx+1]
            sourceB = parts[idx+2]
        except ValueError:
            continue

        path = os.path.join(results_dir, f)
        matches = load_json(path)

        for m in matches:
            idA = m.get("profileA_id")
            idB = m.get("profileB_id")
            if idA is None or idB is None:
                continue

            # record multi-source matches
            if idA not in matches_by_sourceA:
                matches_by_sourceA[idA]["sourceA_id"] = idA
            matches_by_sourceA[idA][sourceB] = idB

            # prevent duplicate pair entries
            pair_key = tuple(sorted([(sourceA, idA), (sourceB, idB)]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # add base match info
            base_match = {
                "sourceA": sourceA,
                "sourceB": sourceB,
                "profileA_id": idA,
                "profileB_id": idB
            }
            consolidated.append(base_match)

    # --- add multi-source consolidated entries ---
    multi_source_entries = []
    for idA, sources in matches_by_sourceA.items():
        if len(sources) > 1:  # plus de 1 source (A + B/C...)
            entry = {"sourceA_id": idA}
            for key, val in sources.items():
                if key != "sourceA_id":
                    entry[key+"_id"] = val
            multi_source_entries.append(entry)

    # --- compute statistics ---
    total_profiles = {source: len(load_json(path)) for source, path in EMBED_FILES.items()}

    matched_counts = Counter()
    for m in consolidated:
        matched_counts[m["sourceA"]] += 1
        matched_counts[m["sourceB"]] += 1

    stats = {
        "total_profiles": total_profiles,
        "matched_counts": dict(matched_counts),
        "perc_matched": {src: round(100*matched_counts.get(src,0)/total,2) 
                         for src, total in total_profiles.items()},
        "multi_source_matches": len(multi_source_entries)
    }

    # save everything
    output = {
        "base_matches": consolidated,
        "multi_source_matches": multi_source_entries,
        "statistics": stats
    }

    save_json(output, OUT_FILE)
    print(f"Saved {len(consolidated)} base matches and {len(multi_source_entries)} multi-source matches to {OUT_FILE}")
    print("\n=== Statistics ===")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    return output

if __name__ == "__main__":
    consolidate_matches()
