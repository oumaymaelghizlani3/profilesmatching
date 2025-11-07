# compare_runs.py
import os
import json
import argparse
from collections import defaultdict
from typing import Dict, Set, List, Tuple

# Adjust these to your environment if needed
EMBED_FILES = {
    "linkedin": "linkedin_profiles_normalized.json",
    "twitter":  "twitter_data_cleaned.json",
    "github":   "github_cleaned.json"
}

PAIRS = [
    ("github","linkedin"),
    ("github","twitter"),
    ("linkedin","github"),
    ("linkedin","twitter"),
    ("twitter","github"),
    ("twitter","linkedin")
]

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_match_files(folder: str):
    files = {}
    if not os.path.isdir(folder):
        return files
    for fname in os.listdir(folder):
        if fname.startswith("singlevec_matches_") or fname.startswith("matches_"):
            # accept both naming schemes
            base = fname.replace(".json","")
            # attempt to extract A and B
            parts = base.split("_")
            # find the two source names heuristically
            src_pair = None
            for i in range(len(parts)-1):
                a = parts[i]
                b = parts[i+1]
                if a in EMBED_FILES and b in EMBED_FILES:
                    src_pair = (a,b)
                    break
            if src_pair:
                files[src_pair] = os.path.join(folder, fname)
    return files

def extract_pair_ids(matchlist):
    A_ids = set()
    B_ids = set()
    pair_set = set()
    for m in matchlist:
        a = m.get("profileA_id")
        b = m.get("profileB_id")
        if a is None or b is None:
            continue
        A_ids.add(a)
        B_ids.add(b)
        pair_set.add((str(a), str(b)))
    return A_ids, B_ids, pair_set

def summarize_folder(folder: str):
    files = find_match_files(folder)
    stats = {}
    details = {}
    for pair in PAIRS:
        a,b = pair
        path = files.get((a,b))
        matches = []
        if path and os.path.exists(path):
            matches = load_json(path)
        # load raw counts
        total_in_A = 0
        if os.path.exists(EMBED_FILES[a]):
            total_in_A = len(load_json(EMBED_FILES[a]))
        total_in_B = 0
        if os.path.exists(EMBED_FILES[b]):
            total_in_B = len(load_json(EMED_FILES_PATH := EMBED_FILES[b]) if False else load_json(EMBED_FILES[b])) and len(load_json(EMBED_FILES[b])) or 0
            # previous line kept short-circuit to avoid lint warnings; simpler assignment:
            total_in_B = len(load_json(EMBED_FILES[b])) if os.path.exists(EMBED_FILES[b]) else 0

        A_ids, B_ids, pair_set = extract_pair_ids(matches)
        stats[(a,b)] = {
            "total_in_A": total_in_A,
            "total_in_B": total_in_B,
            "matched_from_A_to_B": len(matches),
            "perc_matched_from_A": round(100.0 * len(matches) / total_in_A, 2) if total_in_A else 0.0,
        }
        details[(a,b)] = {
            "A_ids": A_ids,
            "B_ids": B_ids,
            "pairs": pair_set,
            "examples": matches[:10]
        }
    return stats, details

def compute_comparison(stats_before, details_before, stats_after, details_after):
    summary = {
        "pairs": {}
    }
    mutual_examples = {}
    for pair in PAIRS:
        a,b = pair
        sb = stats_before.get((a,b), {})
        sa = stats_after.get((a,b), {})
        db = details_before.get((a,b), {"A_ids":set(), "B_ids":set(), "pairs":set()})
        da = details_after.get((a,b), {"A_ids":set(), "B_ids":set(), "pairs":set()})

        # reverse pair details
        rev_b = details_before.get((b,a), {"A_ids":set(), "B_ids":set(), "pairs":set()})
        rev_a = details_after.get((b,a), {"A_ids":set(), "B_ids":set(), "pairs":set()})

        mutual_before = db["pairs"] & set((x[1], x[0]) for x in rev_b["pairs"])
        mutual_after = da["pairs"] & set((x[1], x[0]) for x in rev_a["pairs"])

        # counts for cross-agreement of ids (A ids in rev B etc)
        common_A_in_revB_before = len(db["A_ids"] & rev_b["B_ids"])
        common_B_in_revA_before = len(db["B_ids"] & rev_b["A_ids"])
        common_A_in_revB_after = len(da["A_ids"] & rev_a["B_ids"])
        common_B_in_revA_after = len(da["B_ids"] & rev_a["A_ids"])

        delta_matches = sa.get("matched_from_A_to_B",0) - sb.get("matched_from_A_to_B",0)

        summary["pairs"][f"{a}->{b}"] = {
            "before": sb,
            "after": sa,
            "delta_matched": delta_matches,
            "mutual_pairs_before": len(mutual_before),
            "mutual_pairs_after": len(mutual_after),
            "common_A_in_revB_before": common_A_in_revB_before,
            "common_B_in_revA_before": common_B_in_revA_before,
            "common_A_in_revB_after": common_A_in_revB_after,
            "common_B_in_revA_after": common_B_in_revA_after,
            "example_pairs_before": list(db["examples"])[:5] if db.get("examples") else [],
            "example_pairs_after": list(da["examples"])[:5] if da.get("examples") else []
        }
        # collect a few mutual examples
        mutual_examples[f"{a}->{b}"] = {
            "examples_before": list(mutual_before)[:5],
            "examples_after": list(mutual_after)[:5]
        }

    return summary, mutual_examples

def write_report(summary, mutual_examples, out_md="matching_report.md", out_json="comparison_summary.json"):
    # JSON
    json.dump(summary, open(out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # Markdown
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Matching Comparison Report\n\n")
        f.write("This report compares two matching runs (BEFORE vs AFTER). It reports counts, percentages, deltas and a few examples.\n\n")

        for pair, info in summary["pairs"].items():
            f.write(f"## Pair: {pair}\n\n")
            f.write(f"- Before: {info['before']['matched_from_A_to_B']}/{info['before']['total_in_A']} ({info['before']['perc_matched_from_A']}%)\n")
            f.write(f"- After : {info['after']['matched_from_A_to_B']}/{info['after']['total_in_A']} ({info['after']['perc_matched_from_A']}%)\n")
            f.write(f"- Delta (after - before): {info['delta_matched']}\n")
            f.write(f"- Mutual pairs (before): {info['mutual_pairs_before']}\n")
            f.write(f"- Mutual pairs (after): {info['mutual_pairs_after']}\n")
            f.write(f"- Common A ids in reverse B (before): {info['common_A_in_revB_before']}\n")
            f.write(f"- Common B ids in reverse A (before): {info['common_B_in_revA_before']}\n")
            f.write("\nExamples (before):\n\n")
            for ex in info.get("example_pairs_before",[]):
                f.write(f"```\n{ex}\n```\n")
            f.write("\nExamples (after):\n\n")
            for ex in info.get("example_pairs_after",[]):
                f.write(f"```\n{ex}\n```\n")
            f.write("\n---\n\n")

        f.write("\n## Mutual examples (sample)\n\n")
        for pair, ex in mutual_examples.items():
            f.write(f"### {pair}\n")
            f.write(f"- examples before: {ex['examples_before']}\n")
            f.write(f"- examples after : {ex['examples_after']}\n\n")

    print(f"Wrote report {out_md} and summary {out_json}")

def main(before_folder, after_folder):
    stats_before, details_before = summarize_folder(before_folder)
    stats_after, details_after = summarize_folder(after_folder)
    summary, mutual_examples = compute_comparison(stats_before, details_before, stats_after, details_after)
    write_report(summary, mutual_examples)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", default="results", help="folder for before run")
    parser.add_argument("--after", default="results2", help="folder for after run")
    args = parser.parse_args()
    main(args.before, args.after)
