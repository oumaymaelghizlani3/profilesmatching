import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt

# === Configuration des fichiers ===
EMBED_FILES = {
    "linkedin": "linkedin_profiles_normalized.json",
    "twitter":  "twitter_data_cleaned.json",
    "github":   "github_cleaned.json"
}
RESULTS_FOLDER = "results"

# === Chargement des données ===
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# === Analyse des résultats ===
def analyze_all_results(results_folder):
    files = [os.path.join(results_folder, f) for f in os.listdir(results_folder) if f.endswith(".json")]
    stats, comparisons = {}, {}

    for file in files:
        base = os.path.basename(file).replace("matches_", "").replace("_mapped_top1.json", "")
        a_name, b_name = base.split("_")

        matches = load_json(file)
        rawA, rawB = load_json(EMBED_FILES[a_name]), load_json(EMBED_FILES[b_name])
        n_A_total, n_B_total, n_matched = len(rawA), len(rawB), len(matches)
        perc_matched = 100 * n_matched / n_A_total if n_A_total else 0

        stats[(a_name, b_name)] = {
            "total_in_A": n_A_total,
            "total_in_B": n_B_total,
            "matched_from_A_to_B": n_matched,
            "perc_matched_from_A": round(perc_matched, 2)
        }

        comparisons[(a_name, b_name)] = {
            "matched_A_ids": {m["profileA_id"] for m in matches},
            "matched_B_ids": {m["profileB_id"] for m in matches}
        }

    reverse_info = {}
    for (a, b), comp in comparisons.items():
        rev = comparisons.get((b, a))
        if rev:
            reverse_info[(a, b)] = {
                "common_A_in_revB": len(comp["matched_A_ids"] & rev["matched_B_ids"]),
                "common_B_in_revA": len(comp["matched_B_ids"] & rev["matched_A_ids"])
            }

    sources = list(EMBED_FILES.keys())
    matched_ids = {s: set() for s in sources}
    for (a, b), comp in comparisons.items():
        matched_ids[a].update(comp["matched_A_ids"])
        matched_ids[b].update(comp["matched_B_ids"])
    triple_matches = matched_ids[sources[0]] & matched_ids[sources[1]] & matched_ids[sources[2]]

    return stats, reverse_info, triple_matches

# === Affichage texte ===
def print_stats(stats, reverse_info, triple_matches):
    print("\n=== Matching Stats ===")
    for (a, b), info in stats.items():
        print(f"{a} -> {b}: matched {info['matched_from_A_to_B']}/{info['total_in_A']} "
              f"({info['perc_matched_from_A']}%)")

    print("\n=== Reverse Comparison ===")
    for (a, b), info in reverse_info.items():
        print(f"{a} -> {b} vs {b} -> {a}: common A ids in rev B: {info['common_A_in_revB']}, "
              f"common B ids in rev A: {info['common_B_in_revA']}")

    print("\n=== Triple Matches Across All Sources ===")
    print(f"Number of profiles matched in all 3 sources: {len(triple_matches)}")
    if triple_matches:
        print(f"Example IDs: {list(triple_matches)[:10]}")

# === Visualisations ===
def plot_results(stats, reverse_info, triple_matches):
    # 1️⃣ Bar plot : pourcentage de matching
    pairs = [f"{a}->{b}" for (a, b) in stats.keys()]
    percentages = [v["perc_matched_from_A"] for v in stats.values()]

    plt.figure(figsize=(10, 5))
    plt.bar(pairs, percentages)
    plt.title("Taux de Matching par Paire de Plateformes")
    plt.xlabel("Paire de Plateformes")
    plt.ylabel("Pourcentage de Profils Matchés (%)")
    plt.xticks(rotation=30)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("matching_rates.png", dpi=300)
    plt.show()

    # 2️⃣ Heatmap simplifiée de réciprocité
    import numpy as np

    platforms = list(EMBED_FILES.keys())
    matrix = np.zeros((len(platforms), len(platforms)))

    for i, a in enumerate(platforms):
        for j, b in enumerate(platforms):
            if (a, b) in stats:
                matrix[i, j] = stats[(a, b)]["perc_matched_from_A"]

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues", interpolation="nearest")
    plt.title("Heatmap des Pourcentages de Matching")
    plt.xticks(range(len(platforms)), platforms)
    plt.yticks(range(len(platforms)), platforms)
    for i in range(len(platforms)):
        for j in range(len(platforms)):
            plt.text(j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center", color="black")
    plt.colorbar(label="% matched")
    plt.tight_layout()
    plt.savefig("matching_heatmap.png", dpi=300)
    plt.show()

    # 3️⃣ Donut chart : triple matches
    plt.figure(figsize=(5, 5))
    total_profiles = sum(v["total_in_A"] for v in stats.values()) // len(stats)
    plt.pie(
        [len(triple_matches), total_profiles - len(triple_matches)],
        labels=["Triple Matchés", "Non Triple"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#4CAF50", "#E0E0E0"],
        wedgeprops={"width": 0.4}
    )
    plt.title("Part des Profils Matchés sur les 3 Sources")
    plt.savefig("triple_match_donut.png", dpi=300)
    plt.show()

# === Programme principal ===
if __name__ == "__main__":
    stats, reverse_info, triple_matches = analyze_all_results(RESULTS_FOLDER)
    print_stats(stats, reverse_info, triple_matches)
    plot_results(stats, reverse_info, triple_matches)
