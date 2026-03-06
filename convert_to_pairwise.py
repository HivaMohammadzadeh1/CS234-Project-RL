import json
from itertools import combinations

INPUT_FILE = "video_rankings3.json"
OUTPUT_FILE = "video_rankings3_pairwise.json"

with open(INPUT_FILE) as f:
    data = json.load(f)

pairwise_data = {}

for group_key, group_info in data.items():
    group_num = group_info["group"]
    prompt = group_info["prompt"]
    rankings = group_info["rankings"]

    pairs = []
    videos = list(rankings.keys())

    for vid_a, vid_b in combinations(videos, 2):
        rank_a = rankings[vid_a]
        rank_b = rankings[vid_b]

        if rank_a < rank_b:
            winner = vid_a
            loser = vid_b
        elif rank_b < rank_a:
            winner = vid_b
            loser = vid_a
        else:
            winner = None
            loser = None

        pair = {
            "video_a": vid_a,
            "video_b": vid_b,
            "rank_a": rank_a,
            "rank_b": rank_b,
        }
        if winner is not None:
            pair["preferred"] = winner
            pair["rejected"] = loser
        else:
            pair["preferred"] = None
            pair["rejected"] = None
            pair["tie"] = True

        pairs.append(pair)

    pairwise_data[group_key] = {
        "group": group_num,
        "prompt": prompt,
        "num_pairs": len(pairs),
        "pairwise_comparisons": pairs,
    }

with open(OUTPUT_FILE, "w") as f:
    json.dump(pairwise_data, f, indent=2)

total_pairs = sum(g["num_pairs"] for g in pairwise_data.values())
total_ties = sum(
    sum(1 for p in g["pairwise_comparisons"] if p.get("tie"))
    for g in pairwise_data.values()
)
print(f"Groups: {len(pairwise_data)}")
print(f"Total pairwise comparisons: {total_pairs}")
print(f"Total ties: {total_ties}")
print(f"Saved to {OUTPUT_FILE}")
