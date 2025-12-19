import pickle
import random
import numpy as np
import pandas as pd
import networkx as nx


# 1. LOAD GRAPH + ECHO CHAMBER DATA

print("[+] Loading graph and echo chamber files...")

with open("graph_with_echo_hybrid_gpu.pkl", "rb") as f:
    G = pickle.load(f)

df = pd.read_csv("community_echo_stats_hybrid_gpu.csv")
echo_ids = df[df["is_echo"] == True]["community_id"].tolist()
print(f"[] Echo chambers detected: {len(echo_ids)}")


# 2. REBUILD COMMUNITY DICTIONARY

community_dict = {}
for n, data in G.nodes(data=True):
    cid = data.get("community")
    if cid is None:
        continue
    community_dict.setdefault(cid, []).append(n)

echo_comms = {cid: community_dict[cid] for cid in echo_ids}



# 3. SAFE SENTIMENT FETCHER

def safe_sentiment(n):
    """Return sentiment float or 0.0 if missing."""
    s = G.nodes[n].get("sentiment", 0.0)
    return float(s) if isinstance(s, (int, float)) else 0.0



# 4. FEED DIVERSIFICATION SIMULATION

FOREIGN_CONTENT_RATIO = 0.3  # 30% foreign content

print("[+] Performing feed diversification...")

results = []

all_nodes = list(G.nodes())

for cid, members in echo_comms.items():

    # ORIGINAL sentiments (cleaned)
    original_sentiments = [safe_sentiment(n) for n in members]

    # ORIGINAL vectors (cleaned)
    original_vectors = [
        G.nodes[n].get("context")
        for n in members
        if isinstance(G.nodes[n].get("context"), np.ndarray)
    ]

    # foreign nodes
    foreign_candidates = [n for n in all_nodes if n not in members]
    num_inject = max(1, int(len(members) * FOREIGN_CONTENT_RATIO))

    inject_nodes = random.sample(foreign_candidates, num_inject)

    # foreign sentiments (cleaned)
    foreign_sentiments = [safe_sentiment(n) for n in inject_nodes]

    # foreign vectors (cleaned)
    foreign_vectors = [
        G.nodes[n].get("context")
        for n in inject_nodes
        if isinstance(G.nodes[n].get("context"), np.ndarray)
    ]

    # combine
    new_sentiments = original_sentiments + foreign_sentiments
    new_vectors = original_vectors + foreign_vectors

    # safety guard
    if len(new_sentiments) == 0:
        continue

    # new metrics
    new_mean = float(np.mean(new_sentiments))
    new_var = float(np.var(new_sentiments))

    # new cohesion
    if len(new_vectors) >= 2:
        tensor = np.stack(new_vectors)
        norm = tensor / np.linalg.norm(tensor, axis=1, keepdims=True)
        sims = np.dot(norm, norm.T)
        upper = sims[np.triu_indices_from(sims, k=1)]
        new_cohesion = float(np.mean(upper))
    else:
        new_cohesion = 0.0

    # original cohesion
    old_cohesion = float(df[df["community_id"] == cid]["semantic_cohesion"].values[0])

    results.append({
        "community_id": cid,
        "size": len(members),
        "old_sentiment_variance": float(np.var(original_sentiments)),
        "new_sentiment_variance": new_var,
        "old_semantic_cohesion": old_cohesion,
        "new_semantic_cohesion": new_cohesion
    })

print("[] Feed diversification complete.")



# 5. SAVE RESULTS

df_out = pd.DataFrame(results)
df_out.to_csv("feed_diversification_results.csv", index=False)
print("[] Saved → feed_diversification_results.csv")



# HEATMAP: Semantic Cohesion vs Sentiment Variance

import numpy as np
import matplotlib.pyplot as plt

print("[+] Creating heatmaps for cohesion vs variance...")

# Extract data
old_var = df_out["old_sentiment_variance"].values
new_var = df_out["new_sentiment_variance"].values
old_coh = df_out["old_semantic_cohesion"].values
new_coh = df_out["new_semantic_cohesion"].values


# HEATMAP BEFORE DIVERSIFICATION

plt.figure(figsize=(7, 6))
plt.hist2d(old_coh, old_var, bins=30, cmap='viridis')
plt.colorbar(label="Density")
plt.xlabel("Semantic Cohesion (Before)")
plt.ylabel("Sentiment Variance (Before)")
plt.title("Heatmap: Cohesion vs Variance (Before Diversification)")
plt.tight_layout()
plt.savefig("heatmap_before.png", dpi=300)
plt.close()


# HEATMAP AFTER DIVERSIFICATION

plt.figure(figsize=(7, 6))
plt.hist2d(new_coh, new_var, bins=30, cmap='plasma')
plt.colorbar(label="Density")
plt.xlabel("Semantic Cohesion (After)")
plt.ylabel("Sentiment Variance (After)")
plt.title("Heatmap: Cohesion vs Variance (After Diversification)")
plt.tight_layout()
plt.savefig("heatmap_after.png", dpi=300)
plt.close()

print("[] Heatmaps saved → heatmap_before.png & heatmap_after.png")




# 6. SUMMARY

print("\n================= FEED DIVERSIFICATION SUMMARY ================")
for _, row in df_out.iterrows():
    print(f"Community {row['community_id']}:")
    print(f"    Sentiment variance BEFORE: {row['old_sentiment_variance']:.4f}")
    print(f"    Sentiment variance AFTER : {row['new_sentiment_variance']:.4f}")
    print(f"    Semantic cohesion BEFORE: {row['old_semantic_cohesion']:.4f}")
    print(f"    Semantic cohesion AFTER : {row['new_semantic_cohesion']:.4f}")
    print("--------------------------------------------------------------")
print("================================================================\n")




# 7. VISUAL BAR GRAPH COMPARISON (5–10 ECHO CHAMBERS)

import matplotlib.pyplot as plt

print("[+] Creating comparison bar graphs...")

# pick 5–10 echo chambers
sample_size = min(10, len(df_out))
df_sample = df_out.sample(sample_size, random_state=42)

# --- Plot Sentiment Variance ---
plt.figure(figsize=(12, 6))
x = np.arange(len(df_sample))
width = 0.35

plt.bar(x - width/2, df_sample["old_sentiment_variance"], width, label="Before Diversification")
plt.bar(x + width/2, df_sample["new_sentiment_variance"], width, label="After Diversification")

plt.xticks(x, df_sample["community_id"], rotation=45)
plt.xlabel("Community ID")
plt.ylabel("Sentiment Variance")
plt.title("Sentiment Variance Comparison (Before vs After Diversification)")
plt.legend()
plt.tight_layout()
plt.savefig("variance_comparison.png", dpi=300)
plt.close()

# --- Plot Semantic Cohesion ---
plt.figure(figsize=(12, 6))
x = np.arange(len(df_sample))
width = 0.35

plt.bar(x - width/2, df_sample["old_semantic_cohesion"], width, label="Before Diversification")
plt.bar(x + width/2, df_sample["new_semantic_cohesion"], width, label="After Diversification")

plt.xticks(x, df_sample["community_id"], rotation=45)
plt.xlabel("Community ID")
plt.ylabel("Semantic Cohesion")
plt.title("Semantic Cohesion Comparison (Before vs After Diversification)")
plt.legend()
plt.tight_layout()
plt.savefig("cohesion_comparison.png", dpi=300)
plt.close()

print("[] Saved bar graphs → variance_comparison.png & cohesion_comparison.png")


print("[ SUCCESS] Feed diversification simulation complete!")


