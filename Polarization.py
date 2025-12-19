import pickle
import pandas as pd
import numpy as np
import networkx as nx

# ------------------------------------------------------------
# 1. LOAD GRAPH + CSV
# ------------------------------------------------------------
print("[+] Loading hybrid echo graph...")
with open("graph_with_echo_hybrid_gpu.pkl", "rb") as f:
    G = pickle.load(f)

print("[+] Loading community CSV...")
df = pd.read_csv("community_echo_stats_hybrid_gpu.csv")

echo_ids = df[df["is_echo"] == True]["community_id"].tolist()
print(f"[✔] Total echo chambers detected: {len(echo_ids)}")

# ------------------------------------------------------------
# 2. RECONSTRUCT community_dict
# ------------------------------------------------------------
print("[+] Reconstructing community_dict from graph...")

community_dict = {}
for n, data in G.nodes(data=True):
    cid = data.get("community")
    if cid is None:
        continue
    community_dict.setdefault(cid, []).append(n)

print(f"[✔] Total communities found in graph: {len(community_dict)}")

# ------------------------------------------------------------
# 3. COMPUTE POLARITY FOR ALL ECHO CHAMBERS
# ------------------------------------------------------------
print("[+] Computing polarity summary for ALL echo chambers...")

community_stats = []

for cid in echo_ids:
    nodes = community_dict.get(cid, [])

    # Get sentiments but REMOVE None values
    sentiments = [
        (G.nodes[n].get("sentiment") if isinstance(G.nodes[n].get("sentiment"), (int, float)) else 0.0)
        for n in nodes
    ]

    if len(sentiments) == 0:
        continue

    mean_pol = float(np.mean(sentiments))
    var_pol  = float(np.var(sentiments))

    community_stats.append({
        "community_id": cid,
        "size": len(nodes),
        "mean_tweet_polarity": mean_pol,
        "polarity_variance": var_pol
    })

# ------------------------------------------------------------
# 4. SAVE + PRINT ALL RESULTS
# ------------------------------------------------------------
df_comm = pd.DataFrame(community_stats)
df_comm = df_comm.sort_values(by="size", ascending=False)

pd.set_option("display.max_rows", None)
print("\n[📊] FULL Polarity Summary for ALL Echo Chambers:")
print(df_comm)

df_comm.to_csv("all_echo_communities_polarity.csv", index=False)
print("\n[💾] Saved → all_echo_communities_polarity.csv")
