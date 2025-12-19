import pickle
import networkx as nx
import numpy as np
from networkx.algorithms.community import modularity

# ------------------------------------------------------------
# 1. LOAD GRAPH WITH COMMUNITY + SENTIMENT
# ------------------------------------------------------------
print("[+] Loading graph_with_echo_hybrid_gpu.pkl ...")

with open("graph_with_echo_hybrid_gpu.pkl", "rb") as f:
    G = pickle.load(f)

print(f"[✔] Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ------------------------------------------------------------
# 2. EXTRACT COMMUNITIES FROM NODE ATTRIBUTES
# ------------------------------------------------------------
print("[+] Extracting communities...")

community_dict = {}
for n, data in G.nodes(data=True):
    cid = data.get("community")
    if cid is None:
        continue
    if cid not in community_dict:
        community_dict[cid] = []
    community_dict[cid].append(n)

communities = list(community_dict.values())

print(f"[✔] Total communities found: {len(communities)}")

# ------------------------------------------------------------
# 3. COMPUTE MODULARITY
# ------------------------------------------------------------
print("[+] Computing modularity...")
mod = modularity(G, communities, weight="weight")
print(f"[📌] Modularity: {mod:.4f}")

# ------------------------------------------------------------
# 4. COMPUTE SENTIMENT ASSORTATIVITY
# ------------------------------------------------------------
print("[+] Computing sentiment assortativity...")

try:
    sent_assort = nx.attribute_assortativity_coefficient(G, "sentiment")
except Exception as e:
    print("[!] Error computing sentiment assortativity:", e)
    sent_assort = None

print(f"[📌] Sentiment Assortativity: {sent_assort:.4f}")

# ------------------------------------------------------------
# 5. (OPTIONAL) DEGREE ASSORTATIVITY
# ------------------------------------------------------------
print("[+] Computing degree assortativity...")
deg_assort = nx.degree_assortativity_coefficient(G)
print(f"[📌] Degree Assortativity: {deg_assort:.4f}")

# ------------------------------------------------------------
# 6. COMMUNITY LEVEL POLARITY SUMMARY (MTLp)
# ------------------------------------------------------------
print("[+] Computing community-level polarity summary...")

community_stats = []
for cid, nodes in community_dict.items():
    sentiments = [G.nodes[n].get("sentiment", 0.0) for n in nodes]
    if len(sentiments) == 0:
        continue
    
    mean_pol = float(np.mean(sentiments))          # MTLp
    var_pol  = float(np.var(sentiments))           # sentiment variance
    
    community_stats.append({
        "community_id": cid,
        "size": len(nodes),
        "mean_tweet_polarity": mean_pol,
        "polarity_variance": var_pol
    })

# Print top 10 for preview
print("\n[📊] Sample Community Polarity Stats (Top 10):")
for c in community_stats[:10]:
    print(c)

print("\n[✅] Polarization metrics computed successfully!")
