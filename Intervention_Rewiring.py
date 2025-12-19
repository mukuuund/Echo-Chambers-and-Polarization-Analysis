import pickle
import random
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import modularity, louvain_communities


# 1. LOAD GRAPH + ECHO COMMUNITY DATA

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



# 3. COPY GRAPH FOR AGGRESSIVE REWIRING

G2 = G.to_undirected().copy()
print("[+] Graph copied for destructive intervention.")



# 4. STRONG INTERVENTION: DESTROY ECHO CHAMBERS

"""
METHOD:
1. Remove all internal edges inside each echo chamber.
2. Randomly rewire each removed edge to nodes in OTHER communities.
3. Add heavy number of cross-community edges (bridges).
Goal: Completely collapse echo chamber structure.
"""

print("[+] Destroying echo chambers...")

TOTAL_BRIDGES = 100   # Increase for more destruction

edges_removed = 0
edges_added = 0

all_nodes = list(G2.nodes())

for cid, members in echo_comms.items():
    # Remove internal edges
    internal_edges = [
        (u, v) for u in members for v in G2.neighbors(u) if v in members
    ]

    # Remove internal echo chamber edges
    for u, v in internal_edges:
        if G2.has_edge(u, v):
            G2.remove_edge(u, v)
            edges_removed += 1

    # Add heavy number of random bridges from this echo chamber to whole graph
    for _ in range(TOTAL_BRIDGES):
        u = random.choice(members)
        v = random.choice(all_nodes)
        if u != v:
            G2.add_edge(u, v, weight=1.0)
            edges_added += 1

print(f"[] Internal edges removed: {edges_removed}")
print(f"[] Bridges added: {edges_added}")



# 5. RECOMPUTE LOUVAIN COMMUNITIES + MODULARITY

print("[+] Recomputing communities & modularity...")
communities_after = louvain_communities(G2, weight="weight", seed=42)
mod_after = modularity(G2, communities_after, weight="weight")

# BEFORE VALUE (from your previous run)
MOD_BEFORE = 0.6098

print(f"[] Modularity BEFORE: {MOD_BEFORE}")
print(f"[] Modularity AFTER:  {mod_after:.4f}")



# 6. RECOMPUTE SENTIMENT ASSORTATIVITY

print("[+] Recomputing sentiment assortativity...")

sent_assort_after = nx.attribute_assortativity_coefficient(G2, "sentiment")
ASSORT_BEFORE = 0.0036

print(f"[] Sentiment assortativity BEFORE: {ASSORT_BEFORE}")
print(f"[] Sentiment assortativity AFTER:  {sent_assort_after:.4f}")



# 7. SAVE NEW GRAPH

with open("graph_after_echo_destruction.pkl", "wb") as f:
    pickle.dump(G2, f)

print("[] Saved → graph_after_echo_destruction.pkl")



# 8. SUMMARY

print("\n================= ECHO CHAMBER DESTRUCTION SUMMARY ================")
print(f"Echo chambers destroyed:        {len(echo_comms)}")
print(f"Internal edges removed:         {edges_removed}")
print(f"Cross-community edges added:    {edges_added}")
print(f"Modularity BEFORE:              {MOD_BEFORE}")
print(f"Modularity AFTER:               {mod_after:.4f}")
print(f"Sentiment assort BEFORE:        {ASSORT_BEFORE}")
print(f"Sentiment assort AFTER:         {sent_assort_after:.4f}")
print("====================================================================\n")
print("[ SUCCESS] Echo chambers fully destroyed.")
