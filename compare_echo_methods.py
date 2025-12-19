# -------------------------------------------------------------
# COMPARE ECHO CHAMBER DETECTIONS: LOUVAIN vs LABEL PROPAGATION
# -------------------------------------------------------------
import pandas as pd
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import numpy as np

# -------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------
print("[+] Loading community CSV files...")
df_louvain = pd.read_csv("community_echo_stats_hybrid_gpu.csv")
df_label = pd.read_csv("community_echo_stats_labelprop.csv")
print(f"[✔] Louvain communities: {len(df_louvain)}, Label Propagation communities: {len(df_label)}")

# -------------------------------------------------------------
# 2. BASIC STATS
# -------------------------------------------------------------
echo_louvain = df_louvain[df_louvain["is_echo"] == True]
echo_label = df_label[df_label["is_echo"] == True]
print(f"[📊] Louvain echo chambers: {len(echo_louvain)}")
print(f"[📊] Label Propagation echo chambers: {len(echo_label)}")

# -------------------------------------------------------------
# 3. LOAD GRAPHS
# -------------------------------------------------------------
with open("graph_with_echo_hybrid_gpu.pkl", "rb") as f:
    G_louvain = pickle.load(f)
with open("graph_with_echo_labelprop.pkl", "rb") as f:
    G_label = pickle.load(f)
print("[✔] Both graphs loaded successfully.")

# -------------------------------------------------------------
# 4. MAP ECHO NODES FROM BOTH GRAPHS
# -------------------------------------------------------------
echo_nodes_louvain = {n for n, d in G_louvain.nodes(data=True) if d.get("community") in echo_louvain["community_id"].values}
echo_nodes_label = {n for n, d in G_label.nodes(data=True) if d.get("community") in echo_label["community_id"].values}

# Compute overlap
intersection = echo_nodes_louvain & echo_nodes_label
union = echo_nodes_louvain | echo_nodes_label
overlap_ratio = len(intersection) / len(union) if len(union) > 0 else 0

print(f"[🔍] Echo node overlap: {len(intersection)} common out of {len(union)} total ({overlap_ratio*100:.2f}%)")

# -------------------------------------------------------------
# 5. TOPIC SIMILARITY CHECK
# -------------------------------------------------------------
def jaccard_keywords(str1, str2):
    s1, s2 = set(str1.split(", ")), set(str2.split(", "))
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 0

topic_scores = []
for i, row1 in echo_louvain.iterrows():
    for j, row2 in echo_label.iterrows():
        topic_scores.append(jaccard_keywords(str(row1["topic_keywords"]), str(row2["topic_keywords"])))
topic_similarity = np.mean(topic_scores) if topic_scores else 0
print(f"[🧠] Average topic similarity between echo chambers: {topic_similarity:.3f}")

# -------------------------------------------------------------
# 6. VISUALIZATION
# -------------------------------------------------------------
plt.figure(figsize=(8, 5))
methods = ["Louvain", "Label Propagation", "Overlap"]
values = [len(echo_louvain), len(echo_label), len(intersection)]
colors = ["skyblue", "lightgreen", "salmon"]

plt.bar(methods, values, color=colors)
plt.title("Comparison of Echo Chamber Detections", fontsize=14)
plt.ylabel("Number of Echo Chambers / Nodes", fontsize=12)
plt.text(2, len(intersection), f"{overlap_ratio*100:.1f}% overlap", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig("echo_detection_comparison.png", dpi=300)
plt.close()
print("[✔] Saved visualization → echo_detection_comparison.png")

# -------------------------------------------------------------
# 7. SUMMARY
# -------------------------------------------------------------
print("\n[✅] Comparison complete!")
print(f"""
Summary:
---------
• Louvain echo chambers: {len(echo_louvain)}
• Label Propagation echo chambers: {len(echo_label)}
• Node-level overlap: {len(intersection)} ({overlap_ratio*100:.2f}%)
• Avg. topic similarity: {topic_similarity:.3f}
• Visualization: echo_detection_comparison.png
""")
