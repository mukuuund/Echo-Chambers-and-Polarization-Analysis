import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from math import pi
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ============================================================
# 0. CREATE FOLDER
# ============================================================
VIS_FOLDER = "echo_visualizations/advanced"
os.makedirs(VIS_FOLDER, exist_ok=True)

print(f"[✔] Advanced Visualization folder created: {VIS_FOLDER}")

# ============================================================
# 1. LOAD GRAPH & CSV
# ============================================================
with open("graph_with_echo_hybrid_gpu.pkl", "rb") as f:
    G = pickle.load(f)

df = pd.read_csv("community_echo_stats_hybrid_gpu.csv")

echo_ids = df[df["is_echo"] == True]["community_id"].tolist()
largest_echo = df[df["is_echo"]].sort_values("size", ascending=False).iloc[0]["community_id"]
echo_nodes = [n for n, d in G.nodes(data=True) if d.get("community") == largest_echo]

print(f"[✔] Largest Echo Chamber ID: {largest_echo} | Nodes: {len(echo_nodes)}")

echo_graph = G.subgraph(echo_nodes).copy()

# ============================================================
# 2. CLEAN SENTIMENTS
# ============================================================
raw_sents = [G.nodes[n].get("sentiment", 0.0) for n in echo_nodes]
sentiments = np.array([
    0.0 if (s is None or isinstance(s, float) and np.isnan(s)) else float(s)
    for s in raw_sents
])

# ============================================================
# 3. PLOT 1: CHORD DIAGRAM (MENTION FLOW)
# ============================================================
print("[+] Generating chord diagram...")

adj = nx.to_numpy_array(echo_graph)
labels = list(echo_graph.nodes())

theta = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
coords = [(10*np.cos(t), 10*np.sin(t)) for t in theta]

plt.figure(figsize=(12,12))
ax = plt.gca()
ax.set_aspect('equal')
plt.axis("off")

for (x, y) in coords:
    ax.scatter(x, y, s=80, color='steelblue')

for i in range(len(labels)):
    for j in range(i+1, len(labels)):
        if adj[i, j] > 0:
            x1, y1 = coords[i]
            x2, y2 = coords[j]
            ax.plot([x1, x2], [y1, y2], color='gray', alpha=0.25)

plt.title(f"Chord Diagram – Echo Chamber {largest_echo}")
plt.savefig(f"{VIS_FOLDER}/chord_diagram_echo.png", dpi=300)
plt.close()

# ============================================================
# 4. PLOT 2: HUB–SPOKE INFLUENCER STRUCTURE
# ============================================================
print("[+] Generating hub-spoke plot...")

center = max(echo_graph.degree, key=lambda x: x[1])[0]
others = [n for n in echo_nodes if n != center]

theta = np.linspace(0, 2*np.pi, len(others), endpoint=False)

plt.figure(figsize=(10,10))
plt.scatter(0,0,s=1200,color="gold",edgecolors="black",label="Core Influencer")

for i,n in enumerate(others):
    x,y=np.cos(theta[i])*8, np.sin(theta[i])*8
    plt.scatter(x,y,s=100,color="skyblue")
    plt.plot([0,x],[0,y],color="gray",alpha=0.5)

plt.axis("off")
plt.title(f"Hub–Spoke Influencer Plot – Echo {largest_echo}")
plt.savefig(f"{VIS_FOLDER}/hub_spoke_echo.png", dpi=300)
plt.close()

# ============================================================
# 5. PLOT 3: SENTIMENT HISTOGRAM + KDE
# ============================================================
print("[+] Generating sentiment KDE plot...")

plt.figure(figsize=(8,6))
sns.histplot(sentiments, kde=True, color="purple", bins=20)
plt.title(f"Sentiment Distribution – Echo {largest_echo}")
plt.xlabel("Sentiment Score")
plt.ylabel("Count")
plt.savefig(f"{VIS_FOLDER}/sentiment_kde.png", dpi=300)
plt.close()

# ============================================================
# 6. PLOT 4: PCA + t-SNE OF EMBEDDINGS
# ============================================================
print("[+] Preparing embeddings for PCA/UMAP...")

vectors = [
    G.nodes[n].get("context")
    for n in echo_nodes
    if isinstance(G.nodes[n].get("context"), np.ndarray)
]

if len(vectors) >= 5:
    
    X = np.vstack(vectors)
    X_std = StandardScaler().fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(X_std)

    # TSNE
    tsne = TSNE(n_components=2, perplexity=5, learning_rate="auto", init="pca")
    tsne_2d = tsne.fit_transform(X_std)

    plt.figure(figsize=(8,6))
    plt.scatter(pca_2d[:,0], pca_2d[:,1], c=sentiments[:len(pca_2d)], cmap="coolwarm")
    plt.title(f"PCA Semantic Projection – Echo {largest_echo}")
    plt.colorbar(label="Sentiment")
    plt.savefig(f"{VIS_FOLDER}/pca_projection.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8,6))
    plt.scatter(tsne_2d[:,0], tsne_2d[:,1], c=sentiments[:len(tsne_2d)], cmap="coolwarm")
    plt.title(f"t-SNE Semantic Projection – Echo {largest_echo}")
    plt.colorbar(label="Sentiment")
    plt.savefig(f"{VIS_FOLDER}/tsne_projection.png", dpi=300)
    plt.close()

else:
    print("[!] Not enough embeddings — PCA/t-SNE skipped safely.")

# ============================================================
# 7. PLOT 5: MESSAGE FLOW (DIRECTED FLOW MAP)
# ============================================================
print("[+] Generating message flow plot...")

plt.figure(figsize=(12,12))
pos = nx.circular_layout(echo_graph)

degrees = dict(echo_graph.degree())
sizes = [degrees[n]*40 for n in echo_graph.nodes()]

nx.draw_networkx_nodes(echo_graph, pos, node_size=sizes, node_color="skyblue", alpha=0.9)
nx.draw_networkx_edges(echo_graph, pos, alpha=0.4, arrows=True, arrowsize=10)
plt.title(f"Message Flow Diagram – Echo {largest_echo}")
plt.axis("off")
plt.savefig(f"{VIS_FOLDER}/message_flow.png", dpi=300)
plt.close()

# ============================================================
# 8. PLOT 6: COMMUNITY CIRCULAR LAYOUT (WEIGHTED)
# ============================================================
print("[+] Generating circular community layout...")

plt.figure(figsize=(12,12))
nx.draw_circular(echo_graph, node_color="steelblue",
                 node_size=120,
                 edge_color="gray", alpha=0.6,
                 with_labels=False)

plt.title(f"Community Circular Layout – Echo {largest_echo}")
plt.savefig(f"{VIS_FOLDER}/circular_layout.png", dpi=300)
plt.close()

# ============================================================
# 9. PLOT 7: BAR CHART – TOP USERS BY CENTRALITY
# ============================================================
print("[+] Generating centrality bar chart...")

centrality = nx.degree_centrality(echo_graph)
top10 = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]

names = [str(x[0]) for x in top10]
values = [x[1] for x in top10]

plt.figure(figsize=(10,6))
sns.barplot(x=values, y=names, palette="Blues_r")
plt.title(f"Top 10 Influential Users – Echo {largest_echo}")
plt.xlabel("Degree Centrality")
plt.ylabel("User ID")
plt.savefig(f"{VIS_FOLDER}/centrality_barchart.png", dpi=300)
plt.close()

# ============================================================
print("\n🎉 ALL ADVANCED VISUALIZATIONS GENERATED SUCCESSFULLY!")
print(f"Saved in: {VIS_FOLDER}")
