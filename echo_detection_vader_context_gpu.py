# ECHO CHAMBER DETECTION USING LOUVAIN + VADER + CONTEXT (GPU OPTIMIZED + TOPIC STORAGE)
 
import pickle
import networkx as nx
import numpy as np
import pandas as pd
import torch
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt
import re
from collections import Counter
from networkx.algorithms.community import louvain_communities

 
# 1. SETUP
 
print("[+] Checking GPU availability...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[ ] Using device: {device}")

# VADER setup
try:
    analyzer = SentimentIntensityAnalyzer()
except:
    nltk.download("vader_lexicon")
    analyzer = SentimentIntensityAnalyzer()

 
# 2. LOAD GRAPH
 
print("[+] Loading graph_context.pkl ...")
with open("graph_context.pkl", "rb") as f:
    G = pickle.load(f)
print(f"[ ] Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

 
# 3. APPLY VADER SENTIMENT
 
print("[+] Applying VADER sentiment to tweet text...")
node_sentiment = {}
for n, data in G.nodes(data=True):
    text = data.get("tweet_text", "") or ""
    score = analyzer.polarity_scores(str(text))["compound"]
    node_sentiment[n] = score

nx.set_node_attributes(G, node_sentiment, "sentiment")
print(f"[ ] Sentiment computed for {len(node_sentiment)} nodes")

 
# 4. VALIDATE EDGE WEIGHTS
 
print("[+] Checking and repairing edge weights...")
num_edges = G.number_of_edges()
zero_or_invalid = 0
for u, v, d in G.edges(data=True):
    w = d.get("weight", 0)
    if w is None or not isinstance(w, (int, float)) or np.isnan(w) or w <= 0:
        G[u][v]["weight"] = 1.0  # assign minimal positive fallback
        zero_or_invalid += 1

if zero_or_invalid > 0:
    print(f"[!] Repaired {zero_or_invalid} invalid or zero-weight edges out of {num_edges}")

total_weight = sum(float(d.get("weight", 0)) for _, _, d in G.edges(data=True))
if total_weight <= 0:
    raise ValueError(" No positive edge weights remain. Check input data.")
print(f"[ ] Total edge weight after repair: {total_weight:.2f}")

 
# 5. LOUVAIN COMMUNITY DETECTION
 
print("[+] Running Louvain community detection...")
communities = louvain_communities(G, weight="weight", seed=42)
print(f"[ ] Detected {len(communities)} communities")

community_map = {node: i for i, comm in enumerate(communities) for node in comm}
nx.set_node_attributes(G, community_map, "community")

 
# 6. GPU COSINE SIMILARITY FUNCTION

def gpu_cosine_similarity(vectors):
    """Compute mean pairwise cosine similarity using GPU if available."""
    tensor = torch.tensor(np.stack(vectors), device=device, dtype=torch.float32)
    normed = torch.nn.functional.normalize(tensor, p=2, dim=1)
    sims = torch.mm(normed, normed.T)
    upper = sims.triu(diagonal=1)
    non_zero = upper[upper != 0]
    return float(non_zero.mean().item()) if len(non_zero) > 0 else 0.0

 
# 7. COMMUNITY STATS (SENTIMENT + SEMANTIC + CONTEXT)
 
def extract_keywords(texts, top_k=10):
    """Extract top keywords for community topic summary."""
    if not texts:
        return ""
    all_text = " ".join(texts).lower()
    tokens = re.findall(r"\b[a-zA-Z#]{3,}\b", all_text)
    stopwords = {
        "the","and","for","you","https","http","com","www","this","that",
        "with","from","have","your","about","just","they","them","are","was",
        "were","will","would","could","should","into","their","there","what",
        "when","where","who","why","how","but","not","out","has","had","all",
        "can","get","our","more","than","its","it's","i","me","my","we","us",
        "on","in","to","of","is","a","an","at"
    }
    filtered = [t for t in tokens if t not in stopwords]
    if not filtered:
        return ""
    common = Counter(filtered).most_common(top_k)
    return ", ".join([w for w, _ in common])

print("[+] Computing sentiment variance, semantic cohesion, and context...")

stats = []
for i, comm in enumerate(communities):
    comm = list(comm)
    if len(comm) < 3:
        continue

    # Sentiment variance
    sentiments = [G.nodes[n].get("sentiment", 0.0) for n in comm]
    sent_mean = float(np.mean(sentiments))
    sent_var = float(np.var(sentiments))

    # Semantic cohesion and mean context vector
    vectors = [G.nodes[n].get("context") for n in comm if G.nodes[n].get("context") is not None]
    if len(vectors) >= 2:
        sem_mean = gpu_cosine_similarity(vectors)
        context_vector = torch.mean(torch.tensor(np.stack(vectors), device=device), dim=0).cpu().numpy()
    else:
        sem_mean = 0.0
        context_vector = np.zeros(384)

    # Topic extraction (what the echo chamber is about)
    texts = [G.nodes[n].get("tweet_text", "") for n in comm if G.nodes[n].get("tweet_text")]
    topic_keywords = extract_keywords(texts, top_k=10)

    stats.append({
        "community_id": i,
        "size": len(comm),
        "mean_sentiment": sent_mean,
        "sentiment_variance": sent_var,
        "semantic_cohesion": sem_mean,
        "topic_keywords": topic_keywords,
        "context_vector": context_vector.tolist()
    })

df_stats = pd.DataFrame(stats)
print(f"[ ] Computed stats for {len(df_stats)} communities")

 
# 8. DETERMINE ECHO CHAMBERS
 
if len(df_stats) > 0:
    sent_var_thr = np.percentile(df_stats["sentiment_variance"], 25)
    sem_coh_thr = np.percentile(df_stats["semantic_cohesion"], 75)
else:
    sent_var_thr, sem_coh_thr = 0, 0

df_stats["is_echo"] = (df_stats["sentiment_variance"] <= sent_var_thr) & (
    df_stats["semantic_cohesion"] >= sem_coh_thr
)

echo_ids = df_stats[df_stats["is_echo"]]["community_id"].tolist()
non_echo_ids = df_stats[~df_stats["is_echo"]]["community_id"].tolist()

print(f"[ ] Echo chambers detected: {len(echo_ids)} / {len(df_stats)} communities")

 
# 9. SAVE RESULTS
 
df_stats.to_csv("community_echo_stats_hybrid_gpu.csv", index=False)
print("[ ] Saved → community_echo_stats_hybrid_gpu.csv")

 
# 10. COLOR EDGES BY ECHO STATUS
 
edge_colors = {}
for u, v in G.edges():
    cu, cv = community_map.get(u), community_map.get(v)
    if cu == cv and cu in echo_ids:
        edge_colors[(u, v)] = "red"
    else:
        edge_colors[(u, v)] = "green"
nx.set_edge_attributes(G, edge_colors, "color")

with open("graph_with_echo_hybrid_gpu.pkl", "wb") as f:
    pickle.dump(G, f)
print("[ ] Graph saved with echo info → graph_with_echo_hybrid_gpu.pkl")

 
# 11. VISUALIZATION
 
def visualize_community(G, nodes, title, filename):
    H = G.subgraph(nodes).copy()
    pos = nx.spring_layout(H, seed=42, k=0.3)
    edge_colors = [H[u][v].get("color", "gray") for u, v in H.edges()]
    node_sizes = [max(H.degree(n, weight="weight") * 0.1, 20) for n in H.nodes()]

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(H, pos, node_size=node_sizes, node_color="skyblue", alpha=0.85)
    nx.draw_networkx_edges(H, pos, alpha=0.4, edge_color=edge_colors)
    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[ ] Saved visualization → {filename}")

if echo_ids:
    echo_top = max(echo_ids, key=lambda cid: df_stats.loc[df_stats["community_id"] == cid, "size"].iloc[0])
    echo_nodes = list(communities[echo_top])
    visualize_community(G, echo_nodes, f"Echo Chamber #{echo_top}", "echo_chamber_gpu.png")

if non_echo_ids:
    non_top = max(non_echo_ids, key=lambda cid: df_stats.loc[df_stats["community_id"] == cid, "size"].iloc[0])
    non_nodes = list(communities[non_top])
    visualize_community(G, non_nodes, f"Non-Echo Community #{non_top}", "non_echo_gpu.png")

print("\n[ ] GPU-accelerated echo chamber detection completed successfully!")

