import pickle
import networkx as nx
import numpy as np
import pandas as pd
import torch
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from infomap import Infomap
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# -----------------------------------------------------
# 1. SETUP
# -----------------------------------------------------
print("[+] Checking GPU availability...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[✔] Using device: {device}")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# -----------------------------------------------------
# 2. LOAD GRAPH
# -----------------------------------------------------
print("[+] Loading graph_context.pkl ...")
with open("graph_context.pkl", "rb") as f:
    G = pickle.load(f)
print(f"[✔] Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# -----------------------------------------------------
# 3. FIX EDGE WEIGHTS
# -----------------------------------------------------
print("[+] Fixing invalid or zero-weight edges for Infomap...")
for u, v, d in G.edges(data=True):
    w = d.get("weight", 1.0)
    if w is None or not isinstance(w, (int, float)) or w <= 0:
        G[u][v]["weight"] = 1.0
print("[✔] Edge weights fixed.")

# -----------------------------------------------------
# 4. APPLY VADER SENTIMENT
# -----------------------------------------------------
print("[+] Applying VADER sentiment...")
analyzer = SentimentIntensityAnalyzer()
sentiment_scores = {n: analyzer.polarity_scores(G.nodes[n].get("tweet_text", ""))["compound"] for n in G.nodes}
nx.set_node_attributes(G, sentiment_scores, "sentiment")
print("[✔] VADER sentiment scores added.")

# -----------------------------------------------------
# 5. RUN INFOMAP
# -----------------------------------------------------
print("[+] Running Infomap (directed + weighted)...")
infomap_wrapper = Infomap("--directed --two-level")

node2id = {node: i for i, node in enumerate(G.nodes())}
id2node = {i: node for node, i in node2id.items()}

for u, v, d in G.edges(data=True):
    infomap_wrapper.addLink(node2id[u], node2id[v], float(d.get("weight", 1.0)))

infomap_wrapper.run()

# Build communities
communities = {}
for node in infomap_wrapper.nodes:
    real_node = id2node[node.node_id]
    communities.setdefault(node.module_id, []).append(real_node)

print(f"[✔] Infomap found {len(communities)} communities")

# -----------------------------------------------------
# 6. ASSIGN COMMUNITY IDS
# -----------------------------------------------------
community_map = {node: cid for cid, members in communities.items() for node in members}
nx.set_node_attributes(G, community_map, "community")
print("[✔] Assigned Infomap community IDs.")

# -----------------------------------------------------
# 7. ECHO CHAMBER ANALYSIS (UPDATED WITH BERT CONTEXT)
# -----------------------------------------------------
results = []
community_context_vectors = {}  # <<--- ADDED

def cosine_similarity_matrix(embs):
    t = torch.tensor(np.array(embs), device=device)
    t = torch.nn.functional.normalize(t, dim=1)
    sim = torch.mm(t, t.T)
    return sim[np.triu_indices(len(embs), 1)].mean().item() if len(embs) >= 2 else 0.0

for cid, members in communities.items():
    if len(members) < 3:
        continue

    # Sentiment variance
    sentiments = [G.nodes[n].get("sentiment", 0.0) for n in members]
    sentiment_var = np.var(sentiments)

    # Semantic cohesion using BERT
    vectors = [G.nodes[n].get("context") for n in members if G.nodes[n].get("context") is not None]

    if len(vectors) >= 2:
        semantic_cohesion = cosine_similarity_matrix(vectors)

        # Compute mean BERT vector for community
        mean_context = torch.mean(torch.tensor(vectors, device=device), dim=0).cpu().numpy()
    else:
        semantic_cohesion = 0.0
        mean_context = np.zeros(384)

    # Save community mean vector (for PKL)
    community_context_vectors[cid] = mean_context  # <<--- ADDED

    # Keywords
    texts = [G.nodes[n].get("tweet_text", "") for n in members]
    words = pd.Series(" ".join(texts).lower().split()).value_counts()
    keywords = [w for w in words.index if w not in ENGLISH_STOP_WORDS and len(w) > 4][:5]

    # Save row (WITH BERT VECTOR)
    results.append({
        "community_id": cid,
        "size": len(members),
        "sentiment_variance": sentiment_var,
        "semantic_cohesion": semantic_cohesion,
        "keywords": ", ".join(keywords),
        "context_vector": mean_context.tolist()  # <<--- ADDED
    })

df = pd.DataFrame(results)

# Thresholds
if not df.empty:
    sv_thr = np.percentile(df["sentiment_variance"], 25)
    sc_thr = np.percentile(df["semantic_cohesion"], 75)
    df["is_echo"] = (df["sentiment_variance"] <= sv_thr) & (df["semantic_cohesion"] >= sc_thr)
    echo_ids = set(df[df["is_echo"]]["community_id"].tolist())
else:
    df["is_echo"] = []
    echo_ids = set()

# Save CSV (WITH BERT VECTORS)
df.to_csv("community_echo_stats_infomap.csv", index=False)
print("[💾] Saved → community_echo_stats_infomap.csv (with BERT context vector)")

# -----------------------------------------------------
# 8. COLOR EDGES & SAVE GRAPH (UPDATED)
# -----------------------------------------------------
for u, v in G.edges():
    cu = G.nodes[u].get("community")
    cv = G.nodes[v].get("community")
    if cu == cv and cu in echo_ids:
        G[u][v]["color"] = "red"
    else:
        G[u][v]["color"] = "green"

# SAVE COMMUNITY MEAN BERT VECTORS IN PKL
G.graph["community_context_vectors"] = community_context_vectors  # <<--- ADDED

try:
    with open("graph_with_echo_infomap.pkl", "wb") as f:
        pickle.dump(G, f)
    print("[💾] Graph saved → graph_with_echo_infomap.pkl (with BERT vectors)")
except Exception as e:
    print(f"[❌] Failed to save graph: {e}")

print("\n[✅] Infomap echo chamber detection + BERT context vectors completed successfully!")
