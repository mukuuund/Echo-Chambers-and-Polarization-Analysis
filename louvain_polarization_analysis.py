# -------------------------------------------------------------
# POLARIZATION ANALYSIS USING ROBERTA SENTIMENT + LOUVAIN
# -------------------------------------------------------------
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ----------------------- CONFIG ------------------------------
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
BATCH_SIZE = 64       # sentiment batch size
MAX_LEN = 128         # tokenizer max length
GRAPH_PATH = "graph_with_echo_hybrid_gpu.pkl"
OUTPUT_CSV = "louvain_roberta_polarization_stats.csv"

# -------------------------------------------------------------
# 1. LOAD GRAPH
# -------------------------------------------------------------
print("[+] Loading graph_with_echo_hybrid_gpu.pkl ...")
with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)

print(f"[✔] Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[✔] Using device: {device}")

# -------------------------------------------------------------
# 2. LOAD ROBERTA SENTIMENT MODEL (SAFETENSORS ONLY)
# -------------------------------------------------------------
print("[+] Loading RoBERTa Twitter sentiment model (safetensors)...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    use_safetensors=True,      # <-- IMPORTANT: avoid .bin / torch.load
    trust_remote_code=False
).to(device)
model.eval()

# -------------------------------------------------------------
# 3. APPLY ROBERTA SENTIMENT TO ALL NODES (BATCHED)
# -------------------------------------------------------------
print("[+] Applying RoBERTa sentiment to all tweets (batched)...")

# Collect node ids and texts
node_ids = []
texts = []
for n, data in G.nodes(data=True):
    text = data.get("tweet_text", "")
    if text is None or str(text).strip() == "":
        text = ""
    node_ids.append(n)
    texts.append(str(text))

sentiment_map = {}

with torch.no_grad():
    for i in tqdm(range(0, len(node_ids), BATCH_SIZE), desc="[+] Sentiment batches"):
        batch_ids = node_ids[i:i + BATCH_SIZE]
        batch_texts = texts[i:i + BATCH_SIZE]

        # Replace completely empty texts with neutral 0.0 directly
        non_empty_indices = [idx for idx, t in enumerate(batch_texts) if t.strip() != ""]
        if not non_empty_indices:
            # all empty in this batch
            for nid in batch_ids:
                sentiment_map[nid] = 0.0
            continue

        # Prepare inputs only for non-empty texts
        non_empty_texts = [batch_texts[idx] for idx in non_empty_indices]

        inputs = tokenizer(
            non_empty_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        ).to(device)

        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).cpu().numpy()  # shape: [batch, 3]

        # Map back to nodes
        for j, idx in enumerate(non_empty_indices):
            nid = batch_ids[idx]
            # sentiment score in [-1, 1]: pos - neg
            score = float(probs[j, 2] - probs[j, 0])
            sentiment_map[nid] = score

        # Empty texts → neutral
        for idx in range(len(batch_ids)):
            if idx not in non_empty_indices:
                nid = batch_ids[idx]
                sentiment_map[nid] = 0.0

# Attach to graph
nx.set_node_attributes(G, sentiment_map, "sentiment")
print(f"[✔] RoBERTa sentiment computed for {len(sentiment_map)} nodes.")

# -------------------------------------------------------------
# 4. FIX EDGE WEIGHTS (ENSURE POSITIVE & SIMPLE)
# -------------------------------------------------------------
print("[+] Ensuring all edge weights are positive (setting to 1.0)...")

for u, v in G.edges():
    G[u][v]["weight"] = 1.0

print("[✔] All edge weights set to 1.0 for Louvain consistency.")

# -------------------------------------------------------------
# 5. RUN LOUVAIN
# -------------------------------------------------------------
print("[+] Running Louvain community detection...")
communities = louvain_communities(G, weight="weight", seed=42)
print(f"[✔] Number of communities: {len(communities)}")

community_map = {node: i for i, comm in enumerate(communities) for node in comm}
nx.set_node_attributes(G, community_map, "community")

# -------------------------------------------------------------
# 6. COMMUNITY SENTIMENT STATS
# -------------------------------------------------------------
print("[+] Computing community sentiment stats...")

rows = []
for cid, comm in enumerate(communities):
    sentiments = [G.nodes[n].get("sentiment", 0.0) for n in comm]
    if len(sentiments) == 0:
        mean_s = 0.0
        var_s = 0.0
    else:
        mean_s = float(np.mean(sentiments))
        var_s = float(np.var(sentiments))

    rows.append({
        "community_id": cid,
        "size": len(comm),
        "mean_sentiment": mean_s,
        "sentiment_variance": var_s,
    })

df_stats = pd.DataFrame(rows)
print("[✔] Community stats computed.")
print(df_stats.head())

# -------------------------------------------------------------
# 7. GLOBAL POLARIZATION INDEX
# -------------------------------------------------------------
print("[+] Computing polarization index...")

all_sent = np.array([G.nodes[n].get("sentiment", 0.0) for n in G.nodes()])
global_mean = float(np.mean(all_sent))
global_var = float(np.var(all_sent))

means = df_stats["mean_sentiment"].values
weights = df_stats["size"].values
between_var = float(np.average((means - global_mean) ** 2, weights=weights))

polar_index = between_var / global_var if global_var > 0 else 0.0

print(f"[🔥] Polarization Index: {polar_index:.4f}")
print("    (Closer to 1 = more between-community polarization, 0 = no extra polarization)")

# -------------------------------------------------------------
# 8. ASSORTATIVITY (HOMOPHILY BY SENTIMENT BIN)
# -------------------------------------------------------------
print("[+] Computing sentiment assortativity...")

def bucket(s):
    if s > 0.05:
        return 1   # positive
    if s < -0.05:
        return -1  # negative
    return 0       # neutral

nx.set_node_attributes(G, {n: bucket(G.nodes[n].get("sentiment", 0.0)) for n in G.nodes()}, "sent_bin")

assort = nx.attribute_assortativity_coefficient(G.to_undirected(), "sent_bin")

print(f"[🔥] Sentiment Assortativity: {assort:.4f}")
print("    (Closer to 1 = strong homophily, 0 = random mixing, <0 = disassortative)")

# -------------------------------------------------------------
# 9. SAVE RESULTS
# -------------------------------------------------------------
df_stats.to_csv(OUTPUT_CSV, index=False)

print("\n[💾] Saved:")
print(f"    → {OUTPUT_CSV}")
print("[✔] RoBERTa-based polarization analysis complete!")
