import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")

import pickle
import pandas as pd
import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
import re
from collections import Counter

device = "cuda" if torch.cuda.is_available() else "cpu"


# 1. LOAD GRAPH + CSV DATA

print("[+] Loading graph and echo chamber CSVs...")

with open("graph_with_echo_hybrid_gpu.pkl", "rb") as f:
    G = pickle.load(f)

df_base = pd.read_csv("community_echo_stats_hybrid_gpu.csv")  # original echo detection
df_pol = pd.read_csv("all_echo_communities_polarity.csv")      # sentiment polarity summary

echo_ids = df_base[df_base["is_echo"] == True]["community_id"].tolist()
print(f"[] Echo chambers detected: {len(echo_ids)}")


# 2. REBUILD community_dict FROM GRAPH

print("[+] Rebuilding community_dict...")
community_dict = {}
for n, data in G.nodes(data=True):
    cid = data.get("community")
    if cid is None:
        continue
    community_dict.setdefault(cid, []).append(n)

print(f"[] Total communities found: {len(community_dict)}")


# 3. SEMANTIC COHESION FOR ALL ECHO CHAMBERS

print("[+] Computing semantic cohesion...")

def gpu_cosine_mean(vectors):
    tensor = torch.tensor(np.stack(vectors), device=device, dtype=torch.float32)
    normed = torch.nn.functional.normalize(tensor, p=2, dim=1)
    sims = torch.mm(normed, normed.T)
    upper = sims.triu(diagonal=1)
    vals = upper[upper != 0]
    return float(vals.mean().item()) if len(vals) > 0 else 0.0

sem_results = []

for cid in echo_ids:
    nodes = community_dict[cid]
    vectors = [
        G.nodes[n].get("context")
        for n in nodes
        if isinstance(G.nodes[n].get("context"), np.ndarray)
    ]
    cohesion = gpu_cosine_mean(vectors) if len(vectors) >= 2 else 0.0

    sem_results.append({
        "community_id": cid,
        "semantic_cohesion": cohesion
    })

df_sem = pd.DataFrame(sem_results)
df_sem.to_csv("echo_semantic_cohesion.csv", index=False)
print("[] Saved: echo_semantic_cohesion.csv")


# 4. TOPIC EXTRACTION FOR ALL ECHO CHAMBERS

print("[+] Extracting topic keywords...")

stopwords = {
    "the","and","for","you","https","http","com","www","this","that",
    "with","from","have","your","about","just","they","them","are","was",
    "were","will","would","could","should","into","their","there","what",
    "when","where","who","why","how","but","not","out","has","had","all",
    "can","get","our","more","than","its","it's","i","me","my","we","us",
    "on","in","to","of","is","a","an","at"
}

def extract_keywords(texts, top_k=10):
    all_text = " ".join(texts).lower()
    tokens = re.findall(r"\b[a-zA-Z#]{3,}\b", all_text)
    filtered = [t for t in tokens if t not in stopwords]
    common = Counter(filtered).most_common(top_k)
    return ", ".join([w for w, _ in common])

topic_results = []

for cid in echo_ids:
    texts = [G.nodes[n].get("tweet_text", "") for n in community_dict[cid]]
    keywords = extract_keywords(texts, 10)
    topic_results.append({
        "community_id": cid,
        "topic_keywords": keywords
    })

df_topics = pd.DataFrame(topic_results)
df_topics.to_csv("echo_topics.csv", index=False)
print("[] Saved: echo_topics.csv")


# 5. BUILD POLARIZATION INDEX

print("[+] Building Polarization Index...")

# Load metrics
MODULARITY = 0.6098
SENTIMENT_ASSORT = 0.0036

df = df_pol.merge(df_sem, on="community_id", how="left")
df = df.merge(df_topics, on="community_id", how="left")

# Normalize helper
def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-9)

df["norm_cohesion"] = normalize(df["semantic_cohesion"])
df["norm_var"] = 1 - normalize(df["polarity_variance"])  # lower variance = more polarized

df["polarization_index"] = (
    0.4 * MODULARITY +
    0.3 * df["norm_cohesion"] +
    0.2 * df["norm_var"] +
    0.1 * SENTIMENT_ASSORT
)

df.to_csv("echo_polarization_index.csv", index=False)
print("[] Saved: echo_polarization_index.csv")


# 6. POLARIZATION VISUALIZATIONS

print("[+] Creating visualizations...")

# Plot 1: Polarity vs Variance
plt.figure(figsize=(8,6))
plt.scatter(df["mean_tweet_polarity"], df["polarity_variance"], s=80)
for _, row in df.iterrows():
    plt.text(row["mean_tweet_polarity"], row["polarity_variance"], str(row["community_id"]))
plt.xlabel("Mean Polarity (MTLp)")
plt.ylabel("Polarity Variance")
plt.title("Echo Chamber: Polarity vs Variance")
plt.savefig("polarity_vs_variance.png", dpi=300)
plt.close()

# Plot 2: Semantic Cohesion vs Variance
plt.figure(figsize=(8,6))
plt.scatter(df["semantic_cohesion"], df["polarity_variance"], s=80)
for _, row in df.iterrows():
    plt.text(row["semantic_cohesion"], row["polarity_variance"], str(row["community_id"]))
plt.xlabel("Semantic Cohesion")
plt.ylabel("Polarity Variance")
plt.title("Echo Chamber: Cohesion vs Variance")
plt.savefig("cohesion_vs_variance.png", dpi=300)
plt.close()

print("[] Saved: polarity_vs_variance.png, cohesion_vs_variance.png")

print("\n[ ALL STEPS COMPLETED SUCCESSFULLY!]")
