# BUILD & SAVE DIRECTED GRAPH WITH BERT CONTEXT AND LIKE/VIEW WEIGHT (GPU OPTIMIZED)
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import ast
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# LOAD DATA 
print("[+] Loading filtered data...")
df = pd.read_csv("filtered_may_july.csv")

# Clean and standardize ID columns
df["user_id"] = df["user_id"].astype(str)
df["mention_ids"] = df["mention_ids"].apply(
    lambda x: [str(m) for m in ast.literal_eval(x)] if isinstance(x, str) and x.startswith("[") else []
)
df["in_reply_to_user_id_str"] = df["in_reply_to_user_id_str"].astype(str)

# Ensure numeric columns exist
for col in ["like_count", "view_count"]:
    if col not in df.columns:
        df[col] = 0

# Compute like/view ratio safely
df["weight_value"] = df.apply(
    lambda x: x["like_count"] / x["view_count"] if x["view_count"] > 0 else 0, axis=1
)


print("[+] Loading BERT model (all-MiniLM-L6-v2) on GPU...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

print("[+] Encoding tweets for contextual embeddings (batched on GPU)...")
texts = df["text"].astype(str).tolist()
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
df["context_embedding"] = list(embeddings)

#   BUILD DIRECTED GRAPH  
print("[+] Building directed graph...")
G = nx.DiGraph()

def add_edge(src, tgt, weight):
    """Add or update directed weighted edge."""
    if not src or not tgt or tgt == "nan":
        return
    G.add_node(src)
    G.add_node(tgt)
    old_w = G.get_edge_data(src, tgt, {"weight": 0})["weight"]
    G.add_edge(src, tgt, weight=old_w + weight)

# Iterate through rows with progress bar
for _, row in tqdm(df.iterrows(), total=len(df), desc="[+] Adding nodes & edges"):
    src = row["user_id"]
    weight = row["weight_value"]

    # Ensure node exists
    G.add_node(src)

    # Mentions
    for tgt in row["mention_ids"]:
        add_edge(src, tgt, weight)

    # Replies
    tgt = row["in_reply_to_user_id_str"]
    add_edge(src, tgt, weight)

    # Attach tweet-level data to node
    G.nodes[src]["tweet_id"] = row.get("id", None)
    G.nodes[src]["tweet_text"] = row.get("text", "")
    G.nodes[src]["context"] = row["context_embedding"]

#  SUMMARY 
print("\n[=] Directed graph successfully built!")
print(f"Total nodes: {G.number_of_nodes()}")
print(f"Total edges: {G.number_of_edges()}")

#   SAVE COMPLETE GRAPH  
with open("graph_context.pkl", "wb") as f:
    pickle.dump(G, f)
print("[Grph] Graph saved to 'graph_context.pkl' (complete with all nodes and edges)")

#   VISUALIZE SUBGRAPH (TOP 1000 NODES)  
print("\n[+] Preparing 1000-node subgraph for visualization...")
top_nodes = sorted(G.degree(weight="weight"), key=lambda x: x[1], reverse=True)[:1000]
sub_nodes = [n for n, _ in top_nodes]
H = G.subgraph(sub_nodes).copy()

print(f"Subgraph: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(H, k=0.3, seed=42)
node_sizes = [H.degree(n, weight="weight") * 0.1 for n in H.nodes()]
nx.draw_networkx_nodes(H, pos, node_size=node_sizes, node_color="skyblue", alpha=0.8)
nx.draw_networkx_edges(H, pos, alpha=0.3, arrows=True, edge_color="green")
plt.title("Directed Twitter Graph (Top 1000 Nodes Weighted by Like/View Ratio)", fontsize=14)
plt.axis("off")
plt.show()
