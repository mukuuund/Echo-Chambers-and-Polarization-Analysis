import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Load data
df = pd.read_csv("community_echo_stats_hybrid_gpu.csv")

VIS_FOLDER = "echo_visualizations"
os.makedirs(VIS_FOLDER, exist_ok=True)

# Metrics
metrics = ["sentiment_variance", "semantic_cohesion", "mean_sentiment", "size"]

# Only "size" varies — so we'll visualize size prominently
df_plot = df[["community_id", "size"]].copy()
df_plot["metric"] = "size"

plt.figure(figsize=(20, 14))
ax = sns.scatterplot(
    data=df_plot,
    x="metric",
    y="community_id",
    size="size",
    hue="size",
    sizes=(20, 800),
    palette="viridis",
    legend="brief"
)

plt.title("Bubble Heatmap of Community Sizes (Final Echo Detection)", fontsize=18)
plt.xlabel("Metric", fontsize=14)
plt.ylabel("Community ID", fontsize=14)
plt.tight_layout()

plt.savefig(f"{VIS_FOLDER}/bubble_heatmap_size.png", dpi=300)
plt.close()

print("[✔] Bubble Heatmap saved as bubble_heatmap_size.png")
