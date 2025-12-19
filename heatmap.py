import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# 1. Load Data
# ============================================================
df = pd.read_csv("community_echo_stats_hybrid_gpu.csv")

# Create output folder
VIS_FOLDER = "echo_visualizations"
os.makedirs(VIS_FOLDER, exist_ok=True)

# ============================================================
# 2. Select and Prepare Metrics
# ============================================================
df["is_echo"] = df["is_echo"].astype(int)

metrics = ["sentiment_variance", "semantic_cohesion", "mean_sentiment", "size", "is_echo"]
data = df[["community_id"] + metrics].set_index("community_id")

# ============================================================
# 3. Create a high-contrast version of metrics
# ============================================================

# Z-score normalization for all numeric columns
data_z = data.copy()

for col in metrics:
    col_min = data_z[col].min()
    col_max = data_z[col].max()
    
    # If constant/no variation → add tiny jitter to create contrast
    if col_max - col_min == 0:
        data_z[col] = 0.5 + (np.random.rand(len(data_z)) - 0.5) * 0.05
    else:
        mean = data_z[col].mean()
        std = data_z[col].std()
        data_z[col] = (data_z[col] - mean) / std

# Replace "size" with log-scaled version (huge visual contrast)
data_z["size"] = np.log1p(data["size"])

# Sort by largest communities → clearer structure
data_z = data_z.sort_values("size", ascending=False)

# ============================================================
# 4. High-Contrast Heatmap
# ============================================================
plt.figure(figsize=(14, 12))

sns.heatmap(
    data_z,
    cmap="magma",        # Highest contrast palette
    linewidths=0.2,
    linecolor="black",
    cbar_kws={"label": "Contrast-Boosted Scale"}
)

plt.title("High-Contrast Heatmap of All Communities (All Metrics)", fontsize=18)
plt.xlabel("Metrics")
plt.ylabel("Community ID")
plt.tight_layout()

plt.savefig(f"{VIS_FOLDER}/high_contrast_heatmap.png", dpi=300)
plt.close()

print("✔ High-Contrast Heatmap saved at echo_visualizations/high_contrast_heatmap.png")
