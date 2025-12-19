# Echo Chambers and Polarization Analysis

## 📌 Overview
Social media platforms amplify opinions rapidly, often creating **echo chambers** where users are exposed only to similar viewpoints. This leads to increased **polarization**, biased public discourse, and misinformation spread.

This project presents a **graph-based computational framework** to detect echo chambers, measure polarization, and evaluate intervention strategies using real-world social media data from US election discussions on Twitter/X.

The system integrates **network science**, **sentiment analysis**, and **community detection algorithms**, followed by **intervention simulations** to study how polarization can be mitigated.

---

## 🎯 Objectives
- Detect echo chambers in social media interaction networks  
- Quantify polarization using sentiment variance and semantic cohesion  
- Compare community detection approaches  
- Simulate interventions to reduce polarization  
- Visualize structural and opinion-level changes  

---

## 🧠 Methodology Overview
The project is implemented as a **multi-stage pipeline**, where each stage depends on the outputs of the previous one:

1. Data preparation  
2. Graph construction  
3. Echo chamber detection  
4. Polarization measurement  
5. Intervention simulations  

---

## 🔄 Execution Pipeline (Order of Execution)

> ⚠️ **Important:** Scripts must be executed in the following order to reproduce results correctly.

---

### 1️⃣ Data Preparation
**File:** `data.py`

- Loads raw Twitter/X election dataset
- Parses tweet text, user IDs, mentions, timestamps, and engagement metrics
- Prepares structured data for graph construction

---

### 2️⃣ Graph Construction
**File:** `connections.py`

- Constructs a **directed weighted user interaction graph**
- Nodes represent users
- Edges represent mentions and reply relationships
- Edge weights are computed using engagement ratios
- Outputs a NetworkX graph used by all downstream tasks

---

### 3️⃣ Echo Chamber Detection
**File:** `echo_detection_vader_context_gpu.py`

- Applies **VADER sentiment analysis** on tweet text
- Generates semantic context representations
- Performs community detection using:
  - Louvain
  - Label Propagation
- Identifies echo chambers based on:
  - Low sentiment variance
  - High semantic cohesion
- Produces echo chamber statistics and visualizations

Related files:
- `echo_detection_labelprop_gpu.py`
- `compare_echo_methods.py`

---

### 4️⃣ Polarization Measurement
**Files:**
- `Polarization.py`
- `Polarization_master.py`
- `louvain_polarization_analysis.py`
- `louvain_echo_polarization.py`

- Computes polarization metrics such as:
  - Sentiment variance
  - Semantic cohesion
  - Modularity
  - Assortativity
- Generates community-level polarization statistics
- Saves polarization indices and comparison plots

---

### 5️⃣ Intervention Strategies

#### 🔹 Feed Diversification
**File:** `Intervention_FeedDiversification.py`

- Introduces cross-community exposure by injecting external nodes
- Measures change in:
  - Sentiment variance
  - Semantic cohesion
- Evaluates effectiveness of exposure-based depolarization

---

#### 🔹 Echo Destruction (Rewiring)
**File:** `Intervention_Rewiring.py`

- Removes internal edges within echo chambers
- Adds cross-community bridges
- Measures structural and sentiment-level impact post-intervention

---

## 📊 Outputs & Results

### Output CSV Files
- `community_echo_stats_labelprop.csv`
- `community_polarization.csv`
- `echo_polarization_louvain.csv`
- `echo_polarization_index.csv`

These files contain:
- Community-wise cohesion
- Sentiment variance
- Polarization indices

---

### Visualizations
Generated plots include:
- Echo vs non-echo comparisons  
- Polarization vs variance scatter plots  
- Heatmaps (before and after interventions)  

Key files:
- `echo_chamber_gpu.png`
- `non_echo_gpu.png`
- `polarity_vs_variance.png`
- `variance_comparison.png`
- `echo_visualizations/`

---

## 📁 Dataset Information

### 🔗 Data Source
The dataset used in this project was obtained from:

**GitHub Repository:**  
https://github.com/sinking8/x-24-us-election  

(Shared via: https://share.google/opBT2RoLFrpNCOoHL)

---

### 📄 Dataset Description
- Platform: Twitter (X)
- Topic: US Election discussions
- Time Range: May – July
- Content:
  - Tweets
  - Mentions and replies
  - Engagement metrics
  - User interaction data

---

### ⚠️ Data Availability Note
Due to size constraints, **raw dataset files are not included** in this repository.

To reproduce the analysis:
1. Download the dataset from the source above  
2. Place raw CSV files inside a local directory  
3. Execute the pipeline scripts in the order listed above  

---

## 🛠️ Tech Stack
- Python  
- NetworkX  
- VADER Sentiment Analyzer  
- Pandas & NumPy  
- Matplotlib  

---

## 📌 Key Findings
- Multiple strongly polarized echo chambers were detected  
- High semantic cohesion strongly correlates with low sentiment variance  
- Feed diversification and rewiring:
  - Reduced cohesion
  - Increased opinion diversity
  - Lowered structural polarization  

---

## 👤 Authors
- Mukund Nigam  
- Sanskriti Jain  
- Anshumaan Tiwari  

---

## 📚 References
- Baumann et al., *Physical Review Letters*, 2020  
- Impiccichè & Viviani, *ACM Transactions on the Web*, 2024  
- Haque et al., *AI & Society*, 2023  
- Wu et al., *Chaos, Solitons & Fractals*, 2023  
