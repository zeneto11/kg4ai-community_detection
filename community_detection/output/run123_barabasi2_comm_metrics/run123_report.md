# 📊 Graph Analysis Report

---

## 📋 Analysis Metadata

| Property | Value |
|----------|-------|
| **Run ID** | `run123` |
| **Timestamp** | 2025-09-29T02:33:54.166453 |
| **Graph Source** | Simulated Citation Network with Realistic Titles |
| **Analysis Version** | 3.0 |
| **Total Time** | 0s |

## 🎯 Executive Summary

**Network Overview:**
- **Scale:** Medium directed network
- **Original Size:** 200 nodes, 714 edges
- **Post-Cleaning Size:** 200 nodes, 714 edges (removed 0.00% of nodes)
- **Connectivity:** sparse (density: 1.79397%)
- **Average Connections:** 7.1 per node

**Key Characteristics:**
- ✅ **Fully Connected:** All nodes are reachable

---

## 📈 Enhanced Data Visualizations

### Comprehensive Degree Analysis
![Degree Analysis](visualizations/run123_degree_analysis.png)
*Four-panel analysis showing linear distribution, log-log scaling for power-law detection, cumulative distribution, and statistical summary via box plot.*

### Clustering Coefficient Analysis
![Clustering Analysis](visualizations/run123_clustering_analysis.png)
*Distribution and box plot analysis of local clustering coefficients, showing community structure tendencies.*

---

## 📊 Detailed Metrics

### 🏗️ Graph Structure

*Fundamental properties of the graph topology*

| Metric | Value | Status |
|--------|-------|--------|
| **Is Directed** | 1 | ✅ computed |
| **Is Weighted** | 0 | ✅ computed |
| **Is Multigraph** | 0 | ✅ computed |
| **Num Nodes** | 200 | ✅ computed |
| **Num Edges** | 714 | ✅ computed |

---

### 🔗 Connectivity Analysis

*Network connectivity and density measurements*

| Metric | Value | Status |
|--------|-------|--------|
| **Density** | 0.0179 | ✅ computed |
| **Is Connected Undirected** | 1 | ✅ computed |
| **Reciprocity** | 0.3445 | ✅ computed |
| **Is Strongly Connected** | 0 | ✅ computed |
| **Is Weakly Connected** | 1 | ✅ computed |

---

### 🏝️ Node Isolation Analysis

*Statistics about disconnected and removed nodes*

| Metric | Value | Status |
|--------|-------|--------|
| **Num Nodes Zero Total Degree** | 0 | ✅ computed |
| **Num Isolated Nodes Removed** | 0 | ✅ computed |
| **Min Component Size Threshold** | 2 | ✅ computed |
| **Num Small Components Removed** | 0 | ✅ computed |
| **Num Nodes In Small Components Removed** | 0 | ✅ computed |
| **Total Nodes Removed** | 0 | ✅ computed |
| **Num Nodes After Removal** | 200 | ✅ computed |
| **Num Edges After Removal** | 714 | ✅ computed |
| **Removal Percentage** | 0.00% | ✅ computed |

---

### 📈 Degree Distribution

*Statistical analysis of node connection patterns*

| Metric | Value | Status |
|--------|-------|--------|
| **Average Degree** | 7.14 | ✅ computed |
| **Max Degree** | 48 | ✅ computed |
| **Min Degree** | 3 | ✅ computed |
| **Median Degree** | 5.0 | ✅ computed |
| **Std Dev Degree** | 7.2 | ✅ computed |

---

### ➡️ Directed Graph Properties

*Metrics specific to directed networks (in/out degrees, PageRank)*

| Metric | Value | Status |
|--------|-------|--------|
| **Num Nodes Zero In Degree** | 18 | ✅ computed |
| **Max In Degree** | 37 | ✅ computed |
| **Min In Degree** | 0 | ✅ computed |
| **Mean In Degree** | 3.6 | ✅ computed |
| **Median In Degree** | 2.0 | ✅ computed |
| **Num Nodes Zero Out Degree** | 14 | ✅ computed |
| **Max Out Degree** | 41 | ✅ computed |
| **Min Out Degree** | 0 | ✅ computed |
| **Mean Out Degree** | 3.6 | ✅ computed |
| **Median Out Degree** | 3.0 | ✅ computed |
| **Average Pagerank** | 0.01 | ✅ computed |
| **Max Pagerank** | 0.09 | ✅ computed |

---

### 🧬 Structural Features

*Advanced topological characteristics*

| Metric | Value | Status |
|--------|-------|--------|
| **Self Loops** | 0 | ✅ computed |

---

### 🧩 Component Analysis

*Connected component structure and distribution*

| Metric | Value | Status |
|--------|-------|--------|
| **Num Weakly Connected Components** | 1 | ✅ computed |
| **Largest Weakly Connected Component Size** | 200 | ✅ computed |
| **Weakly Cc Size Mean** | 200.00 | ✅ computed |
| **Weakly Cc Size Median** | 200.00 | ✅ computed |
| **Weakly Connected Component Sizes** | [200] | ✅ computed |
| **Num Strongly Connected Components** | 87 | ✅ computed |
| **Largest Strongly Connected Component Size** | 27 | ✅ computed |
| **Strongly Cc Size Mean** | 2.30 | ✅ computed |
| **Strongly Cc Size Median** | 1.00 | ✅ computed |
| **Strongly Connected Component Sizes** | 87 items (μ=2.30, σ=4.55) | ✅ computed |

---

## 📚 Citation Network Analysis

*Specialized analysis for citation/reference networks*

### Citation Statistics

| Metric | Citations Made (Out-degree) | Citations Received (In-degree) |
|--------|------------------------------|----------------------------------|
| **Mean** | 3.57 | 3.57 |
| **Median** | 3.00 | 2.00 |
| **Maximum** | 41 | 37 |
| **Std Dev** | 4.05 | 4.89 |

**Network H-Index:** 11

### Node Categories

- 🏝️ **Isolated Nodes:** 0 (no citations)
- 🔚 **Terminal Nodes:** 14 (don't cite others)
- 🌱 **Source Nodes:** 18 (not cited by others)

### 🏆 Top 10 Most Cited Papers

| Rank | Node ID | Title | Citations |
|------|---------|-------|----------|
| 1 | `5` | Comparative study of statistical analysi... | **37** |
| 2 | `6` | Clustering for education in cv | **35** |
| 3 | `0` | Advances in bioinformatics for manufactu... | **31** |
| 4 | `10` | Advances in neural networks for healthca... | **23** |
| 5 | `7` | Advances in deep learning for education ... | **15** |
| 6 | `16` | Comparative study of classification meth... | **14** |
| 7 | `25` | Enhancing healthcare through deep learni... | **14** |
| 8 | `12` | A novel approach to finance using statis... | **13** |
| 9 | `1` | Optimization for finance in ai | **12** |
| 10 | `3` | Advances in drug discovery for transport... | **12** |

### 📝 Top 10 Most Citing Papers

| Rank | Node ID | Title | Citations Made |
|------|---------|-------|----------------|
| 1 | `4` | Classification for finance in cv | **41** |
| 2 | `7` | Advances in deep learning for education ... | **22** |
| 3 | `34` | Statistical Analysis for manufacturing i... | **17** |
| 4 | `3` | Advances in drug discovery for transport... | **15** |
| 5 | `14` | Classification for finance in data | **15** |
| 6 | `8` | Enhancing education through classificati... | **14** |
| 7 | `12` | A novel approach to finance using statis... | **13** |
| 8 | `6` | Clustering for education in cv | **12** |
| 9 | `29` | Statistical Analysis for manufacturing i... | **11** |
| 10 | `9` | Advances in natural language processing ... | **10** |

### 🌟 Top 10 Most Influential Papers (PageRank)

| Rank | Node ID | Title | Influence Score |
|------|---------|-------|----------------|
| 1 | `5` | Comparative study of statistical analysi... | 0.086372 |
| 2 | `0` | Advances in bioinformatics for manufactu... | 0.079767 |
| 3 | `25` | Enhancing healthcare through deep learni... | 0.042254 |
| 4 | `10` | Advances in neural networks for healthca... | 0.039101 |
| 5 | `45` | Clustering for education in bio | 0.035517 |
| 6 | `20` | Advances in sentiment analysis for manuf... | 0.026408 |
| 7 | `95` | Enhancing manufacturing through clusteri... | 0.025908 |
| 8 | `115` | Comparative study of classification meth... | 0.023322 |
| 9 | `80` | Advances in object detection for finance... | 0.021280 |
| 10 | `6` | Clustering for education in cv | 0.019833 |

