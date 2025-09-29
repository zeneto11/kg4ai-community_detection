# 📊 Graph Analysis Report

---

## 📋 Analysis Metadata

| Property | Value |
|----------|-------|
| **Run ID** | `run129` |
| **Timestamp** | 2025-09-29T11:43:11.492952 |
| **Graph Source** | NQ (Natural Questions) from Google |
| **Analysis Version** | 3.0 |
| **Total Time** | 45s |

## 🎯 Executive Summary

**Network Overview:**
- **Scale:** Very Large directed network
- **Original Size:** 108,071 nodes, 5,122,983 edges
- **Post-Cleaning Size:** 107,534 nodes, 5,122,982 edges (removed 0.68% of nodes)
- **Connectivity:** sparse (density: 0.04430%)
- **Average Connections:** 95.3 per node

**Key Characteristics:**
- ✅ **Fully Connected:** All nodes are reachable
- 📈 **Growth Potential:** Very sparse network with room for expansion

---

## 📈 Enhanced Data Visualizations

### Comprehensive Degree Analysis
![Degree Analysis](visualizations/run129_degree_analysis.png)
*Four-panel analysis showing linear distribution, log-log scaling for power-law detection, cumulative distribution, and statistical summary via box plot.*

---

## 📊 Detailed Metrics

### 🏗️ Graph Structure

*Fundamental properties of the graph topology*

| Metric | Value | Status |
|--------|-------|--------|
| **Is Directed** | 1 | ✅ computed |
| **Is Weighted** | 0 | ✅ computed |
| **Is Multigraph** | 0 | ✅ computed |
| **Num Nodes** | 108,071 | ✅ computed |
| **Num Edges** | 5,122,983 | ✅ computed |

---

### 🔗 Connectivity Analysis

*Network connectivity and density measurements*

| Metric | Value | Status |
|--------|-------|--------|
| **Density** | 0.0004 | ✅ computed |
| **Is Connected Undirected** | 1 | ✅ computed |
| **Reciprocity** | 0.2931 | ✅ computed |
| **Is Strongly Connected** | 0 | ✅ computed |
| **Is Weakly Connected** | 1 | ✅ computed |

---

### 🏝️ Node Isolation Analysis

*Statistics about disconnected and removed nodes*

| Metric | Value | Status |
|--------|-------|--------|
| **Num Nodes Zero Total Degree** | 535 | ✅ computed |
| **Num Isolated Nodes Removed** | 535 | ✅ computed |
| **Min Component Size Threshold** | 108 | ✅ computed |
| **Num Small Components Removed** | 1 | ✅ computed |
| **Num Nodes In Small Components Removed** | 2 | ✅ computed |
| **Total Nodes Removed** | 537 | ✅ computed |
| **Num Nodes After Removal** | 107,534 | ✅ computed |
| **Num Edges After Removal** | 5,122,982 | ✅ computed |
| **Removal Percentage** | 0.68% | ✅ computed |

---

### 📈 Degree Distribution

*Statistical analysis of node connection patterns*

| Metric | Value | Status |
|--------|-------|--------|
| **Average Degree** | 95.28 | ✅ computed |
| **Max Degree** | 46,453 | ✅ computed |
| **Min Degree** | 1 | ✅ computed |
| **Median Degree** | 46.0 | ✅ computed |
| **Std Dev Degree** | 293.8 | ✅ computed |

---

### ➡️ Directed Graph Properties

*Metrics specific to directed networks (in/out degrees, PageRank)*

| Metric | Value | Status |
|--------|-------|--------|
| **Num Nodes Zero In Degree** | 11,453 | ✅ computed |
| **Max In Degree** | 46,388 | ✅ computed |
| **Min In Degree** | 0 | ✅ computed |
| **Mean In Degree** | 47.6 | ✅ computed |
| **Median In Degree** | 14.0 | ✅ computed |
| **Num Nodes Zero Out Degree** | 433 | ✅ computed |
| **Max Out Degree** | 1,394 | ✅ computed |
| **Min Out Degree** | 0 | ✅ computed |
| **Mean Out Degree** | 47.6 | ✅ computed |
| **Median Out Degree** | 28.0 | ✅ computed |

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
| **Largest Weakly Connected Component Size** | 107,534 | ✅ computed |
| **Weakly Cc Size Mean** | 107534.00 | ✅ computed |
| **Weakly Cc Size Median** | 107534.00 | ✅ computed |
| **Weakly Connected Component Sizes** | [107534] | ✅ computed |
| **Num Strongly Connected Components** | 12,751 | ✅ computed |
| **Largest Strongly Connected Component Size** | 94,675 | ✅ computed |
| **Strongly Cc Size Mean** | 8.43 | ✅ computed |
| **Strongly Cc Size Median** | 1.00 | ✅ computed |
| **Strongly Connected Component Sizes** | 12751 items (μ=8.43, σ=838.38) | ✅ computed |

---

## 📚 Citation Network Analysis

*Specialized analysis for citation/reference networks*

### Citation Statistics

| Metric | Citations Made (Out-degree) | Citations Received (In-degree) |
|--------|------------------------------|----------------------------------|
| **Mean** | 47.64 | 47.64 |
| **Median** | 28.00 | 14.00 |
| **Maximum** | 1394 | 46388 |
| **Std Dev** | 59.96 | 269.58 |

**Network H-Index:** 657

### Node Categories

- 🏝️ **Isolated Nodes:** 0 (no citations)
- 🔚 **Terminal Nodes:** 433 (don't cite others)
- 🌱 **Source Nodes:** 11453 (not cited by others)

### 🏆 Top 10 Most Cited Papers

| Rank | Node ID | Title | Citations |
|------|---------|-------|----------|
| 1 | `32515` | International Standard Book Number | **46388** |
| 2 | `93432` | Wikipedia, the free encyclopedia | **46163** |
| 3 | `5115` | United States | **18735** |
| 4 | `63708` | Digital object identifier | **15092** |
| 5 | `19755` | IMDb | **14851** |
| 6 | `10883` | Library of Congress Control Number | **10507** |
| 7 | `8341` | United Kingdom | **10077** |
| 8 | `21518` | The New York Times | **8936** |
| 9 | `17568` | Single (music) | **7564** |
| 10 | `46202` | Geographic coordinate system | **7536** |

### 📝 Top 10 Most Citing Papers

| Rank | Node ID | Title | Citations Made |
|------|---------|-------|----------------|
| 1 | `5115` | United States | **1394** |
| 2 | `59377` | Timeline of United States history | **1153** |
| 3 | `17571` | 2017 in film | **998** |
| 4 | `5565` | List of performances on Top of the Pops | **998** |
| 5 | `4831` | Southern United States | **912** |
| 6 | `5613` | History of the United States | **881** |
| 7 | `14656` | New England | **856** |
| 8 | `22976` | List of Latin phrases (full) | **843** |
| 9 | `25737` | California | **837** |
| 10 | `8341` | United Kingdom | **829** |

### 🌟 Top 10 Most Influential Papers (PageRank)

| Rank | Node ID | Title | Influence Score |
|------|---------|-------|----------------|
| 1 | `32515` | International Standard Book Number | 0.009264 |
| 2 | `93432` | Wikipedia, the free encyclopedia | 0.006816 |
| 3 | `5115` | United States | 0.003430 |
| 4 | `63708` | Digital object identifier | 0.003043 |
| 5 | `10883` | Library of Congress Control Number | 0.002761 |
| 6 | `19755` | IMDb | 0.002462 |
| 7 | `8341` | United Kingdom | 0.001983 |
| 8 | `10642` | New York City | 0.001554 |
| 9 | `21518` | The New York Times | 0.001496 |
| 10 | `46202` | Geographic coordinate system | 0.001375 |

