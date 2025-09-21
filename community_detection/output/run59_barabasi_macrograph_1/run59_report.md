# Graph Analysis Report

- **Run ID:** run59
- **Timestamp:** 2025-09-18T03:29:29.299922
- **Graph Source:** Simulated Citation Network
- **Analysis Version:** 2.0

## Executive Summary

This analysis examined a **medium directed graph** with:

 - 200 nodes (entities) and 714 connections
 - 0 nodes remaining after isolation removal (0 isolated nodes removed)
 - Sparse connectivity (density: 0.0%)
 - Low clustering (0.0% - how tightly grouped entities are)
 - Average connections per entity: 7.1

**Key Insights:**
 - The network is fully connected (all entities are reachable)
 - Low connectivity suggests potential for network growth


## Detailed Metrics

### Basic Graph Properties

*Fundamental characteristics of the graph structure*

- **Is Directed:** True
- **Is Weighted:** False
- **Is Multigraph:** False
- **Num Nodes:** 200
- **Num Edges:** 714
- **Density:** 0.0179
- **Self Loops:** 0

### Graph Connectivity

*How well connected the graph components are*

- **Is Connected Undirected:** True

### Node Isolation Analysis

*Statistics about isolated and removed nodes*

- **Num Nodes Zero Total Degree:** 0

### Degree Distribution

*Statistical distribution of node connections*

- **Average Degree:** 7.1400
- **Max Degree:** 48
- **Min Degree:** 3
- **Median Degree:** 5.0000
- **Std Dev Degree:** 7.2355

### Clustering Analysis

*How tightly connected local neighborhoods are (⚠️ expensive for large graphs)*

- **Transitivity:** 0.0379
- **Average Clustering:** 0.057305
- **Filtered Average Clustering:** 0.057305

### Distance & Path Analysis

*Shortest paths and network diameter (⚠️ very expensive for large graphs)*

- **Average Shortest Path Length:** 2.8346
- **Diameter:** 5
- **Radius:** 3
- **Global Efficiency:** 0.383012

### Node Centrality

*Measures of node importance and influence (⚠️ expensive for large graphs)*

- **Average Degree Centrality:** 0.035879
- **Average Closeness Centrality:** 0.106965
- **Average Betweenness Centrality:** 0.003903
- **Average Eigenvector Centrality:** 0.021873

### Structural Features

*Advanced structural characteristics and patterns*

- **Degree Assortativity:** -0.164822
- **Max Core Number:** 3
- **Mean Core Number:** 3.0000

### Directed Graph Metrics

*Metrics specific to directed graphs (in/out degree, reciprocity, etc.)*

- **Is Strongly Connected:** False
- **Is Weakly Connected:** True
- **Num Weakly Connected Components:** 1
- **Num Strongly Connected Components:** 87
- **Largest Weakly Connected Component Size:** 200
- **Largest Strongly Connected Component Size:** 27
- **Weakly Cc Size Mean:** 200.0000
- **Weakly Cc Size Median:** 200.0000
- **Strongly Cc Size Mean:** 2.2989
- **Strongly Cc Size Median:** 1.0000
- **Num Nodes Zero In Degree:** 18
- **Num Nodes Zero Out Degree:** 14
- **Reciprocity:** 0.3445
- **Max In Degree:** 37
- **Min In Degree:** 0
- **Mean In Degree:** 3.5700
- **Median In Degree:** 2.0000
- **Max Out Degree:** 41
- **Min Out Degree:** 0
- **Mean Out Degree:** 3.5700
- **Median Out Degree:** 3.0000
- **Average Pagerank:** 0.0050
- **Max Pagerank:** 0.0864

### Component Analysis

*Detailed analysis of connected components*

- **Weakly Connected Component Sizes:** [200]
- **Strongly Connected Component Sizes:** 87 items (min: 1, max: 27, mean: 2.30)

## 📚 Citation Network Analysis

*Specialized analysis for citation/reference networks*

### Citation Distribution

**Citations Made (Out-degree):**
- Mean: 3.57
- Median: 3.00
- Max: 41
- Standard Deviation: 4.05

**Citations Received (In-degree):**
- Mean: 3.57
- Median: 2.00
- Max: 37
- Standard Deviation: 4.89
- **Network H-Index:** 11

### Node Categories

- **Isolated Nodes:** 0 (no citations in either direction)
- **Terminal Nodes:** 14 (don't cite others)
- **Source Nodes:** 18 (not cited by others)

### Top 10 Most Cited Nodes

| Rank | Node ID | Title | Citations Received |
|------|---------|-------------------------------|--------------------|
| 1 | 5 | Paper_005 | 37 |
| 2 | 6 | Paper_006 | 35 |
| 3 | 0 | Paper_000 | 31 |
| 4 | 10 | Paper_010 | 23 |
| 5 | 7 | Paper_007 | 15 |
| 6 | 16 | Paper_016 | 14 |
| 7 | 25 | Paper_025 | 14 |
| 8 | 12 | Paper_012 | 13 |
| 9 | 1 | Paper_001 | 12 |
| 10 | 3 | Paper_003 | 12 |

### Top 10 Most Citing Nodes

| Rank | Node ID | Title | Citations Made |
|------|---------|-------------------------------|--------------------|
| 1 | 4 | Paper_004 | 41 |
| 2 | 7 | Paper_007 | 22 |
| 3 | 34 | Paper_034 | 17 |
| 4 | 3 | Paper_003 | 15 |
| 5 | 14 | Paper_014 | 15 |
| 6 | 8 | Paper_008 | 14 |
| 7 | 12 | Paper_012 | 13 |
| 8 | 6 | Paper_006 | 12 |
| 9 | 29 | Paper_029 | 11 |
| 10 | 9 | Paper_009 | 10 |

### Top 10 Most Influential Nodes (PageRank)

| Rank | Node ID | Title | Influence Score |
|------|---------|-------------------------------|--------------------|
| 1 | 5 | Paper_005 | 0.086372 |
| 2 | 0 | Paper_000 | 0.079767 |
| 3 | 25 | Paper_025 | 0.042254 |
| 4 | 10 | Paper_010 | 0.039101 |
| 5 | 45 | Paper_045 | 0.035517 |
| 6 | 20 | Paper_020 | 0.026408 |
| 7 | 95 | Paper_095 | 0.025908 |
| 8 | 115 | Paper_115 | 0.023322 |
| 9 | 80 | Paper_080 | 0.021280 |
| 10 | 6 | Paper_006 | 0.019833 |

