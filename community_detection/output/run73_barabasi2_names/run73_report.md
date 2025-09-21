# Graph Analysis Report

- **Run ID:** run73
- **Timestamp:** 2025-09-18T12:45:41.557826
- **Graph Source:** Simulated Citation Network with Realistic Titles
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
| 1 | 5 | Clustering for education in nlp | 37 |
| 2 | 6 | Classification for healthcare in cv | 35 |
| 3 | 0 | Advances in data mining for finance applications | 31 |
| 4 | 10 | Classification for healthcare in ai | 23 |
| 5 | 7 | Enhancing education through deep learning optimization | 15 |
| 6 | 16 | Enhancing education through statistical analysis optimization | 14 |
| 7 | 25 | Classification for manufacturing in cv | 14 |
| 8 | 12 | Optimization for manufacturing in data | 13 |
| 9 | 1 | A novel approach to finance using statistical analysis | 12 |
| 10 | 3 | Statistical Analysis for finance in nlp | 12 |

### Top 10 Most Citing Nodes

| Rank | Node ID | Title | Citations Made |
|------|---------|-------------------------------|--------------------|
| 1 | 4 | Enhancing education through clustering optimization | 41 |
| 2 | 7 | Enhancing education through deep learning optimization | 22 |
| 3 | 34 | Enhancing healthcare through statistical analysis optimization | 17 |
| 4 | 3 | Statistical Analysis for finance in nlp | 15 |
| 5 | 14 | Advances in sentiment analysis for transportation applications | 15 |
| 6 | 8 | Comparative study of optimization methods in data | 14 |
| 7 | 12 | Optimization for manufacturing in data | 13 |
| 8 | 6 | Classification for healthcare in cv | 12 |
| 9 | 29 | Advances in deep learning for manufacturing applications | 11 |
| 10 | 9 | Advances in image segmentation for education applications | 10 |

### Top 10 Most Influential Nodes (PageRank)

| Rank | Node ID | Title | Influence Score |
|------|---------|-------------------------------|--------------------|
| 1 | 5 | Clustering for education in nlp | 0.086372 |
| 2 | 0 | Advances in data mining for finance applications | 0.079767 |
| 3 | 25 | Classification for manufacturing in cv | 0.042254 |
| 4 | 10 | Classification for healthcare in ai | 0.039101 |
| 5 | 45 | Clustering for education in bio | 0.035517 |
| 6 | 20 | Optimization for finance in nlp | 0.026408 |
| 7 | 95 | Advances in sentiment analysis for finance applications | 0.025908 |
| 8 | 115 | A novel approach to transportation using optimization | 0.023322 |
| 9 | 80 | Clustering for finance in ai | 0.021280 |
| 10 | 6 | Classification for healthcare in cv | 0.019833 |

