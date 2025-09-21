# Graph Analysis Report

- **Run ID:** run60
- **Timestamp:** 2025-09-18T10:30:35.569566
- **Graph Source:** NQ (Natural Questions) from Google
- **Analysis Version:** 2.0

## ⚠️ Scalability Notes

- **Large Graph Warning:** Graph has 108071 nodes - some metrics may be slow
- **Very Large Graph Warning:** Consider disabling expensive metrics for graphs > 50k nodes

## Executive Summary

This analysis examined a **large directed graph** with:

 - 108,071 nodes (entities) and 5,122,983 connections
 - 107,536 nodes remaining after isolation removal (535 isolated nodes removed)
 - Sparse connectivity (density: 0.0%)
 - Low clustering (0.0% - how tightly grouped entities are)
 - Average connections per entity: 95.3

**Key Insights:**
 - The network has 0 disconnected clusters, with the largest containing 0 entities
 - Low connectivity suggests potential for network growth


## Detailed Metrics

### Basic Graph Properties

*Fundamental characteristics of the graph structure*

- **Is Directed:** True
- **Is Weighted:** False
- **Is Multigraph:** False
- **Num Nodes:** 108071
- **Num Edges:** 5122983
- **Density:** 0.0004
- **Self Loops:** 0

### Graph Connectivity

*How well connected the graph components are*

- **Is Connected Undirected:** False

### Node Isolation Analysis

*Statistics about isolated and removed nodes*

- **Num Nodes Zero Total Degree:** 535
- **Num Isolated Nodes Removed:** 535
- **Num Nodes After Removal:** 107536
- **Num Edges After Removal:** 5122983

### Degree Distribution

*Statistical distribution of node connections*

- **Average Degree:** 95.2794
- **Max Degree:** 46453
- **Min Degree:** 1
- **Median Degree:** 46.0000
- **Std Dev Degree:** 293.7996

### Directed Graph Metrics

*Metrics specific to directed graphs (in/out degree, reciprocity, etc.)*

- **Is Strongly Connected:** False
- **Is Weakly Connected:** False
- **Num Weakly Connected Components:** 537
- **Num Strongly Connected Components:** 13288
- **Largest Weakly Connected Component Size:** 107534
- **Largest Strongly Connected Component Size:** 94675
- **Weakly Cc Size Mean:** 201.2495
- **Weakly Cc Size Median:** 1.0000
- **Strongly Cc Size Mean:** 8.1330
- **Strongly Cc Size Median:** 1.0000
- **Num Nodes Zero In Degree:** 11989
- **Num Nodes Zero Out Degree:** 969
- **Reciprocity:** 0.2931
- **Max In Degree:** 46388
- **Min In Degree:** 0
- **Mean In Degree:** 47.6397
- **Median In Degree:** 14.0000
- **Max Out Degree:** 1394
- **Min Out Degree:** 0
- **Mean Out Degree:** 47.6397
- **Median Out Degree:** 28.0000

### Component Analysis

*Detailed analysis of connected components*

- **Weakly Connected Component Sizes:** 537 items (min: 1, max: 107534, mean: 201.25)
- **Strongly Connected Component Sizes:** 13288 items (min: 1, max: 94675, mean: 8.13)

## 📚 Citation Network Analysis

*Specialized analysis for citation/reference networks*

### Citation Distribution

**Citations Made (Out-degree):**
- Mean: 47.64
- Median: 28.00
- Max: 1394
- Standard Deviation: 59.96

**Citations Received (In-degree):**
- Mean: 47.64
- Median: 14.00
- Max: 46388
- Standard Deviation: 269.57
- **Network H-Index:** 657

### Node Categories

- **Isolated Nodes:** 0 (no citations in either direction)
- **Terminal Nodes:** 434 (don't cite others)
- **Source Nodes:** 11454 (not cited by others)

### Top 10 Most Cited Nodes

| Rank | Node ID | Title | Citations Received |
|------|---------|-------------------------------|--------------------|
| 1 | 32515 | International Standard Book Number | 46388 |
| 2 | 93432 | Wikipedia, the free encyclopedia | 46163 |
| 3 | 5115 | United States | 18735 |
| 4 | 63708 | Digital object identifier | 15092 |
| 5 | 19755 | IMDb | 14851 |
| 6 | 10883 | Library of Congress Control Number | 10507 |
| 7 | 8341 | United Kingdom | 10077 |
| 8 | 21518 | The New York Times | 8936 |
| 9 | 17568 | Single (music) | 7564 |
| 10 | 46202 | Geographic coordinate system | 7536 |

### Top 10 Most Citing Nodes

| Rank | Node ID | Title | Citations Made |
|------|---------|-------------------------------|--------------------|
| 1 | 5115 | United States | 1394 |
| 2 | 59377 | Timeline of United States history | 1153 |
| 3 | 17571 | 2017 in film | 998 |
| 4 | 5565 | List of performances on Top of the Pops | 998 |
| 5 | 4831 | Southern United States | 912 |
| 6 | 5613 | History of the United States | 881 |
| 7 | 14656 | New England | 856 |
| 8 | 22976 | List of Latin phrases (full) | 843 |
| 9 | 25737 | California | 837 |
| 10 | 8341 | United Kingdom | 829 |

### Top 10 Most Influential Nodes (PageRank)

| Rank | Node ID | Title | Influence Score |
|------|---------|-------------------------------|--------------------|
| 1 | 32515 | International Standard Book Number | 0.009264 |
| 2 | 93432 | Wikipedia, the free encyclopedia | 0.006816 |
| 3 | 5115 | United States | 0.003430 |
| 4 | 63708 | Digital object identifier | 0.003043 |
| 5 | 10883 | Library of Congress Control Number | 0.002761 |
| 6 | 19755 | IMDb | 0.002462 |
| 7 | 8341 | United Kingdom | 0.001983 |
| 8 | 10642 | New York City | 0.001554 |
| 9 | 21518 | The New York Times | 0.001496 |
| 10 | 46202 | Geographic coordinate system | 0.001375 |

