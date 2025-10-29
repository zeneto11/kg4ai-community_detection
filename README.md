# 🏘️ KG4AI Community Detection

A comprehensive research and engineering framework for community detection in graphs, featuring multiple algorithms, evaluation metrics, visualization tools, and automated reporting pipelines.

## 📋 Overview

This project provides a robust platform for analyzing community structures in complex networks. It supports multiple state-of-the-art community detection algorithms, comprehensive evaluation metrics, and generates detailed reports with visualizations. Originally designed for analyzing citation networks from the Natural Questions (NQ) dataset, the framework is extensible to various graph types.

### Key Features

- 🔍 **Multiple Algorithms**: Infomap, Louvain, Leiden, K-means clustering
- 📊 **Comprehensive Evaluation**: Modularity, conductance, homophily, clustering metrics
- 📈 **Rich Visualizations**: Community graphs, size distributions, macro-graphs
- 📝 **Automated Reporting**: Detailed markdown reports with metrics and plots
- 🔬 **Research-Ready**: Jupyter notebooks for analysis and experimentation
- 🎯 **Extensible Architecture**: Easy to add new detection methods

## 🚀 Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management. Make sure you have Poetry installed on your system.

### Prerequisites

- Python 3.12 or 3.13
- Poetry (for dependency management)

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/zeneto11/kg4ai-community_detection.git
   cd kg4ai-community_detection
   ```

2. **Install dependencies with Poetry:**

   ```bash
   poetry install
   ```

3. **Activate the virtual environment:**
   ```bash
   poetry shell
   ```

### Dependencies

The project includes the following key dependencies:

- **Graph Processing**: `networkx`, `python-igraph`
- **Community Detection**: `infomap`, `leidenalg`, `python-louvain`
- **Machine Learning**: `scikit-learn`, `sentence-transformers`, `umap-learn`
- **Visualization**: `matplotlib`
- **Data Processing**: `pandas`, `numpy`

## 🎯 Pipeline

For a complete analysis pipeline:

```bash
poetry run python -m community_detection.controller.full_run
```

This will execute the **entire community detection pipeline**, including:

1. **Graph Construction** – builds a citation graph from the dataset.
2. **Graph Analysis** – computes key structural metrics and statistics.
3. **Community Detection** – runs multiple algorithms (Infomap, Louvain, Leiden, and K-means).
4. **Evaluation & Comparison** – compares algorithms using internal and structural metrics.
5. **Visualization & Reporting** – produces plots, keyword summaries, and a Markdown report.

### Output Structure

Each run creates a timestamped directory in `community_detection/output/` containing:

```
runXXX_[description]/
├── pipeline.log                    # Execution log
├── runXXX_metrics.json            # Graph and community metrics
├── runXXX_comparison.csv          # Algorithm comparison table
├── runXXX_community_results.json  # Detailed community data
├── runXXX_community_keywords.json # Community keywords and names
├── runXXX_raw_communities.json    # Raw algorithm outputs
├── runXXX_report.md               # Comprehensive analysis report
└── plots/                         # Visualization outputs
    ├── degree_analysis.png
    ├── [algorithm]_community_size_barplot.png
    ├── [algorithm]_macrograph_weighted.png
    └── [algorithm]_macrograph_thresholded.png
```

### Example: Understanding the Results

For reference, examine the `run142-NQv0-best_run` output which demonstrates:

- **Graph Analysis**: 107,534 nodes, 5.1M edges from Wikipedia citation network
- **Algorithm Performance**: Leiden achieved highest modularity (0.5993)
- **Community Insights**: 19-1,249 communities detected depending on algorithm
- **Visualizations**: Community size distributions and interaction networks
- **Detailed Report**: 40+ metrics with explanations and recommendations

## 📊 Output Explanation

### Core Metrics

- **Modularity**: Quality of community division (higher = better)
- **Conductance**: Community boundary quality (lower = better)
- **Homophily**: Tendency of similar nodes to cluster
- **TPR (True Positive Rate)**: Community detection accuracy
- **Clustering Coefficient**: Local connectivity within communities

### Visualizations

1. **Community Size Barplots**: Distribution of community sizes
2. **Weighted Macrographs**: Inter-community connection strengths
3. **Thresholded Macrographs**: Significant community interactions only
4. **Degree Analysis**: Node connectivity patterns and power-law detection

### JSON Outputs

- **`metrics.json`**: All computed graph and community metrics
- **`community_results.json`**: Structured community data with metadata
- **`community_keywords.json`**: Extracted keywords and community names
- **`raw_communities.json`**: Direct algorithm outputs for reproducibility

## 📓 Notebooks

The `notebook/` directory contains Jupyter notebooks for exploratory analysis and method development.

## Authors

**José Almeida Neto**

- Email: josealmeidaneto2002@gmail.com
- GitHub: [@zeneto11](https://github.com/zeneto11)

**Anderson Luis Bento Soares**

- Email: anderson.soares@students.ic.unicamp.br
- GitHub: [@andersonlbsoares](https://github.com/andersonlbsoares)

## Acknowledgments

- Natural Questions dataset from Google Research
- NetworkX and igraph communities for graph processing tools
- Infomap, Leiden, and Louvain algorithm developers
- The broader network science research community

## References

- [Defining and Evaluating Network Communities based on Ground-truth](https://arxiv.org/pdf/1205.6233)
- [Community Detection with the Map Equation and Infomap:Theory and Applications](https://arxiv.org/pdf/2311.04036)
- [GVE-Louvain: Fast Louvain Algorithm for Community Detection in Shared Memory Setting](https://arxiv.org/html/2312.04876v4)
