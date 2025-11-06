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
- **TPR (Triad Participation Ratio)**: Community detection accuracy
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

The `notebook/` directory contains Jupyter notebooks for exploratory analysis and method development:

- **`exploration.ipynb`**: Exploratory data analysis of the graph structure
- **`extract_categories.ipynb`**: Category extraction from Wikipedia data
- **`extract_links.ipynb`**: Link extraction and graph construction
- **`nq_dataset.ipynb`**: Natural Questions dataset processing
- **`wiki_comparison.ipynb`**: Comparison with Wikipedia clustering results
- **`methods_study/`**: In-depth studies of detection algorithms
  - `leiden.ipynb`: Leiden algorithm analysis
  - `louvain.ipynb`: Louvain algorithm analysis
  - `infomap.ipynb` & `infomap_first_test.ipynb`: Infomap algorithm studies
  - `hpmocd.ipynb`: HP-MOCD algorithm exploration
  - `ml-clustering.ipynb`: Machine learning-based clustering approaches
- **`metrics_study/`**: Metric analysis and optimization
  - `distance_metrics.ipynb`: Distance metric comparisons
  - `macrographs.ipynb`: Macro-graph visualization techniques
- **`sample/`**: Sample analyses
  - `subgraph_community.ipynb`: Community detection on subgraphs
- **`imported/`**: External notebooks
  - `wikipedia_knowledge_graph_anderson.ipynb`: Wikipedia KG construction

## 📁 Project Structure

```
kg4ai-community_detection/
├── community_detection/          # Main package for community detection
│   ├── controller/               # Pipeline orchestration
│   │   └── full_run.py          # Complete pipeline execution
│   ├── methods/                  # Detection algorithms
│   │   ├── base.py              # Base detector interface
│   │   ├── leiden_detector.py   # Leiden algorithm
│   │   ├── louvain_detector.py  # Louvain algorithm
│   │   ├── infomap_detector.py  # Infomap algorithm
│   │   ├── kmeans_detector.py   # K-means clustering
│   │   └── hpmocd_detector.py   # HP-MOCD algorithm
│   ├── evaluation/               # Evaluation and metrics
│   │   ├── evaluator.py         # Main evaluation controller
│   │   ├── graph_metrics.py     # Graph-level metrics
│   │   ├── community_metrics.py # Community-level metrics
│   │   ├── reporter.py          # Report generation
│   │   └── community_report_extension.py  # Extended reporting
│   ├── visualization/            # Visualization tools
│   │   ├── visualizer.py        # Main visualization controller
│   │   ├── community_visualizer.py  # Community-specific plots
│   │   └── community_namer.py   # Community naming with keywords
│   ├── utils/                    # Utility modules
│   │   ├── graph_utility.py     # Graph manipulation helpers
│   │   ├── logger.py            # Logging configuration
│   │   ├── run_manager.py       # Run directory management
│   │   ├── metrics_status.py    # Metrics tracking
│   │   └── time.py              # Time formatting utilities
│   └── output/                   # Pipeline outputs (generated)
│       └── runXXX_[description]/ # Individual run results
├── community_analysis/           # Advanced community analysis
│   ├── analisar_comunidades.py  # LLM-based community categorization
│   ├── nodes.json               # Node data for analysis
│   ├── run142_raw_communities.json        # Raw community data
│   ├── run142_community_keywords.json     # Extracted keywords
│   ├── analise_comunidades_leiden_mistral.md  # Analysis report
│   └── analise_comunidades_leiden_mistral_detailed.json  # Detailed results
├── data/                         # Datasets
│   ├── v0.0/                    # Version 0 data
│   │   ├── df_nq_version0.csv   # Natural Questions dataset
│   │   └── nodes.json           # Graph nodes data
│   ├── chatgpt_test/            # ChatGPT test data
│   └── wiki_clustering_results/ # Wikipedia clustering benchmarks
├── notebook/                     # Jupyter notebooks (see above)
├── pyproject.toml               # Poetry dependencies and config
└── README.md                    # This file
```

### Key Components

- **`community_detection/`**: Core framework with modular architecture for detection, evaluation, and visualization
- **`community_analysis/`**: Advanced analysis tools including LLM-based community categorization with Ollama/Mistral
- **`data/`**: Input datasets including Natural Questions citation network
- **`notebook/`**: Research notebooks for experimentation and validation

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
