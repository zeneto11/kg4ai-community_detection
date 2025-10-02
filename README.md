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

## 🎯 Usage

### Quick Start

The project provides two main entry points for running community detection:

#### Standard Run

For a basic community detection analysis:

```bash
python community_detection/controller/run.py
```

This script:

- Loads the NQ dataset graph
- Runs all four community detection algorithms
- Generates evaluation metrics and comparisons
- Creates visualizations and community reports
- Outputs results to `community_detection/output/runXXX/`

#### Full Pipeline Run

For a complete analysis pipeline:

```bash
python community_detection/controller/full_run.py
```

This includes everything from the standard run plus:

- Extended graph analysis
- Enhanced visualization suite
- Comprehensive community keyword extraction
- Detailed performance benchmarking

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

## 📓 Notebooks

The `notebook/` directory contains Jupyter notebooks for exploratory analysis and method development:

### Analysis Notebooks

- **`nq_dataset.ipynb`**: Main analysis of the Natural Questions dataset
- **`distance_metrics.ipynb`**: Exploration of distance and similarity metrics
- **`macrographs.ipynb`**: Community interaction network analysis
- **`wiki_clustering.ipynb`**: Wikipedia article clustering experiments

### Algorithm-Specific Notebooks

- **`nq_dataset_infomap.ipynb`**: Infomap algorithm deep dive
- **`nq_dataset_louvain.ipynb`**: Louvain algorithm analysis
- **`nq_dataset_leiden.ipynb`**: Leiden algorithm evaluation
- **`nq_dataset_hpmocd.ipynb`**: HPMOCD (hierarchical) algorithm testing

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

## 🔧 Contributing

### Adding New Community Detection Methods

To implement a new community detection algorithm:

1. **Create a new detector class** in `community_detection/methods/`:

```python
# community_detection/methods/my_detector.py
from community_detection.methods.base import CommunityDetector
import networkx as nx

class MyDetector(CommunityDetector):
    def __init__(self, param1=default_value, **kwargs):
        super().__init__(**kwargs)
        self.param1 = param1

    def fit(self, graph: nx.Graph):
        # Implement your algorithm here
        communities = your_algorithm(graph, self.param1)

        # Store results (communities should be List[Set[nodes]])
        self._set_results(communities=communities)
        return self
```

2. **Add to the controller scripts**:

```python
# In run.py or full_run.py
from community_detection.methods.my_detector import MyDetector

detectors = {
    # ... existing detectors
    "MyAlgorithm": MyDetector(param1=value),
}
```

3. **Test your implementation**:
   - Run the pipeline with your new detector
   - Check that outputs are generated correctly
   - Verify metrics are computed properly

### Development Guidelines

- Follow the existing code structure and naming conventions
- Add comprehensive docstrings to new methods
- Include parameter validation in detector constructors
- Test with both directed and undirected graphs
- Update this README if adding major features

## 📚 Research Applications

This framework has been used for:

- **Citation Network Analysis**: Understanding academic paper relationships
- **Knowledge Graph Communities**: Detecting topical clusters in Wikipedia
- **Algorithm Benchmarking**: Comparing community detection methods
- **Network Evolution**: Tracking community changes over time

### Citation

If you use this framework in your research, please cite:

```bibtex
@software{kg4ai_community_detection,
  title={KG4AI Community Detection Framework},
  author={José Almeida Neto},
  year={2025},
  url={https://github.com/zeneto11/kg4ai-community_detection}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**José Almeida Neto**

- Email: josealmeidaneto2002@gmail.com
- GitHub: [@zeneto11](https://github.com/zeneto11)

## 🙏 Acknowledgments

- Natural Questions dataset from Google Research
- NetworkX and igraph communities for graph processing tools
- Infomap, Leiden, and Louvain algorithm developers
- The broader network science research community
