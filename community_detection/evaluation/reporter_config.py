"""
Configuration for enhanced reporting system.
"""

# Metric categories and their display properties
METRIC_CATEGORIES = {
    "basic_properties": {
        "title": "Basic Graph Properties",
        "description": "Fundamental characteristics of the graph structure",
        "metrics": {
            "is_directed": {"label": "Directed Graph", "format": "bool", "description": "Whether edges have direction"},
            "is_weighted": {"label": "Weighted Graph", "format": "bool", "description": "Whether edges have weights"},
            "num_nodes": {"label": "Number of Nodes", "format": "int", "description": "Total entities in the graph"},
            "num_edges": {"label": "Number of Edges", "format": "int", "description": "Total connections in the graph"}
        }
    },
    "connectivity": {
        "title": "Graph Connectivity",
        "description": "How well connected the graph components are",
        "metrics": {
            "is_connected_undirected": {"label": "Fully Connected", "format": "bool", "description": "Whether all nodes are reachable"},
            "num_connected_components": {"label": "Connected Components", "format": "int", "description": "Number of disconnected subgraphs"},
            "largest_connected_component_size": {"label": "Largest Component Size", "format": "int", "description": "Size of biggest connected subgraph"}
        }
    },
    "degree_statistics": {
        "title": "Degree Statistics",
        "description": "Distribution of node connections",
        "metrics": {
            "average_degree": {"label": "Average Degree", "format": "float_2", "description": "Average connections per node"},
            "max_degree": {"label": "Maximum Degree", "format": "int", "description": "Highest number of connections for any node"},
            "median_degree": {"label": "Median Degree", "format": "float_1", "description": "Middle value of connection distribution"}
        }
    },
    "advanced_metrics": {
        "title": "Advanced Network Metrics",
        "description": "Complex structural properties",
        "metrics": {
            "density": {"label": "Network Density", "format": "percent_4", "description": "Proportion of possible connections that exist"},
            "transitivity": {"label": "Global Clustering", "format": "percent_4", "description": "Tendency to form triangles"},
            "average_clustering": {"label": "Average Clustering", "format": "percent_4", "description": "Local clustering coefficient"},
            "diameter": {"label": "Network Diameter", "format": "int", "description": "Longest shortest path in the network"},
            "global_efficiency": {"label": "Global Efficiency", "format": "percent_4", "description": "How efficiently information spreads"}
        }
    },
    "centrality_measures": {
        "title": "Node Centrality",
        "description": "Measures of node importance",
        "metrics": {
            "average_degree_centrality": {"label": "Avg Degree Centrality", "format": "percent_6", "description": "Average importance based on connections"},
            "average_closeness_centrality": {"label": "Avg Closeness Centrality", "format": "percent_6", "description": "Average importance based on distance to others"},
            "average_betweenness_centrality": {"label": "Avg Betweenness Centrality", "format": "percent_6", "description": "Average importance as network bridge"}
        }
    },
    "structural_features": {
        "title": "Structural Features",
        "description": "Specialized network characteristics",
        "metrics": {
            "num_bridges": {"label": "Bridge Edges", "format": "int", "description": "Critical connections that would disconnect the network"},
            "degree_assortativity": {"label": "Degree Assortativity", "format": "float_6", "description": "Tendency of similar-degree nodes to connect"},
            "max_core_number": {"label": "Maximum k-Core", "format": "int", "description": "Highest density subgraph core number"}
        }
    }
}

# Display formatting functions


def format_metric_value(value, format_type):
    """Format metric values according to their type."""
    if value is None:
        return "N/A"

    if format_type == "bool":
        return "Yes" if value else "No"
    elif format_type == "int":
        return f"{int(value):,}"
    elif format_type.startswith("float_"):
        decimals = int(format_type.split("_")[1])
        return f"{float(value):.{decimals}f}"
    elif format_type.startswith("percent_"):
        decimals = int(format_type.split("_")[1])
        return f"{float(value)*100:.{decimals}f}%" if format_type == "percent_4" else f"{float(value):.{decimals}f}"
    else:
        return str(value)


# Interpretation thresholds and messages
INTERPRETATION_RULES = {
    "density": {
        "thresholds": [0.05, 0.2, 0.5],
        "labels": ["Very Sparse", "Sparse", "Moderate", "Dense"],
        "descriptions": [
            "Very low connectivity - network has room for growth",
            "Low connectivity - loosely connected structure",
            "Moderate connectivity - balanced network structure",
            "High connectivity - tightly integrated network"
        ]
    },
    "average_clustering": {
        "thresholds": [0.1, 0.3, 0.6],
        "labels": ["Low Clustering", "Moderate Clustering", "High Clustering", "Very High Clustering"],
        "descriptions": [
            "Low clustering - nodes rarely form tight groups",
            "Moderate clustering - some local community structure",
            "High clustering - strong local community structure",
            "Very high clustering - highly cohesive local groups"
        ]
    },
    "num_connected_components": {
        "thresholds": [1, 5, 20],
        "labels": ["Fully Connected", "Few Components", "Many Components", "Highly Fragmented"],
        "descriptions": [
            "Single connected network - all nodes reachable",
            "Few disconnected components - mostly connected",
            "Multiple components - moderately fragmented",
            "Many components - highly fragmented network"
        ]
    }
}


def interpret_metric(metric_name, value):
    """Provide human-readable interpretation of metric values."""
    if metric_name not in INTERPRETATION_RULES or value is None:
        return None

    rules = INTERPRETATION_RULES[metric_name]
    thresholds = rules["thresholds"]
    labels = rules["labels"]
    descriptions = rules["descriptions"]

    for i, threshold in enumerate(thresholds):
        if value <= threshold:
            return {"label": labels[i], "description": descriptions[i]}

    return {"label": labels[-1], "description": descriptions[-1]}


# Report templates
EXECUTIVE_SUMMARY_TEMPLATE = """
## Executive Summary

This analysis examined a **{graph_type}** network with **{num_nodes:,} entities** and **{num_edges:,} connections**.

### Key Characteristics
- **Network Density:** {density_interpretation}
- **Connectivity:** {connectivity_interpretation}  
- **Community Structure:** {clustering_interpretation}
- **Average Connections:** {avg_degree:.1f} per entity

### Business Insights
{business_insights}
"""

# Logging formats
LOG_FORMATS = {
    "structured": "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    "simple": "%(asctime)s - %(levelname)s - %(message)s",
    "detailed": "%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s:%(lineno)-4d | %(message)s"
}

# Output file configurations
OUTPUT_CONFIGS = {
    "json": {
        "enabled": True,
        "filename_template": "{run_id}_metrics.json",
        "indent": 2
    },
    "csv": {
        "enabled": True,
        "filename_template": "{run_id}_summary.csv",
        "include_metadata": True
    },
    "markdown": {
        "enabled": True,
        "filename_template": "{run_id}_report.md",
        "include_executive_summary": True
    },
    "html_dashboard": {
        "enabled": False,  # Optional feature
        "filename_template": "{run_id}_dashboard.html",
        "include_visualizations": True
    }
}
