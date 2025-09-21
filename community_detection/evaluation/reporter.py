import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import networkx as nx

from community_detection.utils.logger import log_substep


class GraphMetricsReporter:
    """Enhanced reporter for graph metrics with multiple output formats."""

    def __init__(self, output_dir: Path, run_id: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = logging.getLogger(__name__)

    def generate_reports(self, metrics: Dict[str, Any], graph_info: Dict[str, Any] = None,
                         citation_metrics: Dict[str, Any] = None):
        """Generate multiple report formats from metrics."""
        # Add metadata
        enhanced_metrics = self._add_metadata(metrics, graph_info)

        # Generate different formats
        self._save_json_report(enhanced_metrics)
        self._save_human_readable_report(enhanced_metrics, citation_metrics)
        self._log_executive_summary(enhanced_metrics)

        return enhanced_metrics

    def _add_metadata(self, metrics: Dict[str, Any], graph_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add metadata and structure to metrics with scalability flags."""
        enhanced = {
            "metadata": {
                "run_id": self.run_id,
                "timestamp": datetime.now().isoformat(),
                "analysis_version": "2.0",
                "graph_source": graph_info.get("source", "unknown") if graph_info else "unknown",
                "scalability_notes": {}
            },
            "basic_properties": {},
            "connectivity": {},
            "node_isolation": {},
            "degree_statistics": {},
            "clustering_metrics": {},
            "distance_metrics": {},
            "centrality_measures": {},
            "structural_features": {},
            "directed_graph_metrics": {},
            "component_analysis": {}
        }

        # Categorize metrics with scalability considerations
        basic_keys = ["is_directed", "is_weighted", "is_multigraph",
                      "num_nodes", "num_edges", "density", "self_loops"]

        connectivity_keys = ["is_connected_undirected", "num_connected_components",
                             "largest_connected_component_size", "cc_size_mean", "cc_size_median"]

        isolation_keys = ["num_nodes_zero_total_degree", "num_isolated_nodes_removed",
                          "num_nodes_after_removal", "num_edges_after_removal"]

        degree_keys = ["average_degree", "max_degree",
                       "min_degree", "median_degree", "std_dev_degree"]

        # SCALABILITY WARNING: These clustering metrics are O(n^3) for transitivity, O(n*m) for clustering
        clustering_keys = ["transitivity",
                           "average_clustering", "filtered_average_clustering"]

        # SCALABILITY WARNING: Distance metrics are very expensive O(n^2) to O(n^3)
        distance_keys = ["average_shortest_path_length", "diameter", "radius",
                         "largest_cc_average_shortest_path_length", "largest_cc_diameter",
                         "largest_cc_radius", "global_efficiency"]

        # SCALABILITY WARNING: Centrality measures range from O(n^2) to O(n^3)
        centrality_keys = ["average_degree_centrality", "average_closeness_centrality",
                           "average_betweenness_centrality", "average_eigenvector_centrality"]

        # SCALABILITY NOTE: Most structural features are reasonable O(n+m) to O(n*m)
        structural_keys = ["num_bridges", "degree_assortativity", "max_core_number",
                           "mean_core_number", "rich_club_coefficient"]

        # Directed graph specific metrics
        directed_keys = ["is_strongly_connected", "is_weakly_connected", "reciprocity",
                         "num_weakly_connected_components", "num_strongly_connected_components",
                         "largest_weakly_connected_component_size", "largest_strongly_connected_component_size",
                         "weakly_cc_size_mean", "weakly_cc_size_median", "strongly_cc_size_mean",
                         "strongly_cc_size_median",
                         "num_nodes_zero_in_degree", "num_nodes_zero_out_degree",
                         "max_in_degree", "min_in_degree", "mean_in_degree", "median_in_degree",
                         "max_out_degree", "min_out_degree", "mean_out_degree", "median_out_degree",
                         "average_pagerank", "max_pagerank"]

        # Component analysis (lists and complex data)
        component_keys = ["connected_component_sizes", "weakly_connected_component_sizes",
                          "strongly_connected_component_sizes", "largest_cc_size", "largest_cc_fraction"]

        # Add scalability warnings to metadata
        num_nodes = metrics.get("num_nodes", 0)
        if num_nodes > 10000:
            enhanced["metadata"]["scalability_notes"][
                "large_graph_warning"] = f"Graph has {num_nodes} nodes - some metrics may be slow"
        if num_nodes > 50000:
            enhanced["metadata"]["scalability_notes"]["very_large_graph_warning"] = "Consider disabling expensive metrics for graphs > 50k nodes"

        # Categorize metrics
        for key, value in metrics.items():
            if key in basic_keys:
                enhanced["basic_properties"][key] = value
            elif key in connectivity_keys:
                enhanced["connectivity"][key] = value
            elif key in isolation_keys:
                enhanced["node_isolation"][key] = value
            elif key in degree_keys:
                enhanced["degree_statistics"][key] = value
            elif key in clustering_keys:
                enhanced["clustering_metrics"][key] = value
                if num_nodes > 10000 and key in ["transitivity", "average_clustering"]:
                    enhanced["metadata"]["scalability_notes"][f"{key}_warning"] = "Expensive for large graphs"
            elif key in distance_keys:
                enhanced["distance_metrics"][key] = value
                if num_nodes > 5000:
                    enhanced["metadata"]["scalability_notes"][
                        f"{key}_warning"] = "Very expensive for large graphs"
            elif key in centrality_keys:
                enhanced["centrality_measures"][key] = value
                if num_nodes > 1000 and key in ["average_betweenness_centrality", "average_eigenvector_centrality"]:
                    enhanced["metadata"]["scalability_notes"][
                        f"{key}_warning"] = "Extremely expensive for large graphs"
            elif key in structural_keys:
                enhanced["structural_features"][key] = value
            elif key in directed_keys:
                enhanced["directed_graph_metrics"][key] = value
            elif key in component_keys:
                enhanced["component_analysis"][key] = value
            else:
                # Catch any uncategorized metrics
                enhanced["basic_properties"][key] = value

        return enhanced

    def _save_json_report(self, metrics: Dict[str, Any]):
        """Save structured JSON report for programmatic access."""
        json_path = self.output_dir / f"{self.run_id}_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        log_substep(f"JSON metrics saved to {json_path}")

    def _save_human_readable_report(self, metrics: Dict[str, Any], citation_metrics: Dict[str, Any] = None):
        """Save formatted human-readable report with optional citation analysis."""
        report_path = self.output_dir / f"{self.run_id}_report.md"

        with open(report_path, 'w') as f:
            f.write(f"# Graph Analysis Report\n\n")
            f.write(f"- **Run ID:** {metrics['metadata']['run_id']}\n")
            f.write(f"- **Timestamp:** {metrics['metadata']['timestamp']}\n")
            f.write(
                f"- **Graph Source:** {metrics['metadata']['graph_source']}\n")
            f.write(
                f"- **Analysis Version:** {metrics['metadata']['analysis_version']}\n\n")

            # Add scalability warnings if present
            if metrics['metadata'].get('scalability_notes'):
                f.write("## ⚠️ Scalability Notes\n\n")
                for note_key, note_value in metrics['metadata']['scalability_notes'].items():
                    f.write(
                        f"- **{note_key.replace('_', ' ').title()}:** {note_value}\n")
                f.write("\n")

            f.write("## Executive Summary\n\n")
            f.write(self._generate_executive_summary(metrics))
            f.write("\n\n")

            f.write("## Detailed Metrics\n\n")

            # Define display order and descriptions for categories
            category_info = {
                "basic_properties": {
                    "title": "Basic Graph Properties",
                    "description": "Fundamental characteristics of the graph structure",
                },
                "connectivity": {
                    "title": "Graph Connectivity",
                    "description": "How well connected the graph components are",
                },
                "node_isolation": {
                    "title": "Node Isolation Analysis",
                    "description": "Statistics about isolated and removed nodes",
                },
                "degree_statistics": {
                    "title": "Degree Distribution",
                    "description": "Statistical distribution of node connections",
                },
                "clustering_metrics": {
                    "title": "Clustering Analysis",
                    "description": "How tightly connected local neighborhoods are (⚠️ expensive for large graphs)",
                },
                "distance_metrics": {
                    "title": "Distance & Path Analysis",
                    "description": "Shortest paths and network diameter (⚠️ very expensive for large graphs)",
                },
                "centrality_measures": {
                    "title": "Node Centrality",
                    "description": "Measures of node importance and influence (⚠️ expensive for large graphs)",
                },
                "structural_features": {
                    "title": "Structural Features",
                    "description": "Advanced structural characteristics and patterns",
                },
                "directed_graph_metrics": {
                    "title": "Directed Graph Metrics",
                    "description": "Metrics specific to directed graphs (in/out degree, reciprocity, etc.)",
                },
                "component_analysis": {
                    "title": "Component Analysis",
                    "description": "Detailed analysis of connected components",
                }
            }

            for category, cat_metrics in metrics.items():
                if category != "metadata" and isinstance(cat_metrics, dict) and cat_metrics:
                    info = category_info.get(category, {
                        "title": category.replace('_', ' ').title(),
                        "description": "",
                    })

                    f.write(f"### {info['title']}\n\n")
                    if info['description']:
                        f.write(f"*{info['description']}*\n\n")

                    for key, value in cat_metrics.items():
                        formatted_key = key.replace('_', ' ').title()
                        if isinstance(value, float):
                            if 'centrality' in key or 'efficiency' in key or 'clustering' in key or 'assortativity' in key:
                                f.write(
                                    f"- **{formatted_key}:** {value:.6f}\n")
                            else:
                                f.write(
                                    f"- **{formatted_key}:** {value:.4f}\n")
                        elif isinstance(value, list) and len(value) <= 10:
                            # Only show small lists
                            f.write(f"- **{formatted_key}:** {value}\n")
                        elif isinstance(value, list):
                            # Summarize large lists
                            f.write(
                                f"- **{formatted_key}:** {len(value)} items (min: {min(value)}, max: {max(value)}, mean: {sum(value)/len(value):.2f})\n")
                        else:
                            f.write(f"- **{formatted_key}:** {value}\n")
                    f.write("\n")

            # Add citation network analysis if provided
            if citation_metrics:
                f.write("## 📚 Citation Network Analysis\n\n")
                f.write("*Specialized analysis for citation/reference networks*\n\n")

                # Citation statistics
                f.write("### Citation Distribution\n\n")
                f.write("**Citations Made (Out-degree):**\n")
                f.write(
                    f"- Mean: {citation_metrics.get('citations_made_mean', 0):.2f}\n")
                f.write(
                    f"- Median: {citation_metrics.get('citations_made_median', 0):.2f}\n")
                f.write(
                    f"- Max: {citation_metrics.get('citations_made_max', 0)}\n")
                f.write(
                    f"- Standard Deviation: {citation_metrics.get('citations_made_std', 0):.2f}\n\n")

                f.write("**Citations Received (In-degree):**\n")
                f.write(
                    f"- Mean: {citation_metrics.get('citations_received_mean', 0):.2f}\n")
                f.write(
                    f"- Median: {citation_metrics.get('citations_received_median', 0):.2f}\n")
                f.write(
                    f"- Max: {citation_metrics.get('citations_received_max', 0)}\n")
                f.write(
                    f"- Standard Deviation: {citation_metrics.get('citations_received_std', 0):.2f}\n")
                f.write(
                    f"- **Network H-Index:** {citation_metrics.get('h_index', 0)}\n\n")

                # Node categories
                f.write("### Node Categories\n\n")
                f.write(
                    f"- **Isolated Nodes:** {citation_metrics.get('num_isolated_nodes', 0)} (no citations in either direction)\n")
                f.write(
                    f"- **Terminal Nodes:** {citation_metrics.get('num_terminal_nodes', 0)} (don't cite others)\n")
                f.write(
                    f"- **Source Nodes:** {citation_metrics.get('num_source_nodes', 0)} (not cited by others)\n\n")

                # Top nodes by various metrics
                top_cited = citation_metrics.get('top_cited_nodes', [])[:10]
                if top_cited:
                    f.write("### Top 10 Most Cited Nodes\n\n")
                    f.write("| Rank | Node ID | Title | Citations Received |\n")
                    f.write(
                        "|------|---------|-------------------------------|--------------------|\n")
                    for i, (node, title, degree) in enumerate(top_cited, 1):
                        f.write(f"| {i} | {node} | {title} | {degree} |\n")
                    f.write("\n")

                top_citing = citation_metrics.get('top_citing_nodes', [])[:10]
                if top_citing:
                    f.write("### Top 10 Most Citing Nodes\n\n")
                    f.write("| Rank | Node ID | Title | Citations Made |\n")
                    f.write(
                        "|------|---------|-------------------------------|--------------------|\n")
                    for i, (node, title, degree) in enumerate(top_citing, 1):
                        f.write(f"| {i} | {node} | {title} | {degree} |\n")
                    f.write("\n")

                # PageRank influence
                pagerank_top = citation_metrics.get(
                    'pagerank_top_nodes', [])[:10]
                if pagerank_top:
                    f.write("### Top 10 Most Influential Nodes (PageRank)\n\n")
                    f.write("| Rank | Node ID | Title | Influence Score |\n")
                    f.write(
                        "|------|---------|-------------------------------|--------------------|\n")
                    for i, (node, title, score) in enumerate(pagerank_top, 1):
                        f.write(f"| {i} | {node} | {title} | {score:.6f} |\n")
                    f.write("\n")

        log_substep(f"Enhanced Markdown report saved to {report_path}")

    def _generate_executive_summary(self, metrics: Dict[str, Any]) -> str:
        """Generate an executive summary for stakeholders."""
        basic = metrics.get("basic_properties", {})
        connectivity = metrics.get("connectivity", {})
        node_isolation = metrics.get("node_isolation", {})
        degree = metrics.get("degree_statistics", {})
        advanced = metrics.get("advanced_metrics", {})

        nodes = basic.get("num_nodes", 0)
        edges = basic.get("num_edges", 0)
        avg_degree = degree.get("average_degree", 0)
        density = advanced.get("density", 0)
        clustering = advanced.get("average_clustering", 0)
        nodes_after_removal = node_isolation.get("num_nodes_after_removal", 0)
        num_isolated_nodes_removed = node_isolation.get(
            "num_isolated_nodes_removed", 0)

        # Interpret metrics for business audience
        size_description = "small" if nodes < 100 else "medium" if nodes < 1000 else "large"
        density_description = "sparse" if density < 0.1 else "moderate" if density < 0.5 else "dense"
        clustering_description = "low" if clustering < 0.3 else "moderate" if clustering < 0.7 else "high"

        summary = f"""This analysis examined a **{size_description} {basic.get('is_directed', False) and 'directed' or 'undirected'} graph** with:

 - {nodes:,} nodes (entities) and {edges:,} connections
 - {nodes_after_removal:,} nodes remaining after isolation removal ({num_isolated_nodes_removed} isolated nodes removed)
 - {density_description.title()} connectivity (density: {density:.1%})
 - {clustering_description.title()} clustering ({clustering:.1%} - how tightly grouped entities are)
 - Average connections per entity: {avg_degree:.1f}

**Key Insights:**
"""

        # Add specific insights based on metrics
        if connectivity.get("is_connected_undirected", False):
            summary += " - The network is fully connected (all entities are reachable)\n"
        else:
            components = connectivity.get("num_connected_components", 0)
            largest = connectivity.get("largest_connected_component_size", 0)
            summary += f" - The network has {components} disconnected clusters, with the largest containing {largest} entities\n"

        if density < 0.05:
            summary += " - Low connectivity suggests potential for network growth\n"
        elif density > 0.5:
            summary += " - High connectivity indicates a tightly integrated network\n"

        if clustering > 0.5:
            summary += " - High clustering suggests strong community structure\n"

        return summary

    def _log_executive_summary(self, metrics: Dict[str, Any]):
        """Log executive summary to main pipeline log."""
        self.logger.info("="*67)
        self.logger.info("Graph Analysis Summary".center(67))
        self.logger.info("-"*67)

        summary = self._generate_executive_summary(metrics)
        for line in summary.split('\n'):
            if line.strip():
                self.logger.info(line)


def analyze_graph_enhanced(G: nx.Graph, output_dir: Path, run_id: str = None,
                           graph_info: Dict[str, Any] = None,
                           include_citation_analysis: bool = False,
                           citation_top_n: int = 10,
                           **kwargs) -> Dict[str, Any]:
    """
    Enhanced graph analysis with improved reporting and optional citation analysis.

    Args:
        G: NetworkX graph to analyze
        output_dir: Directory for output files
        run_id: Unique identifier for this run
        graph_info: Additional metadata about the graph
        include_citation_analysis: Whether to run citation network analysis (only for directed graphs)
        citation_top_n: Number of top nodes to include in citation analysis
        **kwargs: Additional arguments for analyze_graph

    Returns:
        Dictionary of computed metrics
    """
    from community_detection.evaluation.graph_metrics import analyze_graph, analyze_citation_network

    # Run original analysis (suppress print_results)
    metrics, G = analyze_graph(G, print_results=False, **kwargs)

    # Run citation analysis if requested and graph is directed
    citation_metrics = None
    if include_citation_analysis and G.is_directed():
        try:
            citation_metrics = analyze_citation_network(
                G, top_n=citation_top_n, print_results=False)
            log_substep("Citation network analysis completed")
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Citation analysis failed: {e}")
    elif include_citation_analysis and not G.is_directed():
        logging.getLogger(__name__).warning(
            "Citation analysis skipped - requires directed graph")

    # Generate enhanced reports
    reporter = GraphMetricsReporter(output_dir, run_id)
    enhanced_metrics = reporter.generate_reports(
        metrics, graph_info, citation_metrics)

    return enhanced_metrics, G
