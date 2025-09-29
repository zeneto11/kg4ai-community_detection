# community_detection/evaluation/reporter.py

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from community_detection.utils.logger import log_substep
from community_detection.utils.metrics_status import MetricStatus
from community_detection.utils.time import format_time


class EnhancedGraphMetricsReporter:
    """Enhanced reporter with improved formatting and visualizations."""

    def __init__(self, output_dir: Path, run_id: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = logging.getLogger(__name__)

        # Create visualizations directory
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)

    def generate_reports(self, metrics: Dict[str, Any], graph: nx.Graph,
                         graph_info: Dict[str, Any] = None,
                         citation_metrics: Dict[str, Any] = None):
        """Generate comprehensive reports with visualizations."""

        # Add metadata
        enhanced_metrics = self._add_metadata(metrics, graph_info)

        # Generate visualizations
        viz_paths = self._generate_visualizations(graph, metrics)
        enhanced_metrics['_visualizations'] = viz_paths

        # Generate different report formats
        self._save_json_report(enhanced_metrics)
        self._save_enhanced_markdown_report(
            enhanced_metrics, citation_metrics, viz_paths)
        self._log_executive_summary(enhanced_metrics)

        return enhanced_metrics

    def _format_metric_value(self, key: str, value: Any) -> str:
        """Standardize decimal places based on metric type."""
        if not isinstance(value, (int, float)):
            return str(value)

        if isinstance(value, int):
            return f"{value:,}"

        # Decimal formatting rules
        if any(term in key.lower() for term in ['percentage', 'rate', 'fraction']):
            return f"{value:.2%}" if value <= 1 else f"{value:.2f}%"
        elif any(term in key.lower() for term in ['centrality', 'coefficient', 'assortativity', 'clustering']):
            return f"{value:.4f}"
        elif any(term in key.lower() for term in ['efficiency', 'reciprocity', 'density']):
            return f"{value:.4f}"
        elif 'degree' in key.lower() and 'average' not in key.lower():
            return f"{value:.1f}"
        else:
            return f"{value:.2f}"

    def _get_status_indicator(self, metric_name: str, metrics: Dict[str, Any]) -> str:
        """Get status indicator for metric."""
        metadata = metrics.get('_computation_metadata', {})
        statuses = metadata.get('metric_statuses', {})

        # Check various possible metric names
        possible_names = [
            metric_name,
            metric_name.replace('_', ' '),
            next((k for k in statuses.keys() if k in metric_name), None)
        ]

        for name in possible_names:
            if name and name in statuses:
                return statuses[name]

        return MetricStatus.COMPUTED

    def _generate_visualizations(self, graph: nx.Graph, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Generate enhanced visualization plots and return their paths."""
        viz_paths = {}

        try:
            # Set up plotting style
            plt.style.use(
                'seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

            # 1. Enhanced Degree Distribution Analysis
            degrees = [d for _, d in graph.degree()]
            if degrees:
                degree_counts = Counter(degrees)

                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
                    2, 2, figsize=(15, 12))

                # Histogram with better binning - focus on the meaningful range
                max_degree = max(degrees)
                min_degree = min(degrees)

                # Use intelligent binning
                if max_degree > 100:
                    # For high-degree networks, use log-spaced bins for linear plot
                    bins = np.logspace(0, np.log10(max_degree + 1), 50)
                    ax1.hist(degrees, bins=bins, alpha=0.7,
                             color='lightcoral', edgecolor='black')
                    ax1.set_xscale('log')
                else:
                    # Normal binning for smaller degree ranges
                    bins = min(50, max_degree - min_degree + 1)
                    ax1.hist(degrees, bins=bins, alpha=0.7,
                             color='lightcoral', edgecolor='black')

                ax1.set_xlabel('Degree')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Degree Distribution')
                ax1.grid(True, alpha=0.3)

                # Log-log plot with better handling
                degrees_sorted = sorted(degree_counts.keys())
                counts = [degree_counts[d] for d in degrees_sorted]
                ax2.loglog(degrees_sorted, counts, 'o-', alpha=0.7,
                           color='darkblue', markersize=3)
                ax2.set_xlabel('Degree (log)')
                ax2.set_ylabel('Frequency (log)')
                ax2.set_title('Degree Distribution (Log-Log Scale)')
                ax2.grid(True, alpha=0.3)

                # Cumulative distribution - fix the calculation
                unique_degrees = sorted(set(degrees))
                cumulative_probs = []
                total_nodes = len(degrees)

                for deg in unique_degrees:
                    prob = sum(1 for d in degrees if d >= deg) / total_nodes
                    cumulative_probs.append(prob)

                ax3.loglog(unique_degrees, cumulative_probs, 'o-', alpha=0.7,
                           color='green', markersize=3)
                ax3.set_xlabel('Degree (log)')
                ax3.set_ylabel('P(X ≥ x) - Cumulative Probability (log)')
                ax3.set_title('Cumulative Degree Distribution')
                ax3.grid(True, alpha=0.3)

                # Box plot with outlier handling
                # Remove extreme outliers for better visualization
                q75, q25 = np.percentile(degrees, [75, 25])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr

                # Show both full and filtered box plots
                ax4.boxplot([degrees, [d for d in degrees if lower_bound <= d <= upper_bound]],
                            labels=['Full', 'Filtered'])
                ax4.set_ylabel('Degree')
                ax4.set_title('Degree Distribution Box Plot')
                ax4.grid(True, alpha=0.3)

                plt.tight_layout()
                degree_path = self.viz_dir / \
                    f"{self.run_id}_degree_analysis.png"
                plt.savefig(degree_path, dpi=300, bbox_inches='tight')
                plt.close()
                viz_paths['degree_distribution'] = str(degree_path.name)

            # 2. Enhanced Component Size Distribution - Fixed
            if graph.is_directed():
                components = list(nx.weakly_connected_components(graph))
            else:
                components = list(nx.connected_components(graph))

            if len(components) > 1:
                comp_sizes = [len(c) for c in components]
                size_counts = Counter(comp_sizes)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Better histogram binning for component sizes
                max_size = max(comp_sizes)
                min_size = min(comp_sizes)

                if max_size > 1000:
                    # Use log-spaced bins for very large components
                    bins = np.logspace(0, np.log10(max_size + 1), 30)
                    ax1.hist(comp_sizes, bins=bins, alpha=0.7,
                             color='orange', edgecolor='black')
                    ax1.set_xscale('log')
                else:
                    # Regular binning
                    bins = min(30, max_size - min_size + 1)
                    ax1.hist(comp_sizes, bins=bins, alpha=0.7,
                             color='orange', edgecolor='black')

                ax1.set_xlabel('Component Size')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Component Size Distribution')
                ax1.grid(True, alpha=0.3)

                # Second plot: Smart visualization based on data
                sizes_sorted = sorted(size_counts.keys(), reverse=True)
                counts = [size_counts[s] for s in sizes_sorted]

                if len(sizes_sorted) > 20:
                    # Log-log for many different sizes
                    ax2.loglog(sizes_sorted, counts, 'o-',
                               alpha=0.7, color='purple', markersize=4)
                    ax2.set_xlabel('Component Size (log)')
                    ax2.set_ylabel('Number of Components (log)')
                    ax2.set_title('Component Size Distribution (Log-Log)')
                else:
                    # Bar plot for fewer unique sizes, but limit to top 15
                    top_15_sizes = sizes_sorted[:15]
                    top_15_counts = counts[:15]

                    bars = ax2.bar(range(len(top_15_sizes)), top_15_counts,
                                   alpha=0.7, color='purple')
                    ax2.set_xlabel('Component Size')
                    ax2.set_ylabel('Number of Components')
                    ax2.set_title(f'Top {len(top_15_sizes)} Component Sizes')
                    ax2.set_xticks(range(len(top_15_sizes)))
                    ax2.set_xticklabels(
                        [f'{s:,}' for s in top_15_sizes], rotation=45)

                    # Add value labels on bars if not too many
                    if len(top_15_sizes) <= 10:
                        for bar, count in zip(bars, top_15_counts):
                            height = bar.get_height()
                            ax2.text(bar.get_x() + bar.get_width()/2., height,
                                     f'{count}', ha='center', va='bottom')

                ax2.grid(True, alpha=0.3)
                plt.tight_layout()

                comp_path = self.viz_dir / \
                    f"{self.run_id}_component_analysis.png"
                plt.savefig(comp_path, dpi=300, bbox_inches='tight')
                plt.close()
                viz_paths['component_distribution'] = str(comp_path.name)

            # 3. Clustering coefficient visualization (keep existing but add error handling)
            if graph.number_of_nodes() <= 5000:
                try:
                    clustering_coeffs = nx.clustering(graph)
                    clustering_values = list(clustering_coeffs.values())

                    # Only if varied
                    if clustering_values and len(set(clustering_values)) > 1:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                        # Histogram with appropriate binning
                        non_zero_clustering = [
                            c for c in clustering_values if c > 0]
                        if non_zero_clustering:
                            ax1.hist(clustering_values, bins=50, alpha=0.7,
                                     color='skyblue', edgecolor='black')
                            ax1.set_xlabel('Clustering Coefficient')
                            ax1.set_ylabel('Frequency')
                            ax1.set_title(
                                'Clustering Coefficient Distribution')
                            ax1.grid(True, alpha=0.3)

                            mean_clustering = np.mean(clustering_values)
                            ax1.axvline(mean_clustering, color='red', linestyle='--',
                                        label=f'Mean: {mean_clustering:.3f}')
                            ax1.legend()

                            # Box plot
                            ax2.boxplot(clustering_values)
                            ax2.set_ylabel('Clustering Coefficient')
                            ax2.set_title('Clustering Coefficient Box Plot')
                            ax2.grid(True, alpha=0.3)

                            plt.tight_layout()
                            clustering_path = self.viz_dir / \
                                f"{self.run_id}_clustering_analysis.png"
                            plt.savefig(clustering_path, dpi=300,
                                        bbox_inches='tight')
                            plt.close()
                            viz_paths['clustering_distribution'] = str(
                                clustering_path.name)

                except Exception as e:
                    self.logger.warning(
                        f"Failed to generate clustering distribution: {e}")

        except Exception as e:
            self.logger.warning(f"Visualization generation failed: {e}")

        return viz_paths

    def _add_metadata(self, metrics: Dict[str, Any], graph_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add metadata with improved categorization."""
        enhanced = {
            "metadata": {
                "run_id": self.run_id,
                "timestamp": datetime.now().isoformat(),
                "analysis_version": "3.0",
                "graph_source": graph_info.get("source", "unknown") if graph_info else "unknown",
            },
            "graph_structure": {},
            "connectivity": {},
            "node_isolation": {},
            "degree_statistics": {},
            "directed_properties": {},
            "clustering_metrics": {},
            "distance_metrics": {},
            "centrality_measures": {},
            "structural_features": {},
            "component_analysis": {}
        }

        # Improved categorization (fixed grouping issues)
        categorization = {
            "graph_structure": [
                "is_directed", "is_weighted", "is_multigraph", "num_nodes", "num_edges"
            ],
            "connectivity": [
                "density", "is_connected_undirected", "is_strongly_connected",
                "is_weakly_connected", "reciprocity"
            ],
            "node_isolation": [
                "num_nodes_zero_total_degree", "num_isolated_nodes_removed",
                "min_component_size_threshold", "num_small_components_removed", "num_nodes_in_small_components_removed",
                "total_nodes_removed", "num_nodes_after_removal", "num_edges_after_removal", "removal_percentage"
            ],
            "degree_statistics": [
                "average_degree", "max_degree", "min_degree", "median_degree", "std_dev_degree"
            ],
            "directed_properties": [
                "num_nodes_zero_in_degree", "num_nodes_zero_out_degree",
                "max_in_degree", "min_in_degree", "mean_in_degree", "median_in_degree",
                "max_out_degree", "min_out_degree", "mean_out_degree", "median_out_degree",
                "average_pagerank", "max_pagerank"
            ],
            "clustering_metrics": [
                "transitivity", "average_clustering", "filtered_average_clustering"
            ],
            "distance_metrics": [
                "average_shortest_path_length", "diameter", "radius", "global_efficiency",
                "average_shortest_path_length_method", "diameter_method", "radius_method", "global_efficiency_method",
                "largest_cc_average_shortest_path_length", "largest_cc_diameter", "largest_cc_radius", "largest_cc_global_efficiency",
                "largest_cc_average_shortest_path_length_method", "largest_cc_diameter_method", "largest_cc_radius_method", "largest_cc_global_efficiency_method",
                "largest_scc_average_shortest_path_length", "largest_scc_diameter", "largest_scc_radius", "largest_scc_global_efficiency",
                "largest_scc_average_shortest_path_length_method", "largest_scc_diameter_method", "largest_scc_radius_method", "largest_scc_global_efficiency_method"
            ],
            "centrality_measures": [
                "average_degree_centrality", "average_closeness_centrality",
                "average_betweenness_centrality", "average_eigenvector_centrality"
            ],
            "structural_features": [
                "self_loops", "num_bridges", "degree_assortativity",
                "max_core_number", "mean_core_number", "rich_club_coefficient"
            ],
            "component_analysis": [
                "num_connected_components", "num_weakly_connected_components",
                "num_strongly_connected_components", "largest_connected_component_size",
                "largest_weakly_connected_component_size", "largest_strongly_connected_component_size",
                "cc_size_mean", "cc_size_median", "weakly_cc_size_mean", "weakly_cc_size_median",
                "strongly_cc_size_mean", "strongly_cc_size_median",
                "connected_component_sizes", "weakly_connected_component_sizes",
                "strongly_connected_component_sizes"
            ]
        }

        # Categorize metrics (avoiding duplication)
        for key, value in metrics.items():
            if key.startswith('_'):  # Skip metadata
                continue

            categorized = False
            for category, keys in categorization.items():
                if key in keys:
                    enhanced[category][key] = value
                    categorized = True
                    break

            # If not categorized, put in appropriate default category
            if not categorized:
                if 'component' in key and 'size' in key:
                    enhanced["component_analysis"][key] = value
                else:
                    enhanced["graph_structure"][key] = value

        # Add computation metadata if present
        if '_computation_metadata' in metrics:
            enhanced['metadata']['computation_metadata'] = metrics['_computation_metadata']

        return enhanced

    def _save_json_report(self, metrics: Dict[str, Any]):
        """Save structured JSON report."""
        json_path = self.output_dir / f"{self.run_id}_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        log_substep(f"JSON metrics saved to {json_path}")

    def _save_enhanced_markdown_report(self, metrics: Dict[str, Any],
                                       citation_metrics: Dict[str, Any] = None,
                                       viz_paths: Dict[str, str] = None):
        """Save enhanced markdown report with visual improvements."""
        report_path = self.output_dir / f"{self.run_id}_report.md"

        with open(report_path, 'w') as f:

            # Header
            f.write("# 📊 Graph Analysis Report\n\n")
            f.write("---\n\n")

            # Metadata table
            f.write("## 📋 Analysis Metadata\n\n")
            f.write("| Property | Value |\n")
            f.write("|----------|-------|\n")
            f.write(f"| **Run ID** | `{metrics['metadata']['run_id']}` |\n")
            f.write(
                f"| **Timestamp** | {metrics['metadata']['timestamp']} |\n")
            f.write(
                f"| **Graph Source** | {metrics['metadata']['graph_source']} |\n")
            f.write(
                f"| **Analysis Version** | {metrics['metadata']['analysis_version']} |\n")
            comp_metadata = metrics.get('metadata', {}).get(
                'computation_metadata', {})
            f.write(
                f"| **Total Time** | {format_time(comp_metadata['total_time'])} |\n\n")

            # Executive Summary
            f.write("## 🎯 Executive Summary\n\n")
            f.write(self._generate_executive_summary(metrics))
            f.write("\n\n---\n\n")

            # Visualizations section
            if viz_paths:
                f.write("## 📈 Enhanced Data Visualizations\n\n")

                if 'degree_distribution' in viz_paths:
                    f.write(f"### Comprehensive Degree Analysis\n")
                    f.write(
                        f"![Degree Analysis](visualizations/{viz_paths['degree_distribution']})\n")
                    f.write(
                        "*Four-panel analysis showing linear distribution, log-log scaling for power-law detection, ")
                    f.write(
                        "cumulative distribution, and statistical summary via box plot.*\n\n")

                if 'clustering_distribution' in viz_paths:
                    f.write(f"### Clustering Coefficient Analysis\n")
                    f.write(
                        f"![Clustering Analysis](visualizations/{viz_paths['clustering_distribution']})\n")
                    f.write(
                        "*Distribution and box plot analysis of local clustering coefficients, ")
                    f.write("showing community structure tendencies.*\n\n")

                if 'component_distribution' in viz_paths:
                    f.write(f"### Connected Component Analysis\n")
                    f.write(
                        f"![Component Analysis](visualizations/{viz_paths['component_distribution']})\n")
                    f.write(
                        "*Analysis of network fragmentation showing component size distribution ")
                    f.write("and structural connectivity patterns.*\n\n")

                f.write("---\n\n")

            # Detailed Metrics
            f.write("## 📊 Detailed Metrics\n\n")

            # Define display order and descriptions for categories
            category_info = {
                "graph_structure": {
                    "title": "🏗️ Graph Structure",
                    "description": "Fundamental properties of the graph topology",
                    "icon": "🏗️"
                },
                "connectivity": {
                    "title": "🔗 Connectivity Analysis",
                    "description": "Network connectivity and density measurements",
                    "icon": "🔗"
                },
                "node_isolation": {
                    "title": "🏝️ Node Isolation Analysis",
                    "description": "Statistics about disconnected and removed nodes",
                    "icon": "🏝️"
                },
                "degree_statistics": {
                    "title": "📈 Degree Distribution",
                    "description": "Statistical analysis of node connection patterns",
                    "icon": "📈"
                },
                "directed_properties": {
                    "title": "➡️ Directed Graph Properties",
                    "description": "Metrics specific to directed networks (in/out degrees, PageRank)",
                    "icon": "➡️"
                },
                "clustering_metrics": {
                    "title": "🎯 Clustering Analysis",
                    "description": "Local neighborhood connectivity and transitivity",
                    "icon": "🎯"
                },
                "distance_metrics": {
                    "title": "📏 Distance & Path Analysis",
                    "description": "Shortest paths and network diameter measurements",
                    "icon": "📏"
                },
                "centrality_measures": {
                    "title": "⭐ Node Centrality",
                    "description": "Measures of node importance and influence",
                    "icon": "⭐"
                },
                "structural_features": {
                    "title": "🧬 Structural Features",
                    "description": "Advanced topological characteristics",
                    "icon": "🧬"
                },
                "component_analysis": {
                    "title": "🧩 Component Analysis",
                    "description": "Connected component structure and distribution",
                    "icon": "🧩"
                }
            }

            for category, cat_metrics in metrics.items():
                if (category != "metadata" and isinstance(cat_metrics, dict) and
                        cat_metrics and not category.startswith('_')):

                    info = category_info.get(category, {
                        "title": category.replace('_', ' ').title(),
                        "description": "",
                        "icon": "📊"
                    })

                    f.write(f"### {info['title']}\n\n")
                    if info['description']:
                        f.write(f"*{info['description']}*\n\n")

                    # Create metrics table
                    if category == "distance_metrics":
                        f.write("| Metric | Value | Method | Status |\n")
                        f.write("|--------|-------|--------|--------|\n")

                        for key, value in cat_metrics.items():
                            if key.endswith("_method"):
                                continue  # skip method here, handled together

                            formatted_key = key.replace('_', ' ').title()
                            formatted_value = self._format_metric_value(
                                key, value)
                            status = self._get_status_indicator(key, metrics)

                            # look for a method field
                            method_key = f"{key}_method"
                            method_value = cat_metrics.get(method_key)

                            f.write(
                                f"| **{formatted_key}** | {formatted_value} | {method_value} | {status} |\n"
                            )
                    else:
                        f.write("| Metric | Value | Status |\n")
                        f.write("|--------|-------|--------|\n")

                        for key, value in cat_metrics.items():
                            formatted_key = key.replace('_', ' ').title()
                            formatted_value = self._format_metric_value(
                                key, value)
                            status = self._get_status_indicator(key, metrics)

                            # Handle large lists differently
                            if isinstance(value, list) and len(value) > 10:
                                formatted_value = f"{len(value)} items (μ={np.mean(value):.2f}, σ={np.std(value):.2f})"
                            elif isinstance(value, list) and len(value) <= 10:
                                formatted_value = str(value)

                            f.write(
                                f"| **{formatted_key}** | {formatted_value} | {status} |\n"
                            )

                    f.write("\n---\n\n")

            # Citation Analysis (if provided)
            if citation_metrics:
                f.write("## 📚 Citation Network Analysis\n\n")
                f.write("*Specialized analysis for citation/reference networks*\n\n")

                # Citation statistics table
                f.write("### Citation Statistics\n\n")
                f.write(
                    "| Metric | Citations Made (Out-degree) | Citations Received (In-degree) |\n")
                f.write(
                    "|--------|------------------------------|----------------------------------|\n")
                f.write(
                    f"| **Mean** | {citation_metrics.get('citations_made_mean', 0):.2f} | {citation_metrics.get('citations_received_mean', 0):.2f} |\n")
                f.write(
                    f"| **Median** | {citation_metrics.get('citations_made_median', 0):.2f} | {citation_metrics.get('citations_received_median', 0):.2f} |\n")
                f.write(
                    f"| **Maximum** | {citation_metrics.get('citations_made_max', 0)} | {citation_metrics.get('citations_received_max', 0)} |\n")
                f.write(
                    f"| **Std Dev** | {citation_metrics.get('citations_made_std', 0):.2f} | {citation_metrics.get('citations_received_std', 0):.2f} |\n\n")

                f.write(
                    f"**Network H-Index:** {citation_metrics.get('h_index', 0)}\n\n")

                # Node categories
                f.write("### Node Categories\n\n")
                f.write(
                    f"- 🏝️ **Isolated Nodes:** {citation_metrics.get('num_isolated_nodes', 0)} (no citations)\n")
                f.write(
                    f"- 🔚 **Terminal Nodes:** {citation_metrics.get('num_terminal_nodes', 0)} (don't cite others)\n")
                f.write(
                    f"- 🌱 **Source Nodes:** {citation_metrics.get('num_source_nodes', 0)} (not cited by others)\n\n")

                # Top nodes tables
                self._write_citation_rankings(f, citation_metrics)

        log_substep(f"Enhanced Markdown report saved to {report_path}")

    def _write_citation_rankings(self, f, citation_metrics: Dict[str, Any]):
        """Write citation ranking tables."""

        # Top cited nodes
        top_cited = citation_metrics.get('top_cited_nodes', [])[:10]
        if top_cited:
            f.write("### 🏆 Top 10 Most Cited Papers\n\n")
            f.write("| Rank | Node ID | Title | Citations |\n")
            f.write("|------|---------|-------|----------|\n")
            for i, (node, title, degree) in enumerate(top_cited, 1):
                title_short = (
                    title[:40] + "...") if len(title) > 40 else title
                f.write(f"| {i} | `{node}` | {title_short} | **{degree}** |\n")
            f.write("\n")

        # Top citing nodes
        top_citing = citation_metrics.get('top_citing_nodes', [])[:10]
        if top_citing:
            f.write("### 📝 Top 10 Most Citing Papers\n\n")
            f.write("| Rank | Node ID | Title | Citations Made |\n")
            f.write("|------|---------|-------|----------------|\n")
            for i, (node, title, degree) in enumerate(top_citing, 1):
                title_short = (
                    title[:40] + "...") if len(title) > 40 else title
                f.write(f"| {i} | `{node}` | {title_short} | **{degree}** |\n")
            f.write("\n")

        # PageRank influence
        pagerank_top = citation_metrics.get('pagerank_top_nodes', [])[:10]
        if pagerank_top:
            f.write("### 🌟 Top 10 Most Influential Papers (PageRank)\n\n")
            f.write("| Rank | Node ID | Title | Influence Score |\n")
            f.write("|------|---------|-------|----------------|\n")
            for i, (node, title, score) in enumerate(pagerank_top, 1):
                title_short = (
                    title[:40] + "...") if len(title) > 40 else title
                f.write(f"| {i} | `{node}` | {title_short} | {score:.6f} |\n")
            f.write("\n")

    def _generate_executive_summary(self, metrics: Dict[str, Any]) -> str:
        """Generate executive summary with sparklines and progress indicators."""
        structure = metrics.get("graph_structure", {})
        connectivity = metrics.get("connectivity", {})
        isolation = metrics.get("node_isolation", {})
        degree = metrics.get("degree_statistics", {})

        nodes = structure.get("num_nodes", 0)
        edges = structure.get("num_edges", 0)
        avg_degree = degree.get("average_degree", 0)
        density = connectivity.get("density", 0)

        # Size classification
        size_desc = ("small" if nodes < 100 else
                     "medium" if nodes < 1000 else
                     "large" if nodes < 10000 else "very large")

        # Density classification
        density_desc = ("sparse" if density < 0.1 else
                        "moderate" if density < 0.5 else "dense")

        summary = f"""**Network Overview:**
- **Scale:** {size_desc.title()} {structure.get('is_directed', False) and 'directed' or 'undirected'} network
- **Size:** {nodes:,} nodes, {edges:,} edges
- **Connectivity:** {density_desc} (density: {density:.1%})
- **Average Connections:** {avg_degree:.1f} per node

**Key Characteristics:**"""

        # Connectivity insights
        if connectivity.get("is_connected_undirected", False):
            summary += "\n- ✅ **Fully Connected:** All nodes are reachable"
        else:
            components = metrics.get("component_analysis", {}).get(
                "num_connected_components", 0)
            largest = metrics.get("component_analysis", {}).get(
                "largest_connected_component_size", 0)
            if components > 1:
                summary += f"\n- 🧩 **Fragmented:** {components} separate components (largest: {largest} nodes)"

        # Density insights
        if density < 0.01:
            summary += "\n- 📈 **Growth Potential:** Very sparse network with room for expansion"
        elif density > 0.5:
            summary += "\n- 🔗 **Highly Integrated:** Dense connectivity indicates tight coupling"

        # Clustering insights
        clustering = metrics.get("clustering_metrics", {}).get(
            "average_clustering", 0)
        if clustering > 0.3:
            summary += f"\n- 🎯 **Strong Communities:** High clustering ({clustering:.1%}) suggests community structure"

        return summary

    def _log_executive_summary(self, metrics: Dict[str, Any]):
        """Log executive summary with progress indicators."""
        self.logger.info("="*80)
        self.logger.info("🚀 GRAPH ANALYSIS COMPLETE".center(80))
        self.logger.info("-"*80)

        summary = self._generate_executive_summary(metrics)
        for line in summary.split('\n'):
            if line.strip():
                self.logger.info(line)

        self.logger.info("-"*80)

        # Show computation summary
        comp_metadata = metrics.get('metadata', {}).get(
            'computation_metadata', {})
        if comp_metadata:
            self.logger.info(
                f"📊 Computed metrics in {comp_metadata['total_time']:.2f} seconds")

        warnings = comp_metadata.get('warnings', [])
        if warnings:
            self.logger.info(f"⚠️  {len(warnings)} warnings generated")

        self.logger.info("="*80)
