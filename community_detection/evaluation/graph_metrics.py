# community_detection/evaluation/graph_metrics.py

import logging
import random
import time
from typing import Any, Dict, Tuple

import networkx as nx
import numpy as np
from networkx.algorithms import approximation as approx

from community_detection.utils.metrics_status import MetricStatus
from community_detection.utils.time import format_time


class MetricTimer:
    """Context manager for timing metric computations."""

    def __init__(self, metric_name: str, logger: logging.Logger):
        self.metric_name = metric_name
        self.logger = logger
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Computing {self.metric_name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        if exc_type is None:
            self.logger.info(
                f"✅ {self.metric_name} completed in {format_time(duration)}")
        else:
            self.logger.warning(
                f"❌ {self.metric_name} failed after {format_time(duration)}: {exc_val}")

    @property
    def duration(self):
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return None


class FastDistanceAlgorithms:
    """Fast approximation algorithms for distance metrics using BFS sampling."""

    def __init__(self, logger=None, G: nx.Graph = None, percentage: float = 0.25):
        self.logger = logger
        self.distance_graph = G
        self.percentage = percentage

        # Create analysis subgraph upon initialization
        if G:
            self.analysis_graph, self.analysis_info = self._create_analysis_subgraph(
                G, percentage)
        else:
            self.logger.warning(
                "No graph provided for FastDistanceAlgorithms.")

    @staticmethod
    def bfs_sample(G: nx.Graph, start, sample_size: int) -> nx.Graph:
        """Build a subgraph using BFS sampling from a starting node."""
        visited = {start}
        queue = [start]

        while queue and len(visited) < sample_size:
            v = queue.pop(0)
            for u in G.neighbors(v):
                if u not in visited:
                    visited.add(u)
                    queue.append(u)
                    if len(visited) >= sample_size:
                        break

        return G.subgraph(visited).copy()

    def _create_analysis_subgraph(self, G: nx.Graph, percentage: float = 0.25) -> Tuple[nx.Graph, str]:
        """Create a connected subgraph using BFS sampling and return analysis info."""
        original_nodes = G.number_of_nodes()
        sample_size = int(original_nodes * percentage)

        # Step 1: BFS sampling
        start_node = random.choice(list(G.nodes()))
        sample_graph = self.bfs_sample(G, start_node, sample_size)

        self.logger.info(f"     BFS sampling: {sample_graph.number_of_nodes()} nodes from {original_nodes} "
                         f"({percentage*100:.1f}% target, actual: {sample_graph.number_of_nodes()/original_nodes*100:.1f}%)")

        # Step 2: Get largest connected component
        ccs = list(nx.connected_components(sample_graph))
        if not ccs:
            return sample_graph, "bfs_sample_disconnected"

        largest_cc = max(ccs, key=len)
        analysis_graph = sample_graph.subgraph(largest_cc).copy()
        component_info = f"largest_cc_{len(analysis_graph.nodes())}_nodes"

        self.logger.info(f"     Largest CC: {len(analysis_graph.nodes())} nodes "
                         f"({len(analysis_graph.nodes())/sample_graph.number_of_nodes()*100:.1f}% of sample)")

        return analysis_graph, component_info

    def approximate_radius_sampling(self, G: nx.Graph) -> Tuple[float, str]:
        """Fast radius approximation using using X% of nodes."""
        try:
            radius = nx.radius(self.analysis_graph)
        except nx.NetworkXError:
            return float('inf'), "disconnected"

        return radius, f"{self.percentage*100:.1f}% BFS sampling"

    def approximate_avg_path_sampling(self, G: nx.Graph) -> Tuple[float, str]:
        """Average shortest path length using X% of nodes."""
        try:
            average_shortest_path_length = nx.average_shortest_path_length(
                self.analysis_graph)
        except nx.NetworkXNoPath:
            return float('inf'), "disconnected"

        return average_shortest_path_length, f"{self.percentage*100:.1f}% BFS sampling"

    def estimate_global_efficiency(self, G: nx.Graph) -> Tuple[float, str]:
        """Fast global efficiency estimation using using X% of nodes."""
        try:
            global_efficiency = nx.global_efficiency(self.analysis_graph)
        except nx.NetworkXError:
            return float('inf'), "disconnected"

        return global_efficiency, f"{self.percentage*100:.1f}% BFS sampling"


class GraphMetricsComputer:
    """Separated metric computation logic with improved error handling."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.computation_times = {}
        self.metric_statuses = {}
        self.warnings = []

    def compute_all_metrics(self, G: nx.Graph, threshold: int = 10000, sample_percentage: float = 0.25,
                            compute_advanced_metrics: bool = True,
                            remove_isolated: bool = True) -> Tuple[Dict[str, Any], nx.Graph]:
        """Compute all graph metrics with proper validation and error handling."""
        start = time.time()

        # Validate input graph
        if not self._validate_graph(G):
            raise ValueError("Invalid input graph")

        results = {}
        original_G = G.copy()

        # Basic validation and preprocessing
        results.update(self._compute_basic_properties(G))

        # Handle isolated nodes
        if remove_isolated:
            G, isolation_stats = self._handle_isolated_nodes(G)
            results.update(isolation_stats)

        # Core metric categories (with proper grouping)
        results.update(self._compute_structure_metrics(G))
        results.update(self._compute_connectivity_metrics(G, original_G))
        results.update(self._compute_degree_metrics(G))
        results.update(self._compute_component_metrics(G))

        if G.is_directed():
            results.update(self._compute_directed_metrics(G))

        # Advanced metrics (expensive computations)
        if compute_advanced_metrics:
            results.update(self._compute_clustering_metrics(G))
            results.update(self._compute_distance_metrics(
                G, threshold, sample_percentage))
            results.update(self._compute_centrality_metrics(G, threshold))
            results.update(self._compute_structural_features(G))

        # Add computation metadata
        elapsed = time.time() - start
        results['_computation_metadata'] = {
            'total_time': elapsed,
            'computation_times': self.computation_times,
            'metric_statuses': self.metric_statuses,
            'warnings': self.warnings
        }

        return results, G

    def _validate_graph(self, G: nx.Graph) -> bool:
        """Validate input graph."""
        if G is None:
            self.warnings.append("Graph is None")
            return False

        if G.number_of_nodes() == 0:
            self.warnings.append("Graph has no nodes")
            return False

        return True

    def _compute_with_timer(self, metric_name: str, computation_func, *args, **kwargs):
        """Execute computation with timing and error handling."""
        with MetricTimer(metric_name, self.logger) as timer:
            try:
                result = computation_func(*args, **kwargs)
                self.metric_statuses[metric_name] = MetricStatus.COMPUTED
                self.computation_times[metric_name] = timer.duration
                return result
            except Exception as e:
                self.metric_statuses[metric_name] = MetricStatus.FAILED
                self.warnings.append(f"{metric_name}: {str(e)}")
                self.logger.warning(f"Failed to compute {metric_name}: {e}")
                return None

    def _compute_basic_properties(self, G: nx.Graph) -> Dict[str, Any]:
        """Compute basic graph structure properties."""
        return {
            "is_directed": G.is_directed(),
            "is_weighted": nx.is_weighted(G),
            "is_multigraph": G.is_multigraph(),
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges()
        }

    def _handle_isolated_nodes(self, G: nx.Graph) -> Tuple[nx.Graph, Dict[str, Any]]:
        """Handle isolated node and small component removal with proper tracking."""
        original_nodes = G.number_of_nodes()

        # Set default minimum component size based on graph size
        if original_nodes < 1000:
            # Keep components with 2+ nodes
            min_component_size = 2
        elif original_nodes < 10000:
            # Keep components with 5+ nodes
            min_component_size = 5
        else:
            # 0.1% of graph size, minimum 10
            min_component_size = max(10, int(original_nodes * 0.001))

        # Step 1: Remove isolated nodes (degree 0)
        zero_degree_nodes = [n for n, d in G.degree() if d == 0]
        G_after_isolated = G.copy()
        if zero_degree_nodes:
            G_after_isolated.remove_nodes_from(zero_degree_nodes)

        # Step 2: Identify and remove small components
        if G.is_directed():
            components = list(nx.weakly_connected_components(G_after_isolated))
        else:
            components = list(nx.connected_components(G_after_isolated))

        # Find components to remove (smaller than threshold)
        small_components = [comp for comp in components if len(
            comp) < min_component_size]
        nodes_in_small_components = set()
        for comp in small_components:
            nodes_in_small_components.update(comp)

        # Create final graph without small components
        final_graph = G_after_isolated.copy()
        if nodes_in_small_components:
            final_graph.remove_nodes_from(nodes_in_small_components)

        # Calculate statistics
        isolation_stats = {
            "num_nodes_zero_total_degree": len(zero_degree_nodes),
            "num_isolated_nodes_removed": len(zero_degree_nodes),
            "min_component_size_threshold": min_component_size,
            "num_small_components_removed": len(small_components),
            "num_nodes_in_small_components_removed": len(nodes_in_small_components),
            "total_nodes_removed": len(zero_degree_nodes) + len(nodes_in_small_components),
            "num_nodes_after_removal": final_graph.number_of_nodes(),
            "num_edges_after_removal": final_graph.number_of_edges(),
            "removal_percentage": (
                (len(zero_degree_nodes) + len(nodes_in_small_components) * 100) /
                original_nodes if original_nodes > 0 else 0)
        }

        return final_graph, isolation_stats

    def _compute_structure_metrics(self, G: nx.Graph) -> Dict[str, Any]:
        """Compute structural features."""
        metrics = {}

        # Self-loops
        metrics["self_loops"] = self._compute_with_timer(
            "self_loops", nx.number_of_selfloops, G
        ) or 0

        return metrics

    def _compute_connectivity_metrics(self, G: nx.Graph, original_G: nx.Graph) -> Dict[str, Any]:
        """Compute connectivity and density metrics."""
        metrics = {}

        # Density
        metrics["density"] = self._compute_with_timer(
            "density", nx.density, G) or 0.0

        # Connectivity for undirected version
        G_undirected = G.to_undirected() if G.is_directed() else G
        metrics["is_connected_undirected"] = self._compute_with_timer(
            "connectivity_check", nx.is_connected, G_undirected
        ) or False

        # Reciprocity for directed graphs
        if G.is_directed():
            metrics["reciprocity"] = self._compute_with_timer(
                "reciprocity", self._safe_reciprocity, G
            )

        return metrics

    def _safe_reciprocity(self, G: nx.DiGraph) -> float:
        """Safely compute reciprocity."""
        try:
            return nx.reciprocity(G)
        except:
            return 0.0

    def _compute_degree_metrics(self, G: nx.Graph) -> Dict[str, Any]:
        """Compute degree distribution statistics."""
        metrics = {}

        all_degrees = [d for _, d in G.degree()]
        if not all_degrees:
            return metrics

        metrics.update({
            "average_degree": np.mean(all_degrees),
            "max_degree": max(all_degrees),
            "min_degree": min(all_degrees),
            "median_degree": np.median(all_degrees),
            "std_dev_degree": np.std(all_degrees)
        })

        return metrics

    def _compute_component_metrics(self, G: nx.Graph) -> Dict[str, Any]:
        """Compute connected component analysis."""
        metrics = {}

        if G.is_directed():
            # Weakly connected components
            weak_components = list(nx.weakly_connected_components(G))
            weak_sizes = [len(c) for c in weak_components]

            metrics.update({
                "num_weakly_connected_components": len(weak_components),
                "largest_weakly_connected_component_size": max(weak_sizes) if weak_sizes else 0,
                "weakly_cc_size_mean": np.mean(weak_sizes) if weak_sizes else 0,
                "weakly_cc_size_median": np.median(weak_sizes) if weak_sizes else 0,
                "weakly_connected_component_sizes": weak_sizes
            })

            # Strongly connected components
            strong_components = list(nx.strongly_connected_components(G))
            strong_sizes = [len(c) for c in strong_components]

            metrics.update({
                "is_strongly_connected": nx.is_strongly_connected(G),
                "is_weakly_connected": nx.is_weakly_connected(G),
                "num_strongly_connected_components": len(strong_components),
                "largest_strongly_connected_component_size": max(strong_sizes) if strong_sizes else 0,
                "strongly_cc_size_mean": np.mean(strong_sizes) if strong_sizes else 0,
                "strongly_cc_size_median": np.median(strong_sizes) if strong_sizes else 0,
                "strongly_connected_component_sizes": strong_sizes
            })
        else:
            # Undirected connected components
            components = list(nx.connected_components(G))
            comp_sizes = [len(c) for c in components]

            metrics.update({
                "num_connected_components": len(components),
                "largest_connected_component_size": max(comp_sizes) if comp_sizes else 0,
                "cc_size_mean": np.mean(comp_sizes) if comp_sizes else 0,
                "cc_size_median": np.median(comp_sizes) if comp_sizes else 0,
                "connected_component_sizes": comp_sizes
            })

        return metrics

    def _compute_directed_metrics(self, G: nx.DiGraph) -> Dict[str, Any]:
        """Compute directed graph specific metrics."""
        metrics = {}

        # In/out degree statistics
        in_degrees = [d for _, d in G.in_degree()]
        out_degrees = [d for _, d in G.out_degree()]

        if in_degrees:
            metrics.update({
                "num_nodes_zero_in_degree": len([d for d in in_degrees if d == 0]),
                "max_in_degree": max(in_degrees),
                "min_in_degree": min(in_degrees),
                "mean_in_degree": np.mean(in_degrees),
                "median_in_degree": np.median(in_degrees)
            })

        if out_degrees:
            metrics.update({
                "num_nodes_zero_out_degree": len([d for d in out_degrees if d == 0]),
                "max_out_degree": max(out_degrees),
                "min_out_degree": min(out_degrees),
                "mean_out_degree": np.mean(out_degrees),
                "median_out_degree": np.median(out_degrees)
            })

        # PageRank
        if G.number_of_nodes() <= 10000:
            pagerank = self._compute_with_timer(
                "pagerank", nx.pagerank, G, alpha=0.85)
            if pagerank:
                metrics["average_pagerank"] = np.mean(list(pagerank.values()))
                metrics["max_pagerank"] = max(pagerank.values())

        return metrics

    def _compute_clustering_metrics(self, G: nx.Graph) -> Dict[str, Any]:
        """Compute clustering coefficients with scalability awareness."""
        metrics = {}

        # Transitivity
        metrics["transitivity"] = self._compute_with_timer(
            "transitivity", nx.transitivity, G) or 0.0

        # Average clustering
        metrics["average_clustering"] = self._compute_with_timer(
            "average_clustering", nx.average_clustering, G
        ) or 0.0

        # Filtered clustering (nodes with degree > 1)
        clustering_vals = self._compute_with_timer(
            "node_clustering", nx.clustering, G) or {}
        if clustering_vals:
            degrees = dict(G.degree())
            filtered_clustering = [clustering_vals[n] for n in clustering_vals
                                   if degrees.get(n, 0) > 1]
            metrics["filtered_average_clustering"] = np.mean(
                filtered_clustering) if filtered_clustering else 0.0

        return metrics

    def _compute_distance_metrics(self, G: nx.Graph, threshold: int, sample_percentage: float) -> Dict[str, Any]:
        """Compute path-based metrics with proper component validation."""
        metrics = {}

        # Determine if we should use fast approximations
        use_fast_methods = G.number_of_nodes() > threshold // 2

        # Choose appropriate component for analysis
        if G.is_directed():
            if nx.is_strongly_connected(G):
                analysis_graph = G
                prefix = ""
            else:
                # Use largest strongly connected component
                sccs = list(nx.strongly_connected_components(G))
                if sccs:
                    largest_scc = max(sccs, key=len)
                    analysis_graph = G.subgraph(largest_scc).copy()
                    prefix = "largest_scc_"
                else:
                    return metrics
        else:
            if nx.is_connected(G):
                analysis_graph = G
                prefix = ""
            else:
                # Use largest connected component
                ccs = list(nx.connected_components(G))
                if ccs:
                    largest_cc = max(ccs, key=len)
                    analysis_graph = G.subgraph(largest_cc).copy()
                    prefix = "largest_cc_"
                else:
                    return metrics

        # For directed graphs, use undirected version for distance calculations
        distance_graph = analysis_graph.to_undirected(
        ) if analysis_graph.is_directed() else analysis_graph

        # Log computation strategy
        if use_fast_methods:
            self.logger.info(
                f"Distance metrics: using BFS sampling on {distance_graph.number_of_nodes()} nodes")

            # Initialize fast algorithms with the distance graph
            fast_algo = FastDistanceAlgorithms(
                logger=self.logger, G=distance_graph, percentage=sample_percentage)
        else:
            self.logger.info(
                f"Distance metrics: using exact methods on {distance_graph.number_of_nodes()} nodes")

        # DIAMETER
        if use_fast_methods:
            diameter = self._compute_with_timer(
                f"{prefix}diameter_fast",
                approx.diameter,
                distance_graph
            )
            if diameter is not None:
                metrics[f"{prefix}diameter"] = diameter
                metrics[f"{prefix}diameter_method"] = "Aproximation from networkx"
        else:
            diameter = self._compute_with_timer(
                f"{prefix}diameter", nx.diameter, distance_graph)
            if diameter is not None:
                metrics[f"{prefix}diameter"] = diameter
                metrics[f"{prefix}diameter_method"] = "exact"

        # RADIUS
        if use_fast_methods:
            radius, method = self._compute_with_timer(
                f"{prefix}radius_fast",
                fast_algo.approximate_radius_sampling,
                distance_graph  # not used inside function, sample graph used instead
            )
            if radius is not None:
                metrics[f"{prefix}radius"] = radius
                metrics[f"{prefix}radius_method"] = method
        else:
            radius = self._compute_with_timer(
                f"{prefix}radius", nx.radius, distance_graph)
            if radius is not None:
                metrics[f"{prefix}radius"] = radius
                metrics[f"{prefix}radius_method"] = "exact"

        # AVERAGE SHORTEST PATH LENGTH
        if use_fast_methods:
            avg_path, method = self._compute_with_timer(
                f"{prefix}avg_path_fast",
                fast_algo.approximate_avg_path_sampling,
                distance_graph  # not used inside function, sample graph used instead
            )
            if avg_path is not None:
                metrics[f"{prefix}average_shortest_path_length"] = avg_path
                metrics[f"{prefix}average_shortest_path_length_method"] = method
        else:
            avg_path = self._compute_with_timer(
                f"{prefix}average_shortest_path_length",
                nx.average_shortest_path_length,
                distance_graph  # not used inside function, sample graph used instead
            )
            if avg_path is not None:
                metrics[f"{prefix}average_shortest_path_length"] = avg_path
                metrics[f"{prefix}average_shortest_path_length_method"] = "exact"

        # GLOBAL EFFICIENCY
        if use_fast_methods:
            efficiency, method = self._compute_with_timer(
                f"{prefix}global_efficiency_fast",
                fast_algo.estimate_global_efficiency,
                distance_graph
            )
            if efficiency is not None:
                metrics[f"{prefix}global_efficiency"] = efficiency
                metrics[f"{prefix}global_efficiency_method"] = method
        else:
            efficiency = self._compute_with_timer(
                f"{prefix}global_efficiency",
                nx.global_efficiency,
                distance_graph
            )
            if efficiency is not None:
                metrics[f"{prefix}global_efficiency"] = efficiency
                metrics[f"{prefix}global_efficiency_method"] = "exact"

        return metrics

    def _compute_centrality_metrics(self, G: nx.Graph, threshold: int) -> Dict[str, Any]:
        """Compute centrality measures with scalability limits."""
        metrics = {}

        node_count = G.number_of_nodes()

        # Degree centrality (always computable)
        degree_cent = self._compute_with_timer(
            "degree_centrality", nx.degree_centrality, G)
        if degree_cent:
            metrics["average_degree_centrality"] = np.mean(
                list(degree_cent.values()))

        # Closeness centrality (moderate cost)
        if node_count <= threshold:
            closeness_cent = self._compute_with_timer(
                "closeness_centrality", nx.closeness_centrality, G)
            if closeness_cent:
                metrics["average_closeness_centrality"] = np.mean(
                    list(closeness_cent.values()))

        # Betweenness centrality (expensive)
        if node_count <= threshold // 10:
            k = min(500, node_count)
            betw_cent = self._compute_with_timer(
                "betweenness_centrality",
                lambda g: nx.betweenness_centrality(g, normalized=True, k=k),
                G
            )
            if betw_cent:
                metrics["average_betweenness_centrality"] = np.mean(
                    list(betw_cent.values()))

        # Eigenvector centrality (can fail to converge)
        if node_count <= threshold // 5:
            try:
                eigen_cent = self._compute_with_timer(
                    "eigenvector_centrality",
                    lambda g: nx.eigenvector_centrality(
                        g, max_iter=100, tol=1e-4),
                    G
                )
                if eigen_cent:
                    metrics["average_eigenvector_centrality"] = np.mean(
                        list(eigen_cent.values()))
            except:
                metrics["average_eigenvector_centrality"] = "Failed to converge"
                self.metric_statuses["eigenvector_centrality"] = MetricStatus.FAILED

        return metrics

    def _compute_structural_features(self, G: nx.Graph) -> Dict[str, Any]:
        """Compute advanced structural features."""
        metrics = {}

        # Bridges (undirected graphs only)
        if not G.is_directed():
            bridges = self._compute_with_timer(
                "bridges", lambda g: list(nx.bridges(g)), G)
            if bridges is not None:
                metrics["num_bridges"] = len(bridges)

        # Degree assortativity
        metrics["degree_assortativity"] = self._compute_with_timer(
            "degree_assortativity", nx.degree_assortativity_coefficient, G
        )

        # Core decomposition
        G_for_core = G.to_undirected() if G.is_directed() else G
        if nx.number_of_selfloops(G_for_core) > 0:
            G_for_core = G_for_core.copy()
            G_for_core.remove_edges_from(nx.selfloop_edges(G_for_core))

        core_numbers = self._compute_with_timer(
            "core_decomposition", nx.core_number, G_for_core)
        if core_numbers:
            metrics["max_core_number"] = max(core_numbers.values())
            metrics["mean_core_number"] = np.mean(list(core_numbers.values()))

        # Rich club coefficient (expensive for large graphs)
        if not G.is_directed():
            degrees = [d for _, d in G.degree()]
            if degrees:
                k = max(5, int(np.percentile(degrees, 90)))
                rich_club = self._compute_with_timer(
                    "rich_club_coefficient",
                    lambda g: nx.rich_club_coefficient(
                        g, normalized=False).get(k, None),
                    G
                )
                metrics["rich_club_coefficient"] = rich_club

        return metrics


class CitationNetworkAnalyzer:
    """Specialized analysis for citation networks."""

    @staticmethod
    def analyze(G: nx.DiGraph, top_n: int = 10) -> Dict[str, Any]:
        """
        Analyzes a citation network with specialized metrics and visualizations for citation data.

        Parameters
        ----------
        G : networkx.DiGraph
            The citation graph to analyze
        top_n : int, default=10
            Number of top nodes to show in rankings

        Returns
        -------
        dict
            Dictionary containing all computed citation metrics
        """
        results = {}

        # Basic degree statistics
        out_degrees = [deg for _, deg in G.out_degree()]
        in_degrees = [deg for _, deg in G.in_degree()]

        # Summary statistics for citations made (out-degree)
        results["citations_made_mean"] = np.mean(out_degrees)
        results["citations_made_median"] = np.median(out_degrees)
        results["citations_made_max"] = np.max(out_degrees)
        results["citations_made_min"] = np.min(out_degrees)
        results["citations_made_std"] = np.std(out_degrees)

        # Summary statistics for citations received (in-degree)
        results["citations_received_mean"] = np.mean(in_degrees)
        results["citations_received_median"] = np.median(in_degrees)
        results["citations_received_max"] = np.max(in_degrees)
        results["citations_received_min"] = np.min(in_degrees)
        results["citations_received_std"] = np.std(in_degrees)

        # Top citing nodes (articles that cite the most others)
        out_deg_dict = dict(G.out_degree())
        top_out_nodes = sorted(out_deg_dict.items(),
                               key=lambda x: x[1], reverse=True)[:top_n]
        results["top_citing_nodes"] = [
            (node, G.nodes[node].get('title', str(node)), degree)
            for node, degree in top_out_nodes
        ]

        # Top cited nodes (articles cited by the most others)
        in_deg_dict = dict(G.in_degree())
        top_in_nodes = sorted(in_deg_dict.items(),
                              key=lambda x: x[1], reverse=True)[:top_n]
        results["top_cited_nodes"] = [
            (node, G.nodes[node].get('title', str(node)), degree)
            for node, degree in top_in_nodes
        ]

        # Special node categories
        results["num_isolated_nodes"] = len(
            [n for n, d in G.degree() if d == 0])
        results["num_terminal_nodes"] = len(
            [n for n, d in G.out_degree() if d == 0])
        results["num_source_nodes"] = len(
            [n for n, d in G.in_degree() if d == 0])

        # H-index of the network
        in_degrees_sorted = sorted(in_degrees, reverse=True)
        h_index = 0
        for i, citations in enumerate(in_degrees_sorted):
            if i + 1 <= citations:
                h_index = i + 1
            else:
                break
        results["h_index"] = h_index

        # PageRank - article influence score
        pagerank_scores = nx.pagerank(G, alpha=0.85)
        pagerank_top_nodes = sorted(
            pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        results["pagerank_top_nodes"] = [
            (node, G.nodes[node].get('title', str(node)), score)
            for node, score in pagerank_top_nodes
        ]
        results["pagerank_mean"] = np.mean(list(pagerank_scores.values()))
        results["pagerank_std"] = np.std(list(pagerank_scores.values()))

        # Citation concentration (Gini coefficient)
        def gini_coefficient(x):
            """Calculate Gini coefficient for inequality measurement."""
            sorted_x = sorted(x)
            n = len(x)
            cumsum = np.cumsum(sorted_x)
            return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0

        results["citation_gini_coefficient"] = gini_coefficient(in_degrees)

        # Self-citation rate (if graph has self-loops)
        self_citations = nx.number_of_selfloops(G)
        total_citations = G.number_of_edges()
        results["self_citation_rate"] = self_citations / \
            total_citations if total_citations > 0 else 0

        # Citation age analysis (if nodes have temporal data)
        results["temporal_analysis_available"] = any(
            'year' in G.nodes[n] or 'date' in G.nodes[n]
            for n in list(G.nodes())[:10]
        )

        # Authority and hub scores (HITS algorithm)
        try:
            hits_scores = nx.hits(G, max_iter=100, normalized=True)
            authorities, hubs = hits_scores
            results["top_authorities"] = sorted(
                authorities.items(), key=lambda x: x[1], reverse=True)[:top_n]
            results["top_hubs"] = sorted(
                hubs.items(), key=lambda x: x[1], reverse=True)[:top_n]
            results["authority_mean"] = np.mean(list(authorities.values()))
            results["hub_mean"] = np.mean(list(hubs.values()))
        except Exception:
            results["hits_analysis_failed"] = True

        # Citation density
        results["citation_density"] = results.get("citations_made_mean", 0) / (G.number_of_nodes() - 1) \
            if G.number_of_nodes() > 1 else 0

        return results
