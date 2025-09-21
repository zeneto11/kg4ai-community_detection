import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Set, Tuple
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Set, Dict, Tuple
from pathlib import Path
from collections import defaultdict
import matplotlib as mpl
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')


class CommunityVisualizer:
    """Enhanced class for visualizing community detection results through macro-graphs."""

    def __init__(self, original_graph: nx.Graph, max_communities_to_plot: int = 400):
        """Initialize with the original graph.

        Args:
            original_graph: The original graph used for community detection
        """
        self.original_graph = original_graph
        self.max_communities_to_plot = max_communities_to_plot
        self._setup_visualization_style()

    def _setup_visualization_style(self):
        """Setup consistent visualization style."""
        plt.style.use('default')
        mpl.rcParams['figure.dpi'] = 300
        mpl.rcParams['savefig.dpi'] = 300
        mpl.rcParams['font.size'] = 10
        mpl.rcParams['axes.titlesize'] = 14
        mpl.rcParams['axes.labelsize'] = 12

    def _filter_largest_communities(self, communities: List[Set]) -> List[Set]:
        """Filter communities to keep only the largest ones up to the maximum limit."""
        if len(communities) <= self.max_communities_to_plot:
            return communities

        print(
            f"Filtering communities: {len(communities)} -> {self.max_communities_to_plot} largest")

        # Sort communities by size (largest first)
        sorted_communities = sorted(communities, key=len, reverse=True)

        # Keep only the largest communities
        filtered_communities = sorted_communities[:self.max_communities_to_plot]

        # Calculate statistics about what was filtered
        original_total_nodes = sum(len(comm) for comm in communities)
        filtered_total_nodes = sum(len(comm) for comm in filtered_communities)
        coverage_percentage = (filtered_total_nodes /
                               original_total_nodes) * 100

        print(f"Coverage: {coverage_percentage:.1f}% of nodes preserved "
              f"({filtered_total_nodes}/{original_total_nodes} nodes)")

        return filtered_communities

    def create_macro_graph(self, communities: List[Set], weighted: bool = True) -> nx.DiGraph:
        """Create a macro-graph where nodes are communities and edges represent inter-community connections.

        Args:
            communities: List of sets representing communities
            weighted: Whether to create weighted edges based on citation counts

        Returns:
            NetworkX DiGraph representing the macro-graph
        """
        # Filter communities if necessary
        communities = self._filter_largest_communities(communities)

        # Create community mapping
        node_to_community = {}
        for comm_id, community in enumerate(communities):
            for node in community:
                node_to_community[node] = comm_id

        # Create macro-graph
        macro_graph = nx.DiGraph()

        # Add community nodes with metadata
        for comm_id, community in enumerate(communities):
            community_size = len(community)
            # Extract titles if available
            titles = []
            for node in community:
                if 'title' in self.original_graph.nodes[node]:
                    titles.append(self.original_graph.nodes[node]['title'])

            macro_graph.add_node(comm_id,
                                 size=community_size,
                                 titles=titles,
                                 members=community)

        # Count inter-community edges (both directions separately for directed graph)
        edge_counts = defaultdict(int)

        for u, v in self.original_graph.edges():
            if u in node_to_community and v in node_to_community:
                comm_u = node_to_community[u]
                comm_v = node_to_community[v]

                # Only count inter-community edges
                if comm_u != comm_v:
                    edge_counts[(comm_u, comm_v)] += 1

        # Add edges to macro-graph
        for (comm_u, comm_v), count in edge_counts.items():
            if weighted:
                macro_graph.add_edge(
                    comm_u, comm_v, weight=count, citations=count)
            else:
                macro_graph.add_edge(comm_u, comm_v, weight=1, citations=count)

        return macro_graph

    def _calculate_adaptive_threshold(self, macro_graph: nx.DiGraph) -> int:
        """Calculate adaptive threshold based on citation statistics."""
        if not macro_graph.edges():
            return 1

        citations = [macro_graph[u][v].get(
            'citations', 0) for u, v in macro_graph.edges()]

        if not citations:
            return 1

        # Calculate statistics
        mean_citations = np.mean(citations)
        std_citations = np.std(citations)
        median_citations = np.median(citations)

        # Use mean + 0.5 * std as threshold, but ensure it's reasonable
        threshold = max(1, int(mean_citations + 0.5 * std_citations))

        # Alternative: use 75th percentile if it's more conservative
        percentile_75 = np.percentile(citations, 75)
        threshold = max(threshold, int(percentile_75))

        print(f"Citation statistics - Mean: {mean_citations:.1f}, Std: {std_citations:.1f}, "
              f"Median: {median_citations:.1f}, 75th percentile: {percentile_75:.1f}")
        print(f"Adaptive threshold set to: {threshold}")

        return threshold

    def create_thresholded_macro_graph(self, communities: List[Set], threshold: int = None) -> nx.DiGraph:
        """Create a thresholded macro-graph where edges exist only if citation count exceeds threshold.

        Args:
            communities: List of sets representing communities
            threshold: Minimum number of citations required for an edge. If None, calculate adaptively.

        Returns:
            NetworkX DiGraph representing the thresholded macro-graph
        """
        macro_graph = self.create_macro_graph(communities, weighted=True)

        # Calculate adaptive threshold if not provided
        if threshold is None:
            threshold = self._calculate_adaptive_threshold(macro_graph)

        # Remove edges below threshold
        edges_to_remove = []
        for u, v, data in macro_graph.edges(data=True):
            if data.get('citations', 0) < threshold:
                edges_to_remove.append((u, v))

        macro_graph.remove_edges_from(edges_to_remove)

        # Store threshold in graph for later reference
        macro_graph.graph['threshold'] = threshold

        return macro_graph

    def _force_directed_layout_with_collision_detection(self, macro_graph: nx.DiGraph,
                                                        node_sizes: List[float],
                                                        iterations: int = 300) -> Dict[int, Tuple[float, float]]:
        """Enhanced force-directed layout with better collision detection."""
        n_nodes = len(macro_graph.nodes())
        if n_nodes == 0:
            return {}
        if n_nodes == 1:
            return {list(macro_graph.nodes())[0]: (0, 0)}

        # Start with a better initial layout - use circular with more spacing
        try:
            pos = nx.circular_layout(macro_graph, scale=10)  # Increased scale
        except:
            pos = {node: (0, 0) for node in macro_graph.nodes()}

        # Convert to arrays for easier manipulation
        nodes = list(macro_graph.nodes())
        positions = np.array([pos[node] for node in nodes])

        # Convert visualization sizes to layout radii (scaled appropriately)
        # The node_sizes are in points (1000-4000), convert to layout units
        max_vis_size = max(node_sizes) if node_sizes else 4000
        node_radii = np.array(
            [(size / max_vis_size) * 2.0 for size in node_sizes])

        # Force-directed algorithm with improved collision detection
        for iteration in range(iterations):
            forces = np.zeros_like(positions)

            # Enhanced repulsive forces between all nodes
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    diff = positions[i] - positions[j]
                    dist = np.linalg.norm(diff)

                    if dist < 1e-6:  # Avoid division by zero
                        # Add random jitter to overlapping nodes
                        jitter = np.random.normal(0, 0.1, 2)
                        positions[i] += jitter
                        positions[j] -= jitter
                        continue

                    # Minimum distance based on node sizes + padding
                    min_dist = (node_radii[i] + node_radii[j]
                                ) * 3.0  # Increased from 2.5

                    if dist < min_dist:
                        # Stronger repulsive force with inverse square law
                        force_mag = (min_dist - dist) ** 2 / \
                            (dist + 1e-6) * 0.5
                        force = diff * force_mag / (dist + 1e-6)
                        forces[i] += force
                        forces[j] -= force
                    else:
                        # Weak repulsive force even when not overlapping
                        force_mag = 0.1 / (dist + 1e-6)
                        force = diff * force_mag / (dist + 1e-6)
                        forces[i] += force * 0.1
                        forces[j] -= force * 0.1

            # Attractive forces for connected nodes
            for u, v in macro_graph.edges():
                i, j = nodes.index(u), nodes.index(v)
                diff = positions[j] - positions[i]
                dist = np.linalg.norm(diff)

                if dist > 1e-6:
                    # Attractive force with logarithmic scaling
                    weight = macro_graph[u][v].get('weight', 1)
                    ideal_distance = 2.0  # Base ideal distance
                    force_mag = np.log(weight + 1) * 0.1 * \
                        (dist - ideal_distance)
                    force = diff * force_mag / (dist + 1e-6)
                    forces[i] += force
                    forces[j] -= force

            # Apply forces with adaptive damping
            # Reduce damping over time
            damping = 0.8 * (1 - iteration / iterations)
            positions += forces * damping

            # Center the layout periodically
            if iteration % 10 == 0:
                positions -= np.mean(positions, axis=0)

        # Final centering
        positions -= np.mean(positions, axis=0)

        # Scale the final layout for better spacing
        max_pos = np.max(np.abs(positions))
        if max_pos > 0:
            positions *= 5.0 / max_pos  # Scale to fit in [-5, 5] range

        return {node: tuple(positions[i]) for i, node in enumerate(nodes)}

    def _adjust_overlapping_nodes(self, pos: Dict[int, Tuple[float, float]],
                                  node_sizes: List[float]) -> Dict[int, Tuple[float, float]]:
        """Final adjustment to prevent overlapping nodes with better algorithm."""
        if len(pos) <= 1:
            return pos

        nodes = list(pos.keys())
        positions = np.array([pos[node] for node in nodes])

        # Convert visualization sizes to layout radii
        max_vis_size = max(node_sizes) if node_sizes else 4000
        node_radii = np.array(
            [(size / max_vis_size) * 2.0 for size in node_sizes])

        # Multiple passes with increasing force
        for pass_num in range(20):
            max_overlap = 0
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    diff = positions[i] - positions[j]
                    dist = np.linalg.norm(diff)
                    min_dist = (node_radii[i] + node_radii[j]) * 2.5

                    if dist < min_dist and dist > 1e-6:
                        overlap = min_dist - dist
                        max_overlap = max(max_overlap, overlap)

                        # Move nodes apart proportionally to their sizes
                        move_dist = overlap * 0.7
                        move_vector = diff / dist * move_dist

                        # Larger nodes move less
                        size_ratio = node_radii[j] / \
                            (node_radii[i] + node_radii[j])
                        positions[i] += move_vector * size_ratio
                        positions[j] -= move_vector * (1 - size_ratio)

            # Break early if no significant overlaps
            if max_overlap < 0.01:
                break

        # Re-center and scale
        positions -= np.mean(positions, axis=0)
        max_pos = np.max(np.abs(positions))
        if max_pos > 0:
            positions *= 4.0 / max_pos

        return {node: tuple(positions[i]) for i, node in enumerate(nodes)}

    def _get_hierarchical_layout(self, macro_graph: nx.DiGraph, node_sizes: List[float]) -> Dict[int, Tuple[float, float]]:
        """Create hierarchical layout based on community sizes and connections."""
        if not macro_graph.nodes():
            return {}

        # Get node importance (combination of size and connectivity)
        node_importance = {}
        for node in macro_graph.nodes():
            size = macro_graph.nodes[node].get('size', 1)
            in_degree = macro_graph.in_degree(node, weight='weight')
            out_degree = macro_graph.out_degree(node, weight='weight')
            # Importance combines size and connectivity
            importance = size * 0.7 + (in_degree + out_degree) * 0.3
            node_importance[node] = importance

        # Sort nodes by importance
        sorted_nodes = sorted(node_importance.items(),
                              key=lambda x: x[1], reverse=True)

        # Create concentric circles based on importance
        pos = {}
        n_nodes = len(sorted_nodes)

        if n_nodes == 1:
            pos[sorted_nodes[0][0]] = (0, 0)
        else:
            # Place most important nodes in center, others in expanding circles
            center_threshold = max(1, n_nodes // 4)  # Top 25% in center
            middle_threshold = max(2, n_nodes // 2)  # Next 25% in middle ring

            # Central cluster
            center_nodes = [node for node,
                            _ in sorted_nodes[:center_threshold]]
            if len(center_nodes) == 1:
                pos[center_nodes[0]] = (0, 0)
            else:
                angles = np.linspace(
                    0, 2*np.pi, len(center_nodes), endpoint=False)
                for i, node in enumerate(center_nodes):
                    pos[node] = (0.5 * np.cos(angles[i]),
                                 0.5 * np.sin(angles[i]))

            # Middle ring
            middle_nodes = [node for node,
                            _ in sorted_nodes[center_threshold:middle_threshold]]
            if middle_nodes:
                angles = np.linspace(
                    0, 2*np.pi, len(middle_nodes), endpoint=False)
                for i, node in enumerate(middle_nodes):
                    pos[node] = (2.0 * np.cos(angles[i]),
                                 2.0 * np.sin(angles[i]))

            # Outer ring
            outer_nodes = [node for node, _ in sorted_nodes[middle_threshold:]]
            if outer_nodes:
                angles = np.linspace(
                    0, 2*np.pi, len(outer_nodes), endpoint=False)
                for i, node in enumerate(outer_nodes):
                    pos[node] = (3.5 * np.cos(angles[i]),
                                 3.5 * np.sin(angles[i]))

        return pos

    def _get_optimized_layout(self, macro_graph: nx.DiGraph, node_sizes: List[float]) -> Dict[int, Tuple[float, float]]:
        """Get optimized layout for better visualization of directed graphs."""
        n_nodes = len(macro_graph.nodes())

        if n_nodes == 0:
            return {}
        if n_nodes == 1:
            return {list(macro_graph.nodes())[0]: (0, 0)}

        # Choose layout strategy based on graph size and connectivity
        n_edges = len(macro_graph.edges())
        density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0

        print(
            f"Graph stats: {n_nodes} nodes, {n_edges} edges, density: {density:.3f}")

        if n_nodes <= 8:
            # Small graphs: use spring layout with collision detection
            pos = self._force_directed_layout_with_collision_detection(
                macro_graph, node_sizes, iterations=300)
        elif density < 0.1:  # Sparse graph
            # Use hierarchical layout for sparse graphs
            pos = self._get_hierarchical_layout(macro_graph, node_sizes)
        else:
            # Dense graph: use force-directed with collision detection
            pos = self._force_directed_layout_with_collision_detection(
                macro_graph, node_sizes, iterations=200)

        # Final collision check and adjustment
        pos = self._adjust_overlapping_nodes(pos, node_sizes)

        return pos

    def _get_node_colors_and_sizes(self, macro_graph: nx.DiGraph) -> Tuple[List, List]:
        """Get node colors and sizes based on community properties."""
        node_sizes = []
        community_sizes = [macro_graph.nodes[node].get(
            'size', 1) for node in macro_graph.nodes()]

        # Normalize node sizes for better visualization
        if community_sizes:
            max_size = max(community_sizes)
            min_size = min(community_sizes)

            if max_size > min_size:
                # Scale sizes between 1000 and 4000 for better visibility
                node_sizes = [1000 + 3000 * ((size - min_size) / (max_size - min_size))
                              for size in community_sizes]
            else:
                node_sizes = [2000] * len(community_sizes)
        else:
            node_sizes = [2000] * len(macro_graph.nodes())

        # Generate distinct colors using a better color scheme
        n_nodes = len(macro_graph.nodes())
        if n_nodes <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_nodes))
        elif n_nodes <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, n_nodes))
        else:
            # For many nodes, use a continuous colormap
            colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_nodes))

        return colors, node_sizes

    def _get_edge_properties(self, macro_graph: nx.DiGraph) -> Tuple[List, List, List]:
        """Get edge widths, colors, and styles based on weights."""
        edge_widths = []
        edge_colors = []
        edge_styles = []

        if macro_graph.edges():
            edge_weights = [macro_graph[u][v].get(
                'weight', 1) for u, v in macro_graph.edges()]
            max_weight = max(edge_weights) if edge_weights else 1

            # Create colormap for edges
            cmap = LinearSegmentedColormap.from_list(
                'edge_cmap', ['lightsteelblue', 'navy'])

            for weight in edge_weights:
                # Scale edge width (1 to 8)
                width = 1 + 7 * (weight / max_weight) if max_weight > 0 else 2
                edge_widths.append(width)

                # Color based on weight
                if max_weight > 1:
                    color = cmap(weight / max_weight)
                else:
                    color = 'steelblue'
                edge_colors.append(color)

                edge_styles.append('->')

        return edge_widths, edge_colors, edge_styles

    def visualize_macro_graph(self, macro_graph: nx.DiGraph, title: str,
                              output_path: Path, algorithm: str,
                              graph_type: str = "weighted") -> Path:
        """Enhanced visualization of macro-graph with better layout and styling."""
        fig, ax = plt.subplots(figsize=(18, 14))

        # Get node properties first (needed for layout)
        node_colors, node_sizes = self._get_node_colors_and_sizes(macro_graph)

        # Get optimized layout with node size information
        pos = self._get_optimized_layout(macro_graph, node_sizes)

        # Draw nodes with better styling
        nodes = nx.draw_networkx_nodes(
            macro_graph, pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.8,
            edgecolors='black',
            linewidths=2.5,
            ax=ax
        )

        # Get edge properties
        edge_widths, edge_colors, edge_styles = self._get_edge_properties(
            macro_graph)

        # Draw edges with better visibility
        if macro_graph.edges():
            edges = nx.draw_networkx_edges(
                macro_graph, pos,
                width=edge_widths,
                edge_color=edge_colors,
                alpha=0.7,
                arrows=True,
                arrowsize=30,
                arrowstyle='->',
                connectionstyle='arc3,rad=0.15',  # More curved for better separation
                ax=ax
            )

        # Add node labels with better positioning
        labels = {}
        for node in macro_graph.nodes():
            size = macro_graph.nodes[node].get('size', 0)
            labels[node] = f"C{node}\n({size})"

        nx.draw_networkx_labels(
            macro_graph, pos, labels,
            font_size=10,
            font_weight='bold',
            font_family='sans-serif',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
            ax=ax
        )

        # Add edge labels for citation counts (only for manageable number of edges)
        if len(macro_graph.edges()) <= 20:
            edge_labels = {}
            for u, v in macro_graph.edges():
                citations = macro_graph[u][v].get('citations', 0)
                if citations > 0:  # Only show non-zero citations
                    edge_labels[(u, v)] = f"{citations}"

            if edge_labels:
                nx.draw_networkx_edge_labels(
                    macro_graph, pos, edge_labels,
                    font_size=9,
                    font_weight='bold',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='lightyellow', alpha=0.9),
                    ax=ax
                )

        # Enhanced title
        threshold_info = f" (threshold={macro_graph.graph.get('threshold', 'N/A')})" if 'threshold' in macro_graph.graph else ""
        plt.title(
            f"{title}{threshold_info}\n{algorithm.title()} - {graph_type.title()} Macro-Graph",
            fontsize=18,
            fontweight='bold',
            pad=25,
            fontfamily='sans-serif'
        )

        ax.set_axis_off()

        # Enhanced legend
        legend_elements = [
            mpatches.Patch(facecolor='lightblue', edgecolor='black',
                           label='Community Node (size ∝ # papers)', alpha=0.8),
            plt.Line2D([0], [0], color='navy', lw=4,
                       label='Inter-community Citations (thickness ∝ count)'),
        ]

        plt.legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(0.02, 0.98),
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=12
        )

        # Enhanced statistics
        total_citations = sum(macro_graph[u][v].get(
            'citations', 0) for u, v in macro_graph.edges())
        community_sizes = [macro_graph.nodes[n].get(
            'size', 0) for n in macro_graph.nodes()]

        stats_text = (
            f"Communities: {len(macro_graph.nodes())}\n"
            f"Inter-community Edges: {len(macro_graph.edges())}\n"
            f"Total Citations: {total_citations}\n"
            f"Avg. Community Size: {np.mean(community_sizes):.1f}\n"
            f"Largest Community: {max(community_sizes) if community_sizes else 0}\n"
            f"Graph Density: {len(macro_graph.edges()) / (len(macro_graph.nodes()) * (len(macro_graph.nodes()) - 1)):.3f}"
            if len(macro_graph.nodes()) > 1 else "Graph Density: N/A"
        )

        plt.text(
            0.98, 0.02, stats_text,
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='lightgray',
                alpha=0.95,
                edgecolor='black'
            ),
            fontsize=11,
            fontfamily='monospace'
        )

        # Adjust layout and margins
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05)

        # Save high-quality plot
        filename = f"{algorithm}_{graph_type}_macro_graph.png"
        plot_path = output_path / filename
        plt.savefig(
            plot_path,
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            pad_inches=0.2
        )
        plt.close()

        return plot_path

    def generate_all_macro_visualizations(self, communities_dict: Dict[str, List[Set]],
                                          output_path: Path,
                                          threshold: int = None) -> Dict[str, Dict[str, Path]]:
        """Generate all macro-graph visualizations for multiple algorithms."""
        # Create graphs subdirectory
        graphs_dir = output_path / "macro_graphs"
        graphs_dir.mkdir(exist_ok=True)

        visualization_paths = {}

        for algorithm, communities in communities_dict.items():
            print(
                f"\nProcessing {algorithm} with {len(communities)} communities...")
            algorithm_paths = {}

            # Create weighted macro-graph
            weighted_macro = self.create_macro_graph(
                communities, weighted=True)
            weighted_path = self.visualize_macro_graph(
                weighted_macro,
                f"Weighted Citation Network",
                graphs_dir,
                algorithm,
                "weighted"
            )
            algorithm_paths["weighted"] = weighted_path

            # Create thresholded macro-graph (with adaptive threshold if not specified)
            thresholded_macro = self.create_thresholded_macro_graph(
                communities, threshold)
            actual_threshold = thresholded_macro.graph.get(
                'threshold', threshold)
            thresholded_path = self.visualize_macro_graph(
                thresholded_macro,
                f"Thresholded Citation Network",
                graphs_dir,
                algorithm,
                "thresholded"
            )
            algorithm_paths["thresholded"] = thresholded_path

            visualization_paths[algorithm] = algorithm_paths

        return visualization_paths

    def create_comparative_visualization(self, communities_dict: Dict[str, List[Set]],
                                         output_path: Path, threshold: int = None) -> Path:
        """Create a comparative visualization of all algorithms side by side."""
        n_algorithms = len(communities_dict)

        # Dynamic subplot arrangement
        if n_algorithms <= 2:
            rows, cols = 1, n_algorithms
            figsize = (10 * cols, 8)
        elif n_algorithms <= 4:
            rows, cols = 2, 2
            figsize = (16, 12)
        else:
            rows = (n_algorithms + 2) // 3
            cols = 3
            figsize = (18, 6 * rows)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_algorithms == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()

        algorithms = list(communities_dict.keys())

        for i, algorithm in enumerate(algorithms):
            if i >= len(axes):
                break

            communities = communities_dict[algorithm]
            macro_graph = self.create_macro_graph(communities, weighted=True)

            # Use simplified layout for comparative view
            node_sizes = [macro_graph.nodes[node].get(
                'size', 1) * 50 for node in macro_graph.nodes()]
            pos = self._get_optimized_layout(macro_graph, node_sizes)

            node_colors = plt.cm.Set3(
                np.linspace(0, 1, len(macro_graph.nodes())))

            nx.draw_networkx_nodes(macro_graph, pos, node_size=node_sizes,
                                   node_color=node_colors, alpha=0.8,
                                   edgecolors='black', linewidths=1.5, ax=axes[i])

            if macro_graph.edges():
                nx.draw_networkx_edges(macro_graph, pos, width=2, alpha=0.6,
                                       edge_color='steelblue', arrows=True,
                                       arrowsize=20, ax=axes[i])

            # Simplified labels
            simple_labels = {node: f"C{node}" for node in macro_graph.nodes()}
            nx.draw_networkx_labels(macro_graph, pos, simple_labels,
                                    font_size=8, font_weight='bold', ax=axes[i])

            axes[i].set_title(f"{algorithm}\n{len(macro_graph.nodes())} communities, "
                              f"{len(macro_graph.edges())} edges",
                              fontsize=12, fontweight='bold')
            axes[i].set_axis_off()

        # Hide extra subplots
        for i in range(len(algorithms), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle("Comparative Community Macro-Graph Analysis",
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()

        comp_path = output_path / "comparative_macro_graphs.png"
        plt.savefig(comp_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        return comp_path


class SimpleKeywordExtractor:
    """Simple class to extract most frequent words from node titles."""

    def __init__(self):
        # Basic stop words to filter out
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'
        }

    def extract_keywords(self, titles: List[str], top_k: int = 20) -> List[Tuple[str, int]]:
        """
        Extract most frequent words from titles.

        Args:
            titles: List of node titles
            top_k: Number of top keywords to return

        Returns:
            List of tuples (word, count) sorted by frequency
        """
        if not titles:
            return []

        # Combine all titles and make lowercase
        text = ' '.join(titles).lower()

        # Extract words (letters only, min 3 characters)
        words = re.findall(r'\b[a-z]{3,}\b', text)

        # Remove stop words
        filtered_words = [
            word for word in words if word not in self.stop_words]

        # Count frequencies and return top k
        word_counts = Counter(filtered_words)
        return word_counts.most_common(top_k)

    def extract_community_keywords(self, communities: List[Set], graph, top_k: int = 10) -> Dict[int, Dict]:
        """
        Extract keywords for each community from node titles.

        Args:
            communities: List of communities (each is a set of node IDs)
            graph: NetworkX graph containing node titles
            top_k: Number of top keywords to return per community

        Returns:
            Dictionary mapping community index to keyword data
        """
        community_keywords = {}

        for idx, community_set in enumerate(communities):
            # Get titles of all nodes in this community
            titles = []
            for node_id in community_set:
                if graph.has_node(node_id):
                    # Adjust this based on your actual node attribute structure
                    title = graph.nodes[node_id].get('title', '')
                    if title:
                        titles.append(title)

            # Extract keywords for this community
            keywords = self.extract_keywords(titles, top_k=top_k)

            community_keywords[idx] = {
                'keywords': keywords,
                'node_count': len(community_set),
                'titles_processed': len(titles)
            }

        return community_keywords
