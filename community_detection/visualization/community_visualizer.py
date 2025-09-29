import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm, colors

warnings.filterwarnings('ignore')


class MacroGraphBuilder:
    def __init__(self, G, communities_dict):
        """
        Parameters
        ----------
        G : nx.DiGraph
            The original graph.
        communities_dict : dict
            Dictionary of algorithms -> list of communities (each community is a list of node ids).
        """
        self.G = G
        self.communities_dict = communities_dict

    def build_macrograph(self, algorithm, max_communities=100):
        """
        Build macrographs for a given community detection algorithm.

        Parameters
        ----------
        algorithm : str
            One of the keys in self.communities_dict, e.g. 'infomap', 'louvain'.
        max_communities : int or None
            If set, keep only the top-N largest communities (by size).

        Returns
        -------
        tuple of (nx.DiGraph, nx.DiGraph)
            - Weighted macrograph: edges carry weights (# of inter-community edges).
            - Thresholded macrograph: edges exist only if weight >= mean weight; edges are unweighted.
        """
        communities = self.communities_dict[algorithm]

        # Sort communities by size (descending)
        sorted_comms = sorted(
            enumerate(communities),
            key=lambda x: len(x[1]),
            reverse=True
        )

        # Keep only the top-N
        if max_communities is not None:
            sorted_comms = sorted_comms[:max_communities]

        # Map node -> community id (only for kept)
        node_to_comm = {}
        for comm_id, members in sorted_comms:
            for node in members:
                node_to_comm[node] = comm_id

        # Initialize macrograph
        M = nx.DiGraph()

        # Add nodes with original community IDs
        for comm_id, members in sorted_comms:
            M.add_node(comm_id, size=len(members))

        # Count edges between communities
        edge_weights = defaultdict(int)
        for u, v in self.G.edges():
            if u in node_to_comm and v in node_to_comm:
                cu, cv = node_to_comm[u], node_to_comm[v]
                if cu != cv:
                    edge_weights[(cu, cv)] += 1

        # == WEIGHTED GRAPH ==
        M_weighted = M.copy()
        for (cu, cv), weight in edge_weights.items():
            M_weighted.add_edge(cu, cv, weight=weight)

        # == THRESHOLDED GRAPH ==
        M_thresholded = M.copy()
        if not edge_weights:
            return M  # no edges at all

        # Compute threshold = mean weight
        threshold = sum(edge_weights.values()) / len(edge_weights)
        for (cu, cv), weight in edge_weights.items():
            if weight >= threshold:
                M_thresholded.add_edge(cu, cv)  # unweighted edge
        # Store threshold for reference
        M_thresholded.graph["threshold"] = threshold

        return M_weighted, M_thresholded


class MacroGraphVisualizer:
    def __init__(self, weighted_graph, thresholded_graph, algorithm_name, output_dir):
        """
        Parameters
        ----------
        weighted_graph : nx.DiGraph
            Macrograph with weighted edges.
        thresholded_graph : nx.DiGraph
            Macrograph with thresholded (unweighted) edges.
        algorithm_name : str
            The name of the community detection algorithm (e.g. "Infomap").
        output_dir : str
            Directory where plots will be saved.
        """
        self.weighted_graph = weighted_graph
        self.thresholded_graph = thresholded_graph
        self.algorithm_name = algorithm_name

        graphics_dir = Path(output_dir) / "plots"
        graphics_dir.mkdir(exist_ok=True)

        self.output_dir = graphics_dir

    def _compute_stats(self, macro_graph, thresholded=False):
        """Compute text block with graph stats."""
        community_sizes = [d["size"] for _, d in macro_graph.nodes(data=True)]

        stats_lines = []

        if thresholded:
            threshold_val = macro_graph.graph.get('threshold', 'N/A')
            stats_lines.append(f"{'Threshold:':<12} {threshold_val:>8.1f}")

        # Format all stats with consistent spacing
        stats_data = [
            ("Communities:", len(macro_graph.nodes())),
            ("Inter-edges:", len(macro_graph.edges())),
            ("Avg Size:", np.mean(community_sizes) if community_sizes else 0),
            ("Largest:", max(community_sizes) if community_sizes else 0),
            ("Smallest:", min(community_sizes) if community_sizes else 0),
        ]

        for label, value in stats_data:
            if isinstance(value, float):
                formatted_value = f"{value:>8.1f}"
            else:
                formatted_value = f"{value:>8}"
            stats_lines.append(f"{label:<12} {formatted_value}")

        # Add density
        if len(macro_graph.nodes()) > 1:
            density = len(macro_graph.edges()) / \
                (len(macro_graph.nodes()) * (len(macro_graph.nodes()) - 1))
            stats_lines.append(f"{'Density:':<12} {density:>8.3f}")
        else:
            stats_lines.append(f"{'Density:':<12} {'N/A':>8}")

        return "\n".join(stats_lines)

    def _get_optimal_layout(self, macro_graph):
        """Choose the best layout algorithm based on graph properties."""
        n_nodes = len(macro_graph.nodes())

        if n_nodes <= 50:
            # Small graphs: circular layout
            return nx.circular_layout(macro_graph)
        elif n_nodes <= 70:
            # Medium graphs: use spring layout with good parameters
            return nx.spring_layout(
                macro_graph,
                k=3/np.sqrt(n_nodes),  # Optimal node spacing
                iterations=100,
                seed=42
            )
        else:
            # Large graphs: use force-directed with clustering
            try:
                return nx.kamada_kawai_layout(macro_graph)
            except:
                # Fallback to spring layout
                return nx.spring_layout(
                    macro_graph,
                    k=2/np.sqrt(n_nodes),
                    iterations=50,
                    seed=42
                )

    def _get_node_colors(self, macro_graph):
        """Generate visually distinct colors for nodes."""
        n_nodes = len(macro_graph.nodes())

        if n_nodes <= 8:
            # Use categorical colors for small graphs
            cmap = plt.cm.Set3
            colors = [cmap(i % 12) for i in range(n_nodes)]
        elif n_nodes <= 20:
            # Use qualitative colormap
            cmap = plt.cm.tab20
            colors = [cmap(i % 20) for i in range(n_nodes)]
        else:
            # Use continuous colormap based on community size
            sizes = [d["size"] for _, d in macro_graph.nodes(data=True)]
            cmap = plt.cm.viridis
            norm = plt.Normalize(min(sizes), max(sizes))
            colors = [cmap(norm(size)) for size in sizes]

        return colors

    def _plot_graph(self, macro_graph, title, filename, thresholded=False):
        """Helper to plot and save a macrograph with stats legend."""
        plt.figure(figsize=(12, 9))

        # Improved node positioning
        pos = self._get_optimal_layout(macro_graph)

        # Get node colors
        node_colors = self._get_node_colors(macro_graph)

        # Calculate node sizes with better scaling
        base_sizes = [d["size"] for _, d in macro_graph.nodes(data=True)]
        if base_sizes:
            min_size, max_size = min(base_sizes), max(base_sizes)
            if min_size == max_size:
                node_sizes = [300] * len(base_sizes)  # All same size
            else:
                # Logarithmic scaling with minimum and maximum bounds
                log_sizes = np.log1p(base_sizes)
                node_sizes = 100 + 900 * \
                    (log_sizes - min(log_sizes)) / \
                    (max(log_sizes) - min(log_sizes))
        else:
            node_sizes = [300] * len(macro_graph.nodes())

        # Draw nodes with improved styling
        nx.draw_networkx_nodes(
            macro_graph, pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.9,
            edgecolors='black',
            linewidths=0.8
        )

        # Improved edge drawing
        if thresholded:
            # For thresholded graphs: uniform edges
            nx.draw_networkx_edges(
                macro_graph, pos,
                edge_color='gray',
                alpha=0.6,
                arrows=True,
                arrowsize=8,
                arrowstyle='->',
                width=1
            )
        else:
            # For weighted graphs: vary width and color by weight
            weights = [d["weight"] for _, _, d in macro_graph.edges(data=True)]
            if weights:
                min_weight, max_weight = min(weights), max(weights)
                if min_weight == max_weight:
                    edge_widths = [2.0] * len(weights)
                    edge_alphas = [0.7] * len(weights)
                else:
                    # Normalize widths and alphas
                    edge_widths = [
                        1.0 + 3.0 * (w - min_weight) / (max_weight - min_weight) for w in weights]
                    edge_alphas = [
                        0.3 + 0.5 * (w - min_weight) / (max_weight - min_weight) for w in weights]

                # Use colormap for edge weights
                # edge_colors = plt.cm.Reds([(w - min_weight) / (max_weight - min_weight) for w in weights])
                cmap = cm.Reds
                norm = colors.Normalize(vmin=min_weight, vmax=max_weight)

                # Set a minimum normalized value so low weights aren't washed out
                min_color_val = 0.2
                edge_colors = [
                    cmap(min_color_val + (1 - min_color_val) * norm(w)) for w in weights]

                for (u, v, d), width, alpha, color in zip(macro_graph.edges(data=True), edge_widths, edge_alphas, edge_colors):
                    nx.draw_networkx_edges(
                        macro_graph, pos,
                        edgelist=[(u, v)],
                        width=width,
                        alpha=alpha,
                        edge_color=[color],
                        arrows=True,
                        arrowsize=10,
                        arrowstyle='->',
                        connectionstyle='arc3,rad=0.1'  # Curved edges to avoid overlap
                    )
            else:
                # Fallback if no weights
                nx.draw_networkx_edges(
                    macro_graph, pos,
                    edge_color='gray',
                    alpha=0.5,
                    arrows=True,
                    arrowsize=15
                )

        # Improved labels
        labels = {node: f"C{node}\n({macro_graph.nodes[node]['size']})"
                  for node in macro_graph.nodes()}

        text_items = nx.draw_networkx_labels(
            macro_graph, pos,
            labels=labels,
            font_size=7,
            font_weight='bold',
            font_family='sans-serif',
            font_color='white'   # main text color
        )

        # Add black outline to each label
        for _, t in text_items.items():
            t.set_path_effects(
                [pe.withStroke(linewidth=2, foreground="black")])

        # Stats box
        stats_text = self._compute_stats(macro_graph, thresholded)
        plt.gcf().text(0.028, 0.035,
                       stats_text,
                       fontsize=10,
                       fontfamily='DejaVu Sans Mono',
                       ha="left", va="bottom",
                       bbox=dict(
                           boxstyle='round,pad=0.7',
                           facecolor='lightgray',
                           alpha=0.95,
                           edgecolor='black')
                       )

        # Improved title
        plt.title(title, fontsize=16, fontweight='bold', pad=20)

        # Remove axes for cleaner look
        plt.axis('off')

        # Add legend for weighted graphs
        if not thresholded and weights and len(set(weights)) > 1:
            self._add_edge_weight_legend(weights)

        # Save with tight layout
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{filename}",
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _add_edge_weight_legend(self, weights):
        """Add a legend showing edge weight scale for weighted graphs."""
        from matplotlib.lines import Line2D

        min_weight, max_weight = min(weights), max(weights)

        # Create legend elements
        legend_elements = []
        if max_weight > min_weight:
            # Sample a few weight values for legend
            sample_weights = np.linspace(min_weight, max_weight, 4)
            for w in sample_weights:
                width = 1.0 + 3.0 * (w - min_weight) / \
                    (max_weight - min_weight)
                legend_elements.append(
                    Line2D([0], [0],
                           color='red',
                           linewidth=width,
                           label=f'{w:.1f}')
                )

            plt.legend(handles=legend_elements,
                       title='Edge Weights',
                       loc='upper right',
                       fontsize=8,
                       title_fontsize=9)

    def plot_both(self):
        """Plot and save both weighted and thresholded macrographs."""

        self._plot_graph(
            self.weighted_graph,
            f"{self.algorithm_name} - Weighted Macro-Graph",
            f"{self.algorithm_name.lower()}_macrograph_weighted.png",
            thresholded=False
        )
        self._plot_graph(
            self.thresholded_graph,
            f"{self.algorithm_name} - Thresholded Macro-Graph",
            f"{self.algorithm_name.lower()}_macrograph_thresholded.png",
            thresholded=True
        )
