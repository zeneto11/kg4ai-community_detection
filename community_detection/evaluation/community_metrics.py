import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Set, Any, Union, Optional, Tuple


def conductance(graph: nx.Graph, community: Set) -> float:
    """Calculate conductance of a community.

    Conductance is the fraction of total edge weight that points outside the community.

    Args:
        graph: NetworkX graph
        community: Set of nodes in the community

    Returns:
        float: Conductance score (lower is better)
    """
    internal_edges = 0
    external_edges = 0

    for u in community:
        for v in graph.neighbors(u):
            if v in community:
                internal_edges += graph.get_edge_data(u, v).get('weight', 1.0)
            else:
                external_edges += graph.get_edge_data(u, v).get('weight', 1.0)

    if internal_edges + external_edges == 0:
        return 1.0  # Worst conductance for isolated communities

    return external_edges / (internal_edges + external_edges)


def modularity(graph: nx.Graph, communities: List[Set]) -> float:
    """Calculate modularity of a community partition.

    Modularity measures the strength of division into communities.

    Args:
        graph: NetworkX graph
        communities: List of sets of nodes representing communities

    Returns:
        float: Modularity score (higher is better)
    """
    # Convert communities to format expected by NetworkX
    community_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_map[node] = i

    return nx.algorithms.community.quality.modularity(graph, communities)


def triad_participation_ratio(graph: nx.Graph, community: Set) -> float:
    """Calculate the triad participation ratio (TPR) for a community.

    TPR is the fraction of nodes in a community that participate in at least one triangle.

    Args:
        graph: NetworkX graph
        community: Set of nodes in the community

    Returns:
        float: TPR score (higher means more triadic closure)
    """
    # Extract the subgraph for the community
    subgraph = graph.subgraph(community)

    # Convert to undirected for triangle counting
    subgraph = subgraph.to_undirected()

    # Count nodes that participate in at least one triangle
    participating_nodes = 0
    triangle_counts = nx.triangles(subgraph)

    for node, count in triangle_counts.items():
        if count > 0:
            participating_nodes += 1

    if len(community) == 0:
        return 0.0

    return participating_nodes / len(community)


def clustering_coefficient(graph: nx.Graph, community: Set) -> float:
    """Calculate the average clustering coefficient for a community.

    Args:
        graph: NetworkX graph
        community: Set of nodes in the community

    Returns:
        float: Average clustering coefficient (higher means more clustering)
    """
    # Extract the subgraph for the community
    subgraph = graph.subgraph(community)

    # Calculate average clustering coefficient
    return nx.average_clustering(subgraph)


def homophily(graph: nx.Graph, community_map: Dict[Any, int], attribute: str = None) -> float:
    """Calculate attribute homophily within communities.

    If attribute is provided, calculates attribute homophily.
    Otherwise, calculates structural homophily based on community assignments.

    Args:
        graph: NetworkX graph
        community_map: Dictionary mapping nodes to community IDs
        attribute: Node attribute to use for homophily calculation

    Returns:
        float: Homophily score (higher means more homogeneous communities)
    """
    total_edges = 0
    same_group_edges = 0

    for u, v in graph.edges():
        # Skip nodes not in community map
        if u not in community_map or v not in community_map:
            continue

        total_edges += 1

        if attribute is not None:
            # Attribute homophily
            if graph.nodes[u].get(attribute) == graph.nodes[v].get(attribute):
                same_group_edges += 1
        else:
            # Structural homophily
            if community_map[u] == community_map[v]:
                same_group_edges += 1

    if total_edges == 0:
        return 0.0

    return same_group_edges / total_edges


def report_community_size_distribution(communities: List[Set]) -> Dict[str, Any]:
    """Report statistics about community sizes before filtering."""
    sizes = [len(comm) for comm in communities]
    stats = {
        "num_communities": len(sizes),
        "min_size": min(sizes) if sizes else 0,
        "max_size": max(sizes) if sizes else 0,
        "mean_size": np.mean(sizes) if sizes else 0,
        "median_size": np.median(sizes) if sizes else 0,
        "num_singleton": sum(1 for s in sizes if s == 1),
        "num_size_2": sum(1 for s in sizes if s == 2),
        "num_size_3": sum(1 for s in sizes if s == 3),
        "num_small": sum(1 for s in sizes if s <= 3),
        "size_histogram": {int(k): int(v) for k, v in zip(*np.unique(sizes, return_counts=True))}
        # "size_histogram": dict(zip(*np.unique(sizes, return_counts=True)))
    }
    return stats


def filter_small_communities(communities: List[Set], min_size: int = 3) -> List[Set]:
    """Remove communities with fewer than min_size nodes (default: 3)."""
    return [comm for comm in communities if len(comm) >= min_size]


def evaluate_communities(graph: nx.Graph, communities: List[Set]) -> Dict[str, float]:
    """Evaluate communities using multiple metrics.

    Args:
        graph: NetworkX graph
        communities: List of sets of nodes representing communities

    Returns:
        Dict: Dictionary of evaluation metrics
    """
    # Report community size distribution before filtering
    size_stats = report_community_size_distribution(communities)

    # Filter small communities (optional, default min_size=3)
    # communities = filter_small_communities(communities, min_size=3)

    # Ensure all nodes are assigned to a community
    assigned_nodes = set()
    for comm in communities:
        assigned_nodes.update(comm)
    all_nodes = set(graph.nodes())
    unassigned_nodes = all_nodes - assigned_nodes
    if unassigned_nodes:
        # Add a miscellaneous community for unassigned nodes
        communities.append(unassigned_nodes)

    # Create community map for efficient lookups
    community_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_map[node] = i

    # Calculate global metrics
    results = {
        "modularity": modularity(graph, communities),
        "homophily": homophily(graph, community_map),
        "community_size_stats": size_stats
    }

    # Calculate per-community metrics and take weighted average
    conductance_values = []
    tpr_values = []
    clustering_values = []

    for community in communities:
        size = len(community)
        if size < 2:
            continue  # Skip singleton communities

    conductance_values.append((conductance(graph, community), size))
    tpr_values.append((triad_participation_ratio(graph, community), size))
    clustering_values.append(
        (clustering_coefficient(graph, community), size))

    # Compute weighted averages
    if conductance_values:
        total_weight = sum(weight for _, weight in conductance_values)
        results["avg_conductance"] = sum(
            value * weight for value, weight in conductance_values) / total_weight

    if tpr_values:
        total_weight = sum(weight for _, weight in tpr_values)
        results["avg_tpr"] = sum(
            value * weight for value, weight in tpr_values) / total_weight

    if clustering_values:
        total_weight = sum(weight for _, weight in clustering_values)
        results["avg_clustering"] = sum(
            value * weight for value, weight in clustering_values) / total_weight

    return results


def export_community_results_to_csv(community_results: Dict, output_path: Path) -> Path:
    """Export community detection results to CSV for easy comparison.

    Args:
        community_results: Dictionary containing results from multiple algorithms
        output_path: Directory where to save the CSV file

    Returns:
        Path: Path to the saved CSV file
    """
    # Extract data for CSV
    data = []

    # Get the list of algorithm titles by filtering the dictionary keys
    exclude_keys = ['run_id', 'timestamp', 'graph_info']
    titles_list = [key for key in community_results.keys()
                   if key not in exclude_keys]

    for algorithm in titles_list:
        if algorithm in community_results:
            algo_data = community_results[algorithm]
            metrics = algo_data['metrics']

            row = {
                'run_id': community_results.get('run_id', 'unknown'),
                'timestamp': community_results.get('timestamp', ''),
                'algorithm': algorithm.title(),
                'num_communities': algo_data['num_communities'],
                'modularity': metrics.get('modularity', 0),
                'homophily': metrics.get('homophily', 0),
                'avg_conductance': metrics.get('avg_conductance', 0),
                'avg_tpr': metrics.get('avg_tpr', 0),
                'avg_clustering': metrics.get('avg_clustering', 0),
                'min_community_size': metrics.get('community_size_stats', {}).get('min_size', 0),
                'max_community_size': metrics.get('community_size_stats', {}).get('max_size', 0),
                'mean_community_size': metrics.get('community_size_stats', {}).get('mean_size', 0),
                'median_community_size': metrics.get('community_size_stats', {}).get('median_size', 0),
                'num_singleton_communities': metrics.get('community_size_stats', {}).get('num_singleton', 0),
                'num_small_communities': metrics.get('community_size_stats', {}).get('num_small', 0)
            }
            data.append(row)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    csv_path = output_path / \
        f"{community_results.get('run_id', 'unknown')}_comparison.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


def generate_community_size_histograms(communities_dict: Dict[str, List[Set]],
                                       output_path: Path) -> Dict[str, Path]:
    """Generate histogram plots showing community size distributions.

    Args:
        communities_dict: Dictionary mapping algorithm names to their communities
        output_path: Directory where to save the histogram plots

    Returns:
        Dict: Mapping of algorithm names to saved plot paths
    """
    # Create graphs subdirectory
    graphs_dir = output_path / "graphs"
    graphs_dir.mkdir(exist_ok=True)

    saved_plots = {}

    # Set up the matplotlib style
    plt.style.use('default')

    for algorithm, communities in communities_dict.items():
        # Calculate community sizes
        sizes = [len(comm) for comm in communities]

        if not sizes:
            continue

        # Create the histogram
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create histogram with appropriate bins
        max_size = max(sizes)
        min_size = min(sizes)

        # Use logarithmic binning for better visualization if range is large
        if max_size > 50:
            bins = np.logspace(np.log10(min_size), np.log10(max_size), 20)
            ax.set_xscale('log')
        else:
            bins = range(min_size, max_size + 2)

        n, bins_used, patches = ax.hist(
            sizes, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')

        # Customize the plot
        ax.set_xlabel('Community Size (Number of Nodes)', fontsize=12)
        ax.set_ylabel('Number of Communities', fontsize=12)
        ax.set_title(f'{algorithm.title()} - Community Size Distribution\n'
                     f'Total Communities: {len(communities)}, Nodes Distribution', fontsize=14, fontweight='bold')

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

        # Add statistics text box
        stats_text = (f'Total Communities: {len(communities)}\n'
                      f'Min Size: {min_size}\n'
                      f'Max Size: {max_size}\n'
                      f'Mean Size: {np.mean(sizes):.1f}\n'
                      f'Median Size: {np.median(sizes):.1f}')

        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)

        # Save the plot
        plot_path = graphs_dir / f"{algorithm}_community_size_histogram.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close to free memory

        saved_plots[algorithm] = plot_path

    return saved_plots


def generate_community_size_rankplots(communities_dict: Dict[str, List[Set]],
                                      output_path: Path,
                                      logy: bool = False,
                                      label_top: int = 0) -> Dict[str, Path]:
    """
    Create size-vs-community rank plots.
    X-axis: community rank (1 = largest), Y-axis: community size (nodes).

    Args:
        communities_dict: {algorithm_name: [set(node_id), ...]}
        output_path: base directory to save plots
        logy: if True, use log scale on Y (useful for heavy tails)
        label_top: if >0, annotate the top N bars with their sizes

    Returns:
        {algorithm_name: saved_plot_path}
    """
    graphs_dir = output_path / "graphs"
    graphs_dir.mkdir(exist_ok=True)
    saved_plots = {}

    plt.style.use('default')

    for algorithm, communities in communities_dict.items():
        sizes = [len(comm) for comm in communities]
        if not sizes:
            continue

        sizes_sorted = sorted(sizes, reverse=True)
        k = len(sizes_sorted)
        x = np.arange(1, k + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x, sizes_sorted, edgecolor='black')  # no explicit facecolor

        if logy:
            ax.set_yscale('log')

        ax.set_xlabel('Communities (ranked by size)')
        ax.set_ylabel('Number of nodes')
        ax.set_title(f'{algorithm.title()} – Community sizes by rank')

        # Optional labels on the top N bars
        if label_top > 0:
            N = min(label_top, k)
            for i in range(N):
                ax.text(x[i], sizes_sorted[i],
                        f'{sizes_sorted[i]}',
                        ha='center', va='bottom', fontsize=8, rotation=0)

        # Grid for readability
        ax.grid(True, which='both', axis='y', alpha=0.3)

        # Stats box (same content/format as your histogram)
        stats_text = (f'Total Communities: {k}\n'
                      f'Min Size: {int(np.min(sizes_sorted))}\n'
                      f'Max Size: {int(np.max(sizes_sorted))}\n'
                      f'Mean Size: {np.mean(sizes_sorted):.1f}\n'
                      f'Median Size: {np.median(sizes_sorted):.1f}')
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)

        fig.tight_layout()
        plot_path = graphs_dir / f"{algorithm}_community_size_rankplot.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved_plots[algorithm] = plot_path

    return saved_plots


def generate_community_size_barplots(communities_dict: Dict[str, List[Set]],
                                     output_path: Path,
                                     logy: bool = False,
                                     max_labels: int = 30) -> Dict[str, Path]:
    """
    Create bar plots of community sizes.
    X-axis: community IDs, Y-axis: community size (nodes).

    Args:
        communities_dict: {algorithm_name: [set(node_id), ...]}
        output_path: base directory to save plots
        logy: if True, use log scale on Y (useful for heavy tails)
        max_labels: if too many communities, only show this many X labels

    Returns:
        {algorithm_name: saved_plot_path}
    """
    graphics_dir = output_path / "graphics"
    graphics_dir.mkdir(exist_ok=True)
    saved_plots = {}

    plt.style.use('default')

    for algorithm, communities in communities_dict.items():
        sizes = [len(comm) for comm in communities]
        if not sizes:
            continue

        community_ids = list(range(len(sizes)))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(community_ids, sizes, edgecolor='black')

        if logy:
            ax.set_yscale('log')

        ax.set_xlabel('Community ID')
        ax.set_ylabel('Number of Nodes')
        ax.set_title(f'{algorithm.title()} – Community Sizes')

        # Show fewer X labels if there are too many
        if len(community_ids) > max_labels:
            step = max(1, len(community_ids) // max_labels)
            ax.set_xticks(community_ids[::step])
            ax.set_xticklabels(community_ids[::step], rotation=45, ha='right')
        else:
            ax.set_xticks(community_ids)
            ax.set_xticklabels(community_ids, rotation=45, ha='right')

        # Add grid
        ax.grid(True, axis='y', alpha=0.3)

        # Stats box (same as before)
        stats_text = (f'Total Communities: {len(sizes)}\n'
                      f'Min Size: {min(sizes)}\n'
                      f'Max Size: {max(sizes)}\n'
                      f'Mean Size: {np.mean(sizes):.1f}\n'
                      f'Median Size: {np.median(sizes):.1f}')
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)

        fig.tight_layout()
        plot_path = graphics_dir / f"{algorithm}_community_size_barplot.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved_plots[algorithm] = plot_path

    return saved_plots
