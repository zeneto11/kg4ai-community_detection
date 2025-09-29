from pathlib import Path
from typing import Any, Dict, List, Set

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


class CommunityEvaluator:
    """Evaluate communities with multiple metrics and generate reports."""

    def __init__(self, graph: nx.Graph):
        self.graph = graph

    # ===================== Metrics =====================
    def homophily(self, community_map: Dict[Any, int], attribute: str = None) -> float:
        """Fraction of edges that connect nodes within the same community or sharing the same attribute."""
        total_edges, same_group_edges = 0, 0
        for u, v in self.graph.edges():
            if u not in community_map or v not in community_map:
                continue
            total_edges += 1
            if attribute:
                if self.graph.nodes[u].get(attribute) == self.graph.nodes[v].get(attribute):
                    same_group_edges += 1
            else:
                if community_map[u] == community_map[v]:
                    same_group_edges += 1
        return same_group_edges / total_edges if total_edges > 0 else 0.0

    def modularity(self, communities: List[Set]) -> float:
        """Modularity of the partitioning of the graph into communities."""
        return nx.algorithms.community.quality.modularity(self.graph, communities)

    def conductance(self, community: Set) -> float:
        """Conductance of a community: fraction of total edge volume that points outside the community."""
        internal_edges, external_edges = 0, 0
        for u in community:
            for v in self.graph.neighbors(u):
                weight = self.graph.get_edge_data(u, v).get('weight', 1.0)
                if v in community:
                    internal_edges += weight
                else:
                    external_edges += weight

        # Divide internal edges by 2 since each edge was counted twice
        internal_edges = internal_edges / 2

        # Avoid division by zero
        if internal_edges + external_edges == 0:
            return 1.0
        return external_edges / (2 * internal_edges + external_edges)

    def triad_participation_ratio(self, community: Set) -> float:
        """Fraction of nodes in the community that participate in at least one triangle."""
        subgraph = self.graph.subgraph(community).to_undirected()
        triangle_counts = nx.triangles(subgraph)
        participating_nodes = sum(
            1 for _, c in triangle_counts.items() if c > 0)
        return participating_nodes / len(community) if community else 0.0

    def clustering_coefficient(self, community: Set) -> float:
        """Average clustering coefficient of nodes in the community."""
        return nx.average_clustering(self.graph.subgraph(community))

    # ===================== Helpers =====================

    @staticmethod
    def report_community_size_distribution(communities: List[Set]) -> Dict[str, Any]:
        sizes = [len(comm) for comm in communities]
        return {
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
        }

    @staticmethod
    def filter_small_communities(communities: List[Set], min_size: int = 3) -> List[Set]:
        return [comm for comm in communities if len(comm) >= min_size]

    # ===================== Evaluation =====================
    def evaluate(self, communities: List[Set]) -> Dict[str, Any]:
        size_stats = self.report_community_size_distribution(communities)

        # Ensure all nodes are assigned
        assigned_nodes = set().union(*communities)
        unassigned_nodes = set(self.graph.nodes()) - assigned_nodes
        if unassigned_nodes:
            communities.append(unassigned_nodes)

        # Build community map
        community_map = {
            node: i for i, comm in enumerate(communities) for node in comm
        }

        results = {
            "modularity": self.modularity(communities),
            "homophily": self.homophily(community_map),
            "community_size_stats": size_stats,
        }

        # Weighted averages
        conductance_values, tpr_values, clustering_values = [], [], []
        for comm in communities:
            size = len(comm)
            if size < 2:
                continue
            conductance_values.append((self.conductance(comm), size))
            tpr_values.append((self.triad_participation_ratio(comm), size))
            clustering_values.append((self.clustering_coefficient(comm), size))

        if conductance_values:
            total = sum(w for _, w in conductance_values)
            results["avg_conductance"] = sum(
                v * w for v, w in conductance_values) / total
        if tpr_values:
            total = sum(w for _, w in tpr_values)
            results["avg_tpr"] = sum(v * w for v, w in tpr_values) / total
        if clustering_values:
            total = sum(w for _, w in clustering_values)
            results["avg_clustering"] = sum(
                v * w for v, w in clustering_values) / total

        return results

    # ===================== Export & Visualization =====================
    @staticmethod
    def export_to_csv(community_results: Dict, output_path: Path) -> Path:
        exclude_keys = ['run_id', 'timestamp', 'graph_info']
        titles_list = [k for k in community_results if k not in exclude_keys]
        data = []

        for algorithm in titles_list:
            algo_data = community_results[algorithm]
            metrics = algo_data['metrics']
            data.append({
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
                'num_small_communities': metrics.get('community_size_stats', {}).get('num_small', 0),
            })

        df = pd.DataFrame(data)
        csv_path = output_path / \
            f"{community_results.get('run_id', 'unknown')}_comparison.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    @staticmethod
    def plot_size_distribution(
        communities_dict: Dict[str, List[Set]],
        output_path: Path,
        logy: bool = False,
        max_labels: int = 30
    ) -> Dict[str, Path]:
        graphics_dir = output_path / "plots"
        graphics_dir.mkdir(exist_ok=True)
        saved_plots = {}
        plt.style.use('default')

        for algorithm, communities in communities_dict.items():
            sizes = [len(comm) for comm in communities]
            if not sizes:
                continue

            ids = list(range(len(sizes)))
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(ids, sizes, edgecolor='black')
            if logy:
                ax.set_yscale('log')

            ax.set_xlabel('Community ID')
            ax.set_ylabel('Number of Nodes')
            ax.set_title(f'{algorithm.title()} – Community Sizes')

            if len(ids) > max_labels:
                step = max(1, len(ids) // max_labels)
                ax.set_xticks(ids[::step])
                ax.set_xticklabels(ids[::step], rotation=45, ha='right')
            else:
                ax.set_xticks(ids)
                ax.set_xticklabels(ids, rotation=45, ha='right')

            ax.grid(True, axis='y', alpha=0.3)
            stats_text = (f'Total: {len(sizes)}\n'
                          f'Min: {min(sizes)}\n'
                          f'Max: {max(sizes)}\n'
                          f'Mean: {np.mean(sizes):.1f}\n'
                          f'Median: {np.median(sizes):.1f}')
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                    va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=10)

            fig.tight_layout()
            plot_path = graphics_dir / \
                f"{algorithm}_community_size_barplot.png"
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_plots[algorithm] = plot_path

        return saved_plots
