# src/community_detection/leiden_detector.py

import logging
from typing import Union

import igraph as ig
import leidenalg as la
import networkx as nx

from community_detection.utils.graph_utility import \
    convert_to_weighted_undirected

from .base import CommunityDetector


class LeidenDetector(CommunityDetector):
    """Community detection using Leiden algorithm.

    The Leiden method is an improvement over the Louvain method, providing better
    partition quality and faster convergence.

    Args:
        resolution: Resolution parameter for the Leiden algorithm. Higher values lead to
                    more, smaller communities. Default is None, which uses the algorithm's default.
        objective: Objective function to optimize. Options include 'modularity', 'cpm',
                    'surprise', and 'significance'. Default is 'modularity'.
        seed: Random seed for reproducibility. Default is 42.
    """

    def __init__(self, resolution: float = None, objective: str = 'modularity', seed: int = 42, **kwargs):
        """Initialize the Louvain detector."""
        super().__init__(resolution=resolution, objective=objective, seed=seed, **kwargs)
        self.resolution = resolution
        self.objective = objective
        self.seed = seed

    def fit(self, graph: Union[nx.Graph, nx.DiGraph]) -> 'LeidenDetector':
        """Detect communities using Leiden method.

        Args:
            graph: NetworkX graph object

        Returns:
            self: The fitted detector
        """

        # Convert directed graph to undirected if necessary
        if graph.is_directed():
            graph = convert_to_weighted_undirected(graph)

        # Convert to igraph (handles both directed and undirected)
        if isinstance(graph, nx.DiGraph):
            g = ig.Graph.from_networkx(graph)
            # Convert to undirected for Leiden (collapse bidirectional edges)
            g = g.as_undirected(mode='collapse', combine_edges='sum')
        else:
            g = ig.Graph.from_networkx(graph)

        # Run Louvain based on objective
        if self.objective == 'modularity':
            if self.resolution is not None:
                partition = la.RBConfigurationVertexPartition(
                    g, resolution_parameter=self.resolution)
            else:
                partition = la.ModularityVertexPartition(g)

        elif self.objective == 'cpm':
            # Good default for citation networks
            res = self.resolution if self.resolution is not None else 0.05
            partition = la.CPMVertexPartition(g, resolution_parameter=res)

        elif self.objective == 'surprise':
            partition = la.SurpriseVertexPartition(g)

        elif self.objective == 'significance':
            partition = la.SignificanceVertexPartition(g)

        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        # Run optimization
        optimizer = la.Optimiser()
        optimizer.set_rng_seed(self.seed)
        diff = optimizer.optimise_partition(partition)

        # Convert to a dict mapping for consistency
        community_map = {g.vs[i]['_nx_name']: partition.membership[i]
                         for i in range(g.vcount())}

        # Store results
        self._set_results(community_map=community_map)

        return self
