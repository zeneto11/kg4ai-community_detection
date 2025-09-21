# src/community_detection/louvain_detector.py

from collections import Counter
import logging
import networkx as nx
from typing import Union, Dict, Any

from community_detection.utils.graph_utility import convert_to_weighted_undirected
from .base import CommunityDetector
import community as community_louvain


class LouvainDetector(CommunityDetector):
    """Community detection using Louvain algorithm.

    The Louvain method is a greedy optimization method that attempts to optimize
    the modularity of a partition of the network.

    Args:
        resolution: Resolution parameter for modularity (higher values = smaller communities)
        random_state: Random seed for reproducibility
    """

    def __init__(self, resolution: float = 1.0, random_state: int = 42, **kwargs):
        """Initialize the Louvain detector."""
        super().__init__(resolution=resolution, random_state=random_state, **kwargs)
        self.resolution = resolution
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

    def fit(self, graph: Union[nx.Graph, nx.DiGraph], threshold=1e-07) -> 'LouvainDetector':
        """Detect communities using Louvain method.

        Args:
            graph: NetworkX graph object

        Returns:
            self: The fitted detector
        """

        # Convert directed graph to undirected if necessary
        if graph.is_directed():
            graph = convert_to_weighted_undirected(graph)

        # Run Louvain
        communities = nx.community.louvain_communities(
            graph,
            resolution=self.resolution,
            threshold=threshold,
            seed=self.random_state
        )

        # Convert to a dict mapping for consistency
        community_map = {node: comm for comm,
                         nodes in enumerate(communities) for node in nodes}

        # Store results
        self._set_results(community_map=community_map)

        return self
