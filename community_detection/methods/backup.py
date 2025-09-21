# src/community_detection/louvain_detector.py

from typing import Any, Dict, Optional, Union

import community as community_louvain
import networkx as nx

from .base import CommunityDetector


class InfomapDetector(CommunityDetector):
    """Community detection using Infomap algorithm.

    Infomap finds communities by optimizing the Map Equation, which exploits
    information-theoretic principles to reveal community structure in networks.

    Args:
        num_trials: Number of trials to run (more trials, better results)
        silent: Whether to silence output during computation
    """

    def __init__(self, num_trials: int = 10, silent: bool = True, **kwargs):
        """Initialize the Infomap detector."""
        super().__init__(num_trials=num_trials, silent=silent, **kwargs)
        self.num_trials = num_trials
        self.silent = silent

    def fit(self, graph: Union[nx.Graph, nx.DiGraph]) -> 'InfomapDetector':
        """Detect communities using Infomap.

        Args:
            graph: NetworkX graph object

        Returns:
            self: The fitted detector
        """
        try:
            from infomap import Infomap
        except ImportError:
            raise ImportError(
                "Infomap package is required. Install with 'pip install infomap'"
            )

        # Initialize Infomap
        if graph.is_directed():
            im = Infomap("--directed", silent=False,
                         num_trials=self.num_trials)
        else:
            im = Infomap(silent=False, num_trials=self.num_trials)

        # Add nodes and edges to Infomap network
        for i, node in enumerate(graph.nodes()):
            im.add_node(i, str(node))

        # Create a mapping from node to index
        node_to_idx = {node: i for i, node in enumerate(graph.nodes())}

        # Add edges to Infomap network
        if graph.is_directed():
            for source, target, data in graph.edges(data=True):
                weight = data.get('weight', 1.0)
                im.add_link(node_to_idx[source], node_to_idx[target], weight)
        else:
            for source, target, data in graph.edges(data=True):
                weight = data.get('weight', 1.0)
                im.add_link(node_to_idx[source], node_to_idx[target], weight)

        # Run Infomap
        im.run()

        # Process results
        idx_to_node = {i: node for node, i in node_to_idx.items()}
        community_map = {}

        for node in im.tree:
            if node.is_leaf:
                original_node = idx_to_node[node.node_id]
                community_map[original_node] = node.module_id

        # Store results
        self._set_results(community_map=community_map)

        return self


class LouvainDetector(CommunityDetector):
    """Community detection using Louvain algorithm.

    The Louvain method is a greedy optimization method that attempts to optimize
    the modularity of a partition of the network.

    Args:
        resolution: Resolution parameter for modularity (higher values = smaller communities)
        random_state: Random seed for reproducibility
    """

    def __init__(self, resolution: float = 1.0, random_state: int = None, **kwargs):
        """Initialize the Louvain detector."""
        super().__init__(resolution=resolution, random_state=random_state, **kwargs)
        self.resolution = resolution
        self.random_state = random_state

    def fit(self, graph: Union[nx.Graph, nx.DiGraph]) -> 'LouvainDetector':
        """Detect communities using Louvain method.

        Args:
            graph: NetworkX graph object

        Returns:
            self: The fitted detector
        """
        # try:
        #     import community as community_louvain
        # except ImportError:
        #     try:
        #         from cdlib import algorithms as cdlib_algorithms
        #         use_cdlib = True
        #     except ImportError:
        #         raise ImportError(
        #             "Either python-louvain or cdlib package is required. "
        #             "Install with 'pip install python-louvain' or 'pip install cdlib'"
        #         )

        # Convert directed graph to undirected if necessary
        if graph.is_directed():
            graph = nx.Graph(graph)

        # Run Louvain
        # if 'community_louvain' in locals():
        partition = community_louvain.best_partition(
            graph,
            resolution=self.resolution,
            random_state=self.random_state,
            **{k: v for k, v in self.params.items() if k not in ['resolution', 'random_state']}
        )
        community_map = partition
        # else:
        #     # Use cdlib as fallback
        #     communities = cdlib_algorithms.louvain(
        #         graph,
        #         resolution=self.resolution,
        #         random_state=self.random_state
        #     )
        #     community_map = {node: comm_id for comm_id, nodes in enumerate(communities.communities)
        #                      for node in nodes}

        # Store results
        self._set_results(community_map=community_map)

        return self
