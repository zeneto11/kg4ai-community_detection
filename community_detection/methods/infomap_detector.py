# src/community_detection/infomap_detector.py

import networkx as nx
from typing import Union, Optional
from .base import CommunityDetector


class InfomapDetector(CommunityDetector):
    """Community detection using Infomap algorithm.

    Infomap finds communities by optimizing the Map Equation, which exploits
    information-theoretic principles to reveal community structure in networks.

    Args:
        num_trials: Number of trials to run (more trials, better results)
        teleportation_prob: Teleportation probability for random walks
        seed: Random seed for reproducibility
        silent: Whether to silence output during computation
    """

    def __init__(self, num_trials: int = 10, teleportation_prob: float = 0.1, seed: int = 42, silent: bool = True,  **kwargs):
        """Initialize the Infomap detector."""
        super().__init__(num_trials=num_trials, teleportation_prob=teleportation_prob,
                         seed=seed, silent=silent, **kwargs)
        self.num_trials = num_trials
        self.teleportation_prob = teleportation_prob
        self.seed = seed
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

        # Configure Infomap
        flags = []
        if graph.is_directed():
            flags.append("--directed")
        has_self_loops = any(graph.has_edge(n, n) for n in graph.nodes())
        if has_self_loops:
            flags.append("--include-self-links")

        flags.extend([
            f"--num-trials {self.num_trials}",
            f"--teleportation-probability {self.teleportation_prob}",
            f"--seed {self.seed}",
            "--silent" if self.silent else "--verbose"
        ])

        # Initialize Infomap with flags
        flag_string = " ".join(flags)
        im = Infomap(flag_string)

        # Add nodes and create mapping
        node_to_idx = {}
        for i, node in enumerate(graph.nodes()):
            node_to_idx[node] = i
            im.add_node(i, str(node))

        # Add edges to Infomap network
        for source, target, data in graph.edges(data=True):
            weight = data.get('weight', 1.0)
            im.add_link(node_to_idx[source], node_to_idx[target], weight)

        # Run Infomap
        im.run()

        # Process results
        idx_to_node = {i: node for node, i in node_to_idx.items()}
        partition = {node.node_id: node.module_id for node in im.nodes}

        # Map back from integer IDs to original node names
        community_map = {
            idx_to_node[idx]: community_id for idx, community_id in partition.items()}

        # Store results
        self._set_results(community_map=community_map)

        return self
