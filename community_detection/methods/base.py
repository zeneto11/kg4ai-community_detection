# src/community_detection/base.py

import abc
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple, Set


class CommunityDetector(abc.ABC):
    """Base class for community detection algorithms.

    All community detection algorithms should inherit from this class
    and implement the required methods.
    """

    def __init__(self, **kwargs):
        """Initialize the detector with algorithm-specific parameters."""
        self.communities = None
        self.community_map = None
        self.params = kwargs

    @abc.abstractmethod
    def fit(self, graph: Union[nx.Graph, nx.DiGraph]) -> 'CommunityDetector':
        """Detect communities in the provided graph.

        Args:
            graph: NetworkX graph object

        Returns:
            self: The fitted detector
        """
        pass

    def get_communities(self) -> List[Set[Any]]:
        """Return communities as list of sets of nodes.

        Returns:
            List of sets, where each set contains nodes in a community
        """
        if self.communities is None:
            raise ValueError("Must call fit() before getting communities")
        return self.communities

    def get_community_map(self) -> Dict[Any, int]:
        """Return a mapping from node to community ID.

        Returns:
            Dict mapping each node to its community ID
        """
        if self.community_map is None:
            raise ValueError("Must call fit() before getting community map")
        return self.community_map

    def _set_results(self, communities=None, community_map=None):
        """Store community detection results."""
        self.communities = communities
        self.community_map = community_map

        # If only communities is provided, generate the map
        if communities and community_map is None:
            self.community_map = {}
            for i, community in enumerate(communities):
                for node in community:
                    self.community_map[node] = i

        # If only map is provided, generate the communities
        elif community_map and communities is None:
            reverse_map = {}
            for node, comm_id in community_map.items():
                if comm_id not in reverse_map:
                    reverse_map[comm_id] = set()
                reverse_map[comm_id].add(node)
            self.communities = [nodes for _,
                                nodes in sorted(reverse_map.items())]
