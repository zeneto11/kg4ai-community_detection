# community_detection/utils/graph_utility.py

import logging
from collections import Counter
from typing import Union

import networkx as nx

# Set up module-level logger
logger = logging.getLogger(__name__)


def convert_to_weighted_undirected(graph: Union[nx.Graph, nx.DiGraph]) -> nx.Graph:
    """Convert directed graph to undirected, summing citation weights."""
    G_undirected = nx.Graph()
    for u, v in graph.edges():
        if G_undirected.has_edge(u, v):
            G_undirected[u][v]['weight'] += 1
        else:
            G_undirected.add_edge(u, v, weight=1)
    # Copy node attributes
    for n, attrs in graph.nodes(data=True):
        G_undirected.add_node(n, **attrs)

    # Show weight stats
    weights = [data['weight']
               for _, _, data in G_undirected.edges(data=True)]
    if weights:
        weights_counts = Counter(weights)
        logger.info(
            f"Converted to undirected weighted graph:")
        logger.info(
            f"   Weight distribution: {dict(sorted(weights_counts.items()))}")

    return G_undirected
