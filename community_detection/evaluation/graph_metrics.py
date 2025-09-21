import logging
import statistics

import networkx as nx
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


def analyze_graph(G, threshold=10000, compute_advanced_metrics=True, remove_isolated=True, print_results=True):
    """
    Comprehensive analysis of graph properties and metrics.

    Parameters:
    -----------
    G : networkx.Graph or networkx.DiGraph
        The graph to analyze
    threshold : int, default=10000
        Node count threshold to decide whether to compute expensive metrics
    compute_advanced_metrics : bool, default=True
        Whether to compute computationally expensive metrics
    remove_isolated : bool, default=True
        Whether to remove isolated nodes before analysis
    print_results : bool, default=True
        Whether to print results (otherwise just returns the metrics dict)

    Returns:
    --------
    dict
        Dictionary containing all computed metrics
    """
    results = {}

    # Basic properties
    results["is_directed"] = G.is_directed()
    results["is_weighted"] = nx.is_weighted(G)
    results["is_multigraph"] = G.is_multigraph()

    # Make a copy to avoid modifying the original graph
    if remove_isolated:
        G = G.copy()

    # Connectivity analysis
    G_undirected = G.to_undirected()
    results["is_connected_undirected"] = nx.is_connected(G_undirected)

    if G.is_directed():
        results["is_strongly_connected"] = nx.is_strongly_connected(G)
        results["is_weakly_connected"] = nx.is_weakly_connected(G)
        results["num_weakly_connected_components"] = nx.number_weakly_connected_components(
            G)
        results["num_strongly_connected_components"] = nx.number_strongly_connected_components(
            G)

        # Get sizes of components
        weak_components = list(nx.weakly_connected_components(G))
        strong_components = list(nx.strongly_connected_components(G))

        results["largest_weakly_connected_component_size"] = len(
            max(weak_components, key=len))
        results["largest_strongly_connected_component_size"] = len(
            max(strong_components, key=len))

        # Component size statistics
        weak_comp_sizes = [len(c) for c in weak_components]
        strong_comp_sizes = [len(c) for c in strong_components]

        results["weakly_connected_component_sizes"] = weak_comp_sizes
        results["strongly_connected_component_sizes"] = strong_comp_sizes

        # Calculate component size distribution stats
        results["weakly_cc_size_mean"] = np.mean(weak_comp_sizes)
        results["weakly_cc_size_median"] = np.median(weak_comp_sizes)
        results["strongly_cc_size_mean"] = np.mean(strong_comp_sizes)
        results["strongly_cc_size_median"] = np.median(strong_comp_sizes)
    else:
        # For undirected graphs
        components = list(nx.connected_components(G))
        results["num_connected_components"] = len(components)
        results["largest_connected_component_size"] = len(
            max(components, key=len))
        results["connected_component_sizes"] = [len(c) for c in components]

        # Calculate component size distribution stats
        comp_sizes = [len(c) for c in components]
        results["cc_size_mean"] = np.mean(comp_sizes)
        results["cc_size_median"] = np.median(comp_sizes)

    # Node-level statistics
    results["num_nodes"] = G.number_of_nodes()
    results["num_edges"] = G.number_of_edges()

    if G.is_directed():
        results["num_nodes_zero_in_degree"] = len(
            [n for n, d in G.in_degree() if d == 0])
        results["num_nodes_zero_out_degree"] = len(
            [n for n, d in G.out_degree() if d == 0])

    results["num_nodes_zero_total_degree"] = len(
        [n for n, d in G.degree() if d == 0])

    # Remove isolated nodes if requested
    if remove_isolated and results["num_nodes_zero_total_degree"] > 0:
        isolated_nodes = [n for n, d in G.degree() if d == 0]
        G.remove_nodes_from(isolated_nodes)
        results["num_isolated_nodes_removed"] = len(isolated_nodes)
        results["num_nodes_after_removal"] = G.number_of_nodes()
        results["num_edges_after_removal"] = G.number_of_edges()

    # Trivial graph statistics
    results["average_degree"] = statistics.mean(
        [d for _, d in G.degree()]) if G.number_of_nodes() > 0 else 0
    results["density"] = nx.density(G)
    results["self_loops"] = nx.number_of_selfloops(G)

    if G.is_directed():
        try:
            results["reciprocity"] = nx.reciprocity(G)
        except:
            results["reciprocity"] = None

    # Degree statistics
    all_degrees = [d for _, d in G.degree()]
    results["max_degree"] = max(all_degrees) if all_degrees else 0
    results["min_degree"] = min(all_degrees) if all_degrees else 0
    results["median_degree"] = statistics.median(
        all_degrees) if all_degrees else 0
    results["std_dev_degree"] = statistics.stdev(
        all_degrees) if len(all_degrees) > 1 else 0

    if G.is_directed():
        in_degrees = [d for _, d in G.in_degree()]
        out_degrees = [d for _, d in G.out_degree()]

        results["max_in_degree"] = max(in_degrees) if in_degrees else 0
        results["min_in_degree"] = min(in_degrees) if in_degrees else 0
        results["mean_in_degree"] = statistics.mean(
            in_degrees) if in_degrees else 0
        results["median_in_degree"] = statistics.median(
            in_degrees) if in_degrees else 0

        results["max_out_degree"] = max(out_degrees) if out_degrees else 0
        results["min_out_degree"] = min(out_degrees) if out_degrees else 0
        results["mean_out_degree"] = statistics.mean(
            out_degrees) if out_degrees else 0
        results["median_out_degree"] = statistics.median(
            out_degrees) if out_degrees else 0

    # Advanced metrics (computationally expensive)
    if compute_advanced_metrics:
        logger.info('Initiating advanced metrics calculation...')

        # Transitivity and clustering
        logger.info('Calculating transitivity...')
        results["transitivity"] = nx.transitivity(G)
        logger.info('Calculating average clustering...')
        results["average_clustering"] = nx.average_clustering(G)

        # Filtered average clustering (only nodes with degree > 1)
        logger.info('Calculating filtered average clustering...')
        all_cc = list(nx.clustering(G).values())
        cc_list = []
        for i, x in enumerate(all_degrees):
            if x > 1:
                cc_list.append(all_cc[i])
        results["filtered_average_clustering"] = statistics.mean(
            cc_list) if cc_list else 0

        # For connected graphs or largest component
        if results.get("is_connected_undirected", False):
            try:
                logger.info('Calculating average shortest path length...')
                results["average_shortest_path_length"] = nx.average_shortest_path_length(
                    G_undirected)
                logger.info('Calculating diameter...')
                results["diameter"] = nx.diameter(G_undirected)
                logger.info('Calculating radius...')
                results["radius"] = nx.radius(G_undirected)
            except nx.NetworkXError:
                # Handle disconnected graphs - use largest component
                pass

        # Get largest connected component for disconnected graphs
        if not results.get("is_connected_undirected", False):
            logger.info(
                'Calculating metrics for largest connected component...')
            largest_cc = max(nx.connected_components(G_undirected), key=len)
            largest_cc_graph = G_undirected.subgraph(largest_cc).copy()
            results["largest_cc_size"] = len(largest_cc)
            results["largest_cc_fraction"] = len(
                largest_cc) / G.number_of_nodes()

            try:
                results["largest_cc_average_shortest_path_length"] = nx.average_shortest_path_length(
                    largest_cc_graph)
                results["largest_cc_diameter"] = nx.diameter(largest_cc_graph)
                results["largest_cc_radius"] = nx.radius(largest_cc_graph)
            except:
                # Handle errors in case of very large components
                pass

        # Centrality measures - use threshold to avoid memory issues
        node_count = G.number_of_nodes()
        if node_count <= threshold:  # Adjust threshold based on available memory
            logger.info('Calculating centrality measures...')
            try:
                results["average_degree_centrality"] = np.mean(
                    list(dict(nx.degree_centrality(G)).values()))
            except:
                pass

            try:
                results["average_closeness_centrality"] = np.mean(
                    list(dict(nx.closeness_centrality(G)).values()))
            except:
                pass

            if node_count <= threshold/10:  # Betweenness is very expensive
                try:
                    results["average_betweenness_centrality"] = np.mean(list(dict(nx.betweenness_centrality(G,
                                                                                                            normalized=True,
                                                                                                            k=min(500, node_count))).values()))
                except:
                    pass

                # Eigenvector centrality
                try:
                    results["average_eigenvector_centrality"] = np.mean(list(dict(nx.eigenvector_centrality(
                        G, max_iter=100, tol=1e-4)).values()))
                except:
                    results["average_eigenvector_centrality"] = "Failed to converge"

            # PageRank for directed graphs
            if G.is_directed() and node_count <= threshold:
                try:
                    pagerank = nx.pagerank(G, alpha=0.85)
                    results["average_pagerank"] = np.mean(
                        list(pagerank.values()))
                    results["max_pagerank"] = max(pagerank.values())
                except:
                    pass

        # Additional metrics
        # Bridges (edges that would increase number of connected components if removed)
        if not G.is_directed():
            logger.info('Calculating number of bridges...')
            try:
                results["num_bridges"] = len(list(nx.bridges(G)))
            except:
                pass

        # Assortativity - do nodes connect to similar degree nodes?
        try:
            logger.info('Calculating degree assortativity...')
            results["degree_assortativity"] = nx.degree_assortativity_coefficient(
                G)
        except:
            results["degree_assortativity"] = None

        # Core decomposition - find k-cores (subgraphs where all nodes have degree >= k)
        # Create a copy without self-loops for core number calculation
        logger.info('Calculating core numbers...')
        if nx.number_of_selfloops(G) > 0:
            G_no_loops = G.copy()
            G_no_loops.remove_edges_from(nx.selfloop_edges(G_no_loops))
            core_graph = G_no_loops.to_undirected()
        else:
            core_graph = G_undirected

        try:
            core_numbers = nx.core_number(core_graph)
            results["max_core_number"] = max(
                core_numbers.values()) if core_numbers else 0
            results["mean_core_number"] = np.mean(list(core_numbers.values()))
        except:
            pass

        # Rich club coefficient - tendency of high-degree nodes to connect to each other
        logger.info('Calculating rich club coefficient...')
        if not G.is_directed() and node_count <= threshold/2:
            try:
                # 90th percentile degree
                k = max(5, int(np.percentile([d for _, d in G.degree()], 90)))
                results["rich_club_coefficient"] = nx.rich_club_coefficient(
                    G, normalized=False).get(k, None)
            except:
                results["rich_club_coefficient"] = None

        # Efficiency measures
        logger.info('Calculating global efficiency...')
        if node_count <= threshold/2:
            try:
                results["global_efficiency"] = nx.global_efficiency(
                    G_undirected)
            except:
                pass

    # Print results if requested
    if print_results:
        logger.info("\n===== Basic Graph Properties =====")
        logger.info(
            f" • Is directed..............................: {results['is_directed']}")
        logger.info(
            f" • Is weighted..............................: {results['is_weighted']}")
        logger.info(
            f" • Is multi-graph...........................: {results['is_multigraph']}")
        logger.info(
            f" • Is connected (undirected view)...........: {results['is_connected_undirected']}")

        if G.is_directed():
            logger.info(
                f" • Is strongly connected....................: {results['is_strongly_connected']}")
            logger.info(
                f" • Is weakly connected......................: {results['is_weakly_connected']}")

        logger.info("\n===== Node-level Statistics =====")
        logger.info(
            f" • Number of nodes..........................: {results['num_nodes']}")

        if G.is_directed():
            logger.info(
                f" • Number of nodes with zero in-degree......: {results['num_nodes_zero_in_degree']}")
            logger.info(
                f" • Number of nodes with zero out-degree.....: {results['num_nodes_zero_out_degree']}")

        logger.info(
            f" • Number of nodes with zero total degree...: {results['num_nodes_zero_total_degree']}")

        if remove_isolated and results.get("num_isolated_nodes_removed", 0) > 0:
            logger.info(
                f"\nRemoved {results['num_isolated_nodes_removed']} isolated nodes")
            logger.info(
                f" • Nodes after removal.....................: {results['num_nodes_after_removal']}")
            logger.info(
                f" • Edges after removal.....................: {results['num_edges_after_removal']}")

        logger.info("\n===== Global Graph Statistics - Trivial =====")
        logger.info(
            f" • Number of edges..........................: {results['num_edges']}")
        logger.info(
            f" • Average degree...........................: {results['average_degree']:.2f}")
        logger.info(
            f" • Density..................................: {results['density']:.6f}")
        logger.info(
            f" • Self-loops...............................: {results['self_loops']}")

        if G.is_directed():
            logger.info(
                f" • Reciprocity..............................: {results.get('reciprocity', 'N/A')}")
            logger.info(
                f" • Number of connected components (weakly)..: {results['num_weakly_connected_components']}")
            logger.info(
                f" • Number of connected components (strongly): {results['num_strongly_connected_components']}")
            logger.info(
                f" • Largest weakly connected component size..: {results['largest_weakly_connected_component_size']}")
            logger.info(
                f" • Largest strongly connected component size: {results['largest_strongly_connected_component_size']}")
            logger.info(
                f" • Weak component size (mean)...............: {results['weakly_cc_size_mean']:.2f}")
            logger.info(
                f" • Strong component size (mean).............: {results['strongly_cc_size_mean']:.2f}")
        else:
            logger.info(
                f" • Number of connected components...........: {results['num_connected_components']}")
            logger.info(
                f" • Largest connected component size.........: {results['largest_connected_component_size']}")
            logger.info(
                f" • Component size (mean)....................: {results.get('cc_size_mean', 'N/A')}")

        if compute_advanced_metrics:
            logger.info("\n===== Global Graph Statistics - Advanced =====")
            logger.info(
                f" • Transitivity.............................: {results['transitivity']:.6f}")
            logger.info(
                f" • Average clustering coefficient...........: {results['average_clustering']:.6f}")
            logger.info(
                f" • Filtered avg clustering (deg > 1)........: {results['filtered_average_clustering']:.6f}")

            if "average_shortest_path_length" in results:
                logger.info(
                    f" • Average distance.........................: {results['average_shortest_path_length']:.4f}")
                logger.info(
                    f" • Diameter.................................: {results['diameter']}")
                logger.info(
                    f" • Radius...................................: {results['radius']}")
            elif "largest_cc_average_shortest_path_length" in results:
                logger.info(
                    " • Graph is disconnected. Metrics for largest connected component:")
                logger.info(
                    f"   - Size...................................: {results['largest_cc_size']} ({results.get('largest_cc_fraction', 0)*100:.1f}% of graph)")
                logger.info(
                    f"   - Average distance.......................: {results.get('largest_cc_average_shortest_path_length', 'N/A')}")
                logger.info(
                    f"   - Diameter...............................: {results.get('largest_cc_diameter', 'N/A')}")
                logger.info(
                    f"   - Radius.................................: {results.get('largest_cc_radius', 'N/A')}")

            if "global_efficiency" in results:
                logger.info(
                    f" • Global efficiency........................: {results['global_efficiency']:.6f}")

            if "average_degree_centrality" in results:
                logger.info(
                    f" • Average degree centrality................: {results['average_degree_centrality']:.6f}")

            if "average_closeness_centrality" in results:
                logger.info(
                    f" • Average closeness centrality.............: {results['average_closeness_centrality']:.6f}")

            if "average_betweenness_centrality" in results:
                logger.info(
                    f" • Average betweenness centrality...........: {results['average_betweenness_centrality']:.6f}")

            if "average_eigenvector_centrality" in results and results["average_eigenvector_centrality"] != "Failed to converge":
                logger.info(
                    f" • Average eigenvector centrality...........: {results['average_eigenvector_centrality']:.6f}")

            if "average_pagerank" in results:
                logger.info(
                    f" • Average PageRank.........................: {results['average_pagerank']:.6f}")
                logger.info(
                    f" • Maximum PageRank.........................: {results['max_pagerank']:.6f}")

            if "max_core_number" in results:
                logger.info(
                    f" • Max core number..........................: {results['max_core_number']}")
                if "mean_core_number" in results:
                    logger.info(
                        f" • Mean core number.........................: {results['mean_core_number']:.2f}")

            if "num_bridges" in results:
                logger.info(
                    f" • Number of bridges........................: {results['num_bridges']}")

            if results.get("degree_assortativity") is not None:
                logger.info(
                    f" • Degree assortativity.....................: {results['degree_assortativity']:.6f}")

            if results.get("rich_club_coefficient") is not None:
                logger.info(
                    f" • Rich club coefficient....................: {results['rich_club_coefficient']:.6f}")

    return results, G


def analyze_citation_network(G, top_n=10, print_results=True):
    """
    Analyzes a citation network with specialized metrics and visualizations for citation data.

    Parameters:
    -----------
    G : networkx.DiGraph
        The citation graph to analyze
    top_n : int, default=10
        Number of top nodes to show in rankings
    print_results : bool, default=True
        Whether to print results

    Returns:
    --------
    dict
        Dictionary containing all computed citation metrics
    """
    results = {}

    # Basic degree statistics
    out_degrees = [deg for _, deg in G.out_degree()]
    in_degrees = [deg for _, deg in G.in_degree()]

    # Summary statistics for citations made (out-degree)
    results["citations_made_mean"] = np.mean(out_degrees)
    results["citations_made_median"] = np.median(out_degrees)
    results["citations_made_max"] = np.max(out_degrees)
    results["citations_made_min"] = np.min(out_degrees)
    results["citations_made_std"] = np.std(out_degrees)

    # Summary statistics for citations received (in-degree)
    results["citations_received_mean"] = np.mean(in_degrees)
    results["citations_received_median"] = np.median(in_degrees)
    results["citations_received_max"] = np.max(in_degrees)
    results["citations_received_min"] = np.min(in_degrees)
    results["citations_received_std"] = np.std(in_degrees)

    # Top citing nodes (articles that cite the most others)
    out_deg_dict = dict(G.out_degree())
    top_out_nodes = sorted(out_deg_dict.items(),
                           key=lambda x: x[1], reverse=True)[:top_n]
    results["top_citing_nodes"] = [(node, G.nodes[node].get(
        'title', str(node)), degree) for node, degree in top_out_nodes]

    # Top cited nodes (articles cited by the most others)
    in_deg_dict = dict(G.in_degree())
    top_in_nodes = sorted(in_deg_dict.items(),
                          key=lambda x: x[1], reverse=True)[:top_n]
    results["top_cited_nodes"] = [(node, G.nodes[node].get(
        'title', str(node)), degree) for node, degree in top_in_nodes]

    # Special node categories
    results["num_isolated_nodes"] = len([n for n, d in G.degree() if d == 0])
    results["num_terminal_nodes"] = len(
        # Nodes that don't cite others
        [n for n, d in G.out_degree() if d == 0])
    results["num_source_nodes"] = len(
        [n for n, d in G.in_degree() if d == 0])     # Nodes not cited by others

    # H-index of the network
    in_degrees_sorted = sorted(in_degrees, reverse=True)
    h_index = 0
    for i, citations in enumerate(in_degrees_sorted):
        if i+1 <= citations:
            h_index = i+1
        else:
            break
    results["h_index"] = h_index

    # PageRank - article influence score
    pagerank_scores = nx.pagerank(G, alpha=0.85)
    pagerank_top_nodes = sorted(
        pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    results["pagerank_top_nodes"] = [(node, G.nodes[node].get(
        'title', str(node)), score) for node, score in pagerank_top_nodes]
    results["pagerank_mean"] = np.mean(list(pagerank_scores.values()))
    results["pagerank_std"] = np.std(list(pagerank_scores.values()))

    # Additional citation-specific metrics
    # Citation concentration (Gini coefficient for citation distribution)
    def gini_coefficient(x):
        """Calculate Gini coefficient for inequality measurement."""
        sorted_x = sorted(x)
        n = len(x)
        cumsum = np.cumsum(sorted_x)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0

    results["citation_gini_coefficient"] = gini_coefficient(in_degrees)

    # Self-citation rate (if graph has self-loops)
    self_citations = nx.number_of_selfloops(G)
    total_citations = G.number_of_edges()
    results["self_citation_rate"] = self_citations / \
        total_citations if total_citations > 0 else 0

    # Citation age analysis (if nodes have temporal data)
    # This would require additional node attributes - placeholder for now
    results["temporal_analysis_available"] = any(
        'year' in G.nodes[n] or 'date' in G.nodes[n] for n in list(G.nodes())[:10])

    # Authority and hub scores (HITS algorithm)
    try:
        hits_scores = nx.hits(G, max_iter=100, normalized=True)
        authorities, hubs = hits_scores
        results["top_authorities"] = sorted(
            authorities.items(), key=lambda x: x[1], reverse=True)[:top_n]
        results["top_hubs"] = sorted(
            hubs.items(), key=lambda x: x[1], reverse=True)[:top_n]
        results["authority_mean"] = np.mean(list(authorities.values()))
        results["hub_mean"] = np.mean(list(hubs.values()))
    except:
        # HITS may not converge for some graphs
        results["hits_analysis_failed"] = True

    # Citation network density vs regular networks
    # (this is already computed in basic analysis, but worth highlighting)
    results["citation_density"] = results.get(
        "citations_made_mean", 0) / (G.number_of_nodes() - 1) if G.number_of_nodes() > 1 else 0

    # Print results if requested
    if print_results:
        logger.info("\n===== Citation Network Analysis =====")

        logger.info("\n----- Citation Distribution Statistics -----")
        logger.info("Citations made (out-degree):")
        logger.info(
            f" • Mean.............: {results['citations_made_mean']:.2f}")
        logger.info(
            f" • Median...........: {results['citations_made_median']:.1f}")
        logger.info(f" • Max..............: {results['citations_made_max']}")
        logger.info(f" • Min..............: {results['citations_made_min']}")
        logger.info(
            f" • Std Deviation....: {results['citations_made_std']:.2f}")

        logger.info("\nCitations received (in-degree):")
        logger.info(
            f" • Mean.............: {results['citations_received_mean']:.2f}")
        logger.info(
            f" • Median...........: {results['citations_received_median']:.1f}")
        logger.info(
            f" • Max..............: {results['citations_received_max']}")
        logger.info(
            f" • Min..............: {results['citations_received_min']}")
        logger.info(
            f" • Std Deviation....: {results['citations_received_std']:.2f}")
        logger.info(f" • Network H-index..: {results['h_index']}")

        logger.info("\n----- Node Categories -----")
        logger.info(
            f" • Isolated nodes (no citations)...: {results['num_isolated_nodes']}")
        logger.info(
            f" • Terminal nodes (don't cite).....: {results['num_terminal_nodes']}")
        logger.info(
            f" • Source nodes (not cited)........: {results['num_source_nodes']}")

        logger.info("\n----- Top Citing Nodes -----")
        logger.info(f"Top {top_n} nodes that cite the most others:")
        for i, (node, degree) in enumerate(results["top_citing_nodes"], 1):
            node_title = G.nodes[node].get('title', str(node))
            logger.info(f" {i}. {node_title} ({degree} citations made)")

        logger.info(f"\n----- Top Cited Nodes -----")
        logger.info(f"Top {top_n} nodes that are cited by the most others:")
        for i, (node, degree) in enumerate(results["top_cited_nodes"], 1):
            node_title = G.nodes[node].get('title', str(node))
            logger.info(f" {i}. {node_title} ({degree} citations received)")

        logger.info(f"\n----- Most Influential Nodes (PageRank) -----")
        logger.info(f"Top {top_n} nodes by PageRank score:")
        for i, (node, score) in enumerate(results["pagerank_top_nodes"], 1):
            node_title = G.nodes[node].get('title', str(node))
            logger.info(f" {i}. {node_title} (score: {score:.6f})")

        logger.info(f"\n----- Additional Citation Metrics -----")
        logger.info(
            f" • Citation Gini coefficient.......: {results['citation_gini_coefficient']:.4f}")
        logger.info(
            f" • Self-citation rate..............: {results['self_citation_rate']:.4f}")
        logger.info(
            f" • Citation density................: {results['citation_density']:.6f}")

        if "top_authorities" in results:
            logger.info(f"\n----- Top Authorities (HITS Algorithm) -----")
            for i, (node, score) in enumerate(results["top_authorities"][:5], 1):
                node_title = G.nodes[node].get('title', str(node))
                logger.info(f" {i}. {node_title} (authority: {score:.6f})")

            logger.info(f"\n----- Top Hubs (HITS Algorithm) -----")
            for i, (node, score) in enumerate(results["top_hubs"][:5], 1):
                node_title = G.nodes[node].get('title', str(node))
                logger.info(f" {i}. {node_title} (hub: {score:.6f})")

        if results.get("hits_analysis_failed"):
            logger.info(" • HITS analysis failed to converge")

        if results.get("temporal_analysis_available"):
            logger.info(
                " • Temporal data detected - consider time-based analysis")

    return results
