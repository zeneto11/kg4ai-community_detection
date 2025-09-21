import ast
import json
import logging
import time
from pathlib import Path

import networkx as nx
import pandas as pd

from community_detection.evaluation.community_metrics import evaluate_communities
from community_detection.evaluation.reporter import analyze_graph_enhanced
from community_detection.methods.infomap_detector import InfomapDetector
from community_detection.methods.leiden_detector import LeidenDetector
from community_detection.methods.louvain_detector import LouvainDetector
from community_detection.utils.logger import (init_logger, log_header,
                                              log_substep)
from community_detection.utils.run_manager import get_run
from community_detection.utils.time import format_time


# def create_test_graph():
#     """Create different types of test graphs to demonstrate reporting."""

#     # 3. Directed citation-like graph
#     citation_graph = nx.barabasi_albert_graph(200, 3, seed=42)
#     # Convert to directed and add some realistic citation patterns
#     citation_graph = citation_graph.to_directed()

#     # Add some node metadata to simulate papers
#     for i, node in enumerate(citation_graph.nodes()):
#         citation_graph.nodes[node]['title'] = f"Paper_{i:03d}"
#         citation_graph.nodes[node]['year'] = 2020 + \
#             (i % 5)  # Papers from 2020-2024

#     # Remove some edges to make it more citation-like (younger papers don't cite older ones)
#     edges_to_remove = []
#     for u, v in citation_graph.edges():
#         u_year = citation_graph.nodes[u].get('year', 2020)
#         v_year = citation_graph.nodes[v].get('year', 2020)
#         if u_year < v_year:  # Remove "backwards" citations
#             edges_to_remove.append((u, v))

#     citation_graph.remove_edges_from(edges_to_remove)

#     citation_info = {
#         "source": "Simulated Citation Network",
#         "description": "Directed graph simulating academic paper citations",
#         "parameters": {"n": 200, "temporal_structure": True}
#     }

#     return citation_graph, citation_info

def create_test_graph():
    """Create test graph with metadata."""
    # Load dataset
    df = pd.read_csv("data/v0.0/df_nq_version0.csv")

    # Convert stringified lists into real Python lists
    df["cites_ids"] = df["cites_ids"].apply(ast.literal_eval)

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges to the graph
    for _, row in df.iterrows():
        src = row["id"]
        G.add_node(src, title=row["d_properties_document_title"])
        for tgt in row["cites_ids"]:
            G.add_edge(src, tgt)

    # Graph metadata for reporting
    graph_info = {
        "source": "NQ (Natural Questions) from Google",
        "description": "Graph constructed from wiki articles citation data in the Natural Questions dataset"
    }

    return G, graph_info


# def create_test_graph():
#     """Create test graph with metadata."""
#     # Use built-in test graph
#     G = nx.LFR_benchmark_graph(
#         n=250, tau1=3, tau2=1.5, mu=0.1,
#         average_degree=5, min_community=20,
#         seed=42
#     )

#     # Graph metadata for reporting
#     graph_info = {
#         "source": "LFR Benchmark",
#         "description": "Synthetic graph generated using LFR benchmark for community detection testing"
#     }

#     return G, graph_info


def main():
    start_time = time.time()

    # Setup with enhanced run management
    output_path = get_run(Path("community_detection/output"))
    run_id = output_path.name

    # Setup logging
    init_logger(output_path / "pipeline.log")
    log_header("Initializing Community Detection Pipeline")
    logger = logging.getLogger(__name__)

    # Load graph
    G, graph_info = create_test_graph()
    logger.info(f"{graph_info['source']} graph")
    logger.info(f"Description: {graph_info['description']}")

    # ===================== Graph Analysis =====================
    log_header("Graph Analysis")

    # Analyze graph and remove isolated nodes
    metrics, G = analyze_graph_enhanced(
        G,
        output_dir=output_path,
        run_id=run_id,
        graph_info=graph_info,
        include_citation_analysis=True,
        citation_top_n=10,
        threshold=10000,
        compute_advanced_metrics=False,
        remove_isolated=True
    )

    # ===================== Community Detection =====================
    log_header("Community Detection")

    # Detect communities with Infomap
    # infomap_detector = InfomapDetector(num_trials=10)
    # infomap_detector.fit(G)
    # infomap_communities = infomap_detector.get_communities()

    # logger.info(f"Infomap found {len(infomap_communities)} communities")

    # Detect communities with Louvain
    # louvain_detector = LouvainDetector()
    # louvain_detector.fit(G)
    # louvain_communities = louvain_detector.get_communities()

    # logger.info(f"Louvain found {len(louvain_communities)} communities")

    # Detect communities with Leiden
    start_time_leiden = time.time()
    leiden_detector = LeidenDetector()
    leiden_detector.fit(G)
    leiden_communities = leiden_detector.get_communities()

    logger.info(
        f"Leiden found {len(leiden_communities)} communities in {format_time(time.time() - start_time_leiden)}")

    # ===================== Evaluation =====================
    log_header("Community Evaluation")

    # Evaluate communities
    # infomap_metrics = evaluate_communities(G, infomap_communities)
    # louvain_metrics = evaluate_communities(G, louvain_communities)
    leiden_metrics = evaluate_communities(G, leiden_communities)

    # Save community metrics in structured format
    community_results = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "graph_info": graph_info,
        # "infomap": {
        #     "num_communities": len(infomap_communities),
        #     "metrics": infomap_metrics
        # },
        # "louvain": {
        #     "num_communities": len(louvain_communities),
        #     "metrics": louvain_metrics
        # },
        "leiden": {
            "num_communities": len(leiden_communities),
            "metrics": leiden_metrics
        }
    }

    # Save structured community results
    community_path = output_path / f"{run_id}_community_results.json"
    with open(community_path, 'w') as f:
        json.dump(community_results, f, indent=2)

    log_substep(f"Community results saved to {community_path}")

    # Log comparison-friendly summary
    logger.info("📊 COMMUNITY DETECTION COMPARISON")
    logger.info("="*50)
    logger.info(
        f"{'Method':<12} {'Communities':<12} {'Modularity':<12} {'Homophily':<12}")
    logger.info("-"*50)
    # logger.info(
    # f"{'Infomap':<12} {len(infomap_communities):<12} {infomap_metrics['modularity']:<12.4f} {infomap_metrics['homophily']:<12.4f}")
    # logger.info(
    #     f"{'Louvain':<12} {len(louvain_communities):<12} {louvain_metrics['modularity']:<12.4f} {louvain_metrics['homophily']:<12.4f}")
    logger.info(
        f"{'Leiden':<12} {len(leiden_communities):<12} {leiden_metrics['modularity']:<12.4f} {leiden_metrics['homophily']:<12.4f}")
    logger.info("="*50)

    # ===================== Visualization =====================
    log_header("Community Visualization")

    # Visualize communities
    # logger.info("\nPlotting community structure...")
    # plot_community_graph(G, infomap_communities, title="Infomap Communities",
    #                      output_file="infomap_communities.png")
    # plot_community_graph(G, louvain_communities, title="Louvain Communities",
    #                      output_file="louvain_communities.png")

    # logger.info("Visualization saved to infomap_communities.png and louvain_communities.png")

    # ===================== Pipeline Summary =====================
    elapsed = time.time() - start_time
    log_header("Pipeline Complete")
    log_substep(f"All outputs saved to {output_path}")

    logger.info("="*67)
    logger.info("Pipeline Summary".center(67))
    logger.info("-"*67)
    logger.info(f"   • Run ID: {run_id}")
    logger.info(
        f"   • Graph: {graph_info['source']} ({metrics['basic_properties']['num_nodes']} nodes)")
    logger.info(f"   • Runtime: {format_time(elapsed)}")


if __name__ == "__main__":
    main()
