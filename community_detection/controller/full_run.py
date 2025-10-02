import ast
import logging
import time
from pathlib import Path

import networkx as nx
import pandas as pd

from community_detection.evaluation import evaluator
from community_detection.methods.infomap_detector import InfomapDetector
from community_detection.methods.kmeans_detector import KMeansDetector
from community_detection.methods.leiden_detector import LeidenDetector
from community_detection.methods.louvain_detector import LouvainDetector
from community_detection.utils.logger import (init_logger, log_header,
                                              log_substep)
from community_detection.utils.run_manager import get_run
from community_detection.utils.time import format_time
from community_detection.visualization.visualizer import run_community_visualization


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


def main():
    start_time = time.time()

    # Create new output folder for this run
    output_path = get_run(Path("community_detection/output"))
    run_id = output_path.name

    # Logging setup
    init_logger(output_path / "pipeline.log")
    log_header("Initializing Community Detection Pipeline")
    logger = logging.getLogger(__name__)

    # Load input graph
    G, graph_info = create_test_graph()
    logger.info(f"{graph_info['source']} graph")
    logger.info(f"Description: {graph_info['description']}")

    # ===================== Graph Analysis =====================
    log_header("Graph Analysis")

    # Analyze graph
    metrics, G = evaluator.analyze_graph(
        G,
        output_dir=output_path,             # where metrics/plots are saved
        run_id=run_id,                      # run identifier for outputs
        graph_info=graph_info,              # metadata for reports
        include_citation_analysis=True,     # enable if nodes have 'title' + citation edges
        citation_top_n=10,                  # report top-N most cited nodes
        # max nodes for exact heavy metrics; larger graphs use approximations/skip
        threshold=10000,
        compute_advanced_metrics=False,     # skip heavy metrics unless needed
        remove_isolated=True                # drop disconnected nodes and small components
    )

    # ===================== Community Detection =====================
    log_header("Community Detection")

    def run_detector(name, detector, G, logger):
        start_time = time.time()
        detector.fit(G)
        communities = detector.get_communities()
        elapsed = format_time(time.time() - start_time)
        logger.info(
            f"{name} found {len(communities)} communities in {elapsed}")
        logger.info("-" * 67)
        return communities

    # Define detectors to run
    detectors = {
        "Infomap": InfomapDetector(),
        "Louvain": LouvainDetector(),
        "Leiden": LeidenDetector(),
        "K-means": KMeansDetector(
            k=50,
            embedding_model="all-MiniLM-L6-v2",
            umap_components=15,
            auto_select_k=True
        )
    }

    # Run all detectors and collect communities
    communities_dict = {
        name.lower(): run_detector(name, detector, G, logger)
        for name, detector in detectors.items()
    }

    # ===================== Evaluation =====================
    log_header("Community Evaluation")

    # Evaluate and compare communities
    evaluator.analyze_communities(
        G,
        communities_dict,
        output_path=output_path,
        run_id=run_id,
        graph_info=graph_info
    )

    # ===================== Visualization =====================
    log_header("Community Visualization")

    # Generate visualizations and keywords
    visualization_results = run_community_visualization(
        G=G,
        communities_dict=communities_dict,
        output_path=output_path,
        run_id=run_id,
        logger=logger
    )

    # ===================== Pipeline Summary =====================
    elapsed = time.time() - start_time
    log_header("Pipeline Complete")
    log_substep(f"All outputs saved to {output_path}")

    logger.info("="*67)
    logger.info("Pipeline Summary".center(67))
    logger.info("-"*67)
    logger.info(f"   • Run ID: {run_id}")
    logger.info(f"   • Runtime: {format_time(elapsed)}")


if __name__ == "__main__":
    main()
