# community_detection/evaluation/evaluate.py

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx

from community_detection.evaluation.community_metrics import CommunityEvaluator
from community_detection.evaluation.graph_metrics import (
    CitationNetworkAnalyzer, GraphMetricsComputer)
from community_detection.evaluation.reporter import \
    EnhancedGraphMetricsReporter
from community_detection.utils.logger import log_substep


def analyze_graph(
    G: nx.Graph,
    output_dir: Path,
    run_id: str = None,
    graph_info: Dict[str, Any] = None,
    include_citation_analysis: bool = False,
    citation_top_n: int = 10,
    **kwargs
) -> Tuple[Dict[str, Any], nx.Graph]:
    """
    Pipeline: compute metrics + generate reports (optionally with citation analysis).
    """
    logger = logging.getLogger(__name__)

    # initialize components
    computer = GraphMetricsComputer(logger)
    citation_analyzer = CitationNetworkAnalyzer()
    reporter = EnhancedGraphMetricsReporter(output_dir, run_id)

    # 1. compute metrics
    try:
        metrics, processed_G = computer.compute_all_metrics(G, **kwargs)
    except Exception as e:
        logger.error(f"Graph analysis failed: {e}")
        raise

    # 2. optional citation analysis
    citation_metrics = None
    if include_citation_analysis:
        try:
            citation_metrics = citation_analyzer.analyze(
                processed_G, top_n=citation_top_n)
        except Exception as e:
            logger.error(f"Citation analysis failed: {e}")
            raise

    # 3. generate reports
    try:
        enhanced_metrics = reporter.generate_reports(
            metrics, processed_G, graph_info, citation_metrics
        )
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise

    return enhanced_metrics, processed_G


def analyze_communities(
    G: nx.Graph,
    communities: Dict[str, List[List[int]]],
    output_path: Path,
    run_id: str,
    graph_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate and compare detected communities from different algorithms.
    """

    # Initialize evaluator and logger
    logger = logging.getLogger(__name__)
    evaluator = CommunityEvaluator(G)

    # Evaluate each community set
    metrics_results = {}
    for method, comms in communities.items():
        metrics_results[method] = {
            "num_communities": len(comms),
            "metrics": evaluator.evaluate(comms)
        }

    # Structure results
    community_results = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "graph_info": graph_info,
        **metrics_results
    }

    # Save JSON
    community_path = output_path / f"{run_id}_community_results.json"
    with open(community_path, "w") as f:
        json.dump(community_results, f, indent=2)
    log_substep(f"Community results saved to {community_path}")

    # Exportar para CSV
    csv_path = evaluator.export_to_csv(community_results, output_path)
    log_substep(f"Community comparison CSV saved to {csv_path}")

    # Generate community size distribution plots
    bar_plot_paths = evaluator.plot_size_distribution(communities, output_path)
    log_substep("Community size bar plots generated:")
    for algorithm, path in bar_plot_paths.items():
        logger.info(f"   • {algorithm.title()}: {path}")

    # Log comparative summary
    logger.info("📊 COMMUNITY DETECTION COMPARISON")
    logger.info("=" * 67)
    logger.info(
        f"{'Method':<12} {'Communities':<12} {'Modularity':<12} {'Avg TPR':<12} {'Avg Conductance':<12}"
    )
    logger.info("-" * 67)

    for method, result in metrics_results.items():
        m = result["metrics"]
        logger.info(
            f"{method.title():<12} "
            f"{result['num_communities']:<12} "
            f"{m['modularity']:<12.4f} "
            f"{m['avg_tpr']:<12.4f} "
            f"{m['avg_conductance']:<12.4f}"
        )

    logger.info("=" * 67)

    return community_results
