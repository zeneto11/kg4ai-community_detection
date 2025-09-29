# community_detection/evaluation/analyze_graph.py

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import networkx as nx

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
        log_substep("Graph metrics computation completed")
    except Exception as e:
        logger.error(f"Graph analysis failed: {e}")
        raise

    # 2. optional citation analysis
    citation_metrics = None
    if include_citation_analysis:
        try:
            citation_metrics = citation_analyzer.analyze(
                processed_G, top_n=citation_top_n)
            log_substep("Citation network analysis completed")
        except Exception as e:
            logger.error(f"Citation analysis failed: {e}")
            raise

    # 3. generate reports
    try:
        enhanced_metrics = reporter.generate_reports(
            metrics, processed_G, graph_info, citation_metrics
        )
        log_substep("Enhanced reports generated")
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise

    return enhanced_metrics, processed_G
