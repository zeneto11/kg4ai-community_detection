# community_detection/visualization/visualizer.py

import json
import logging
from pathlib import Path
from typing import Dict

from community_detection.utils.logger import log_substep
from community_detection.visualization.community_namer import SimpleKeywordExtractor
from community_detection.visualization.community_visualizer import MacroGraphBuilder, MacroGraphVisualizer


def run_community_visualization(
    G,
    communities_dict: Dict,
    output_path: Path,
    run_id: str,
    logger: logging.Logger = None
) -> Dict:
    """
    Run community visualization pipeline including macro-graph visualizations
    and keyword extraction.

    Args:
        G: NetworkX graph
        communities_dict: Dictionary of communities by algorithm
        output_path: Output directory path
        run_id: Run identifier
        logger: Logger instance

    Returns:
        Dictionary containing visualization results and paths
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # ====== Generate community keywords ======
    log_substep("Extracting community keywords...")
    keyword_extractor = SimpleKeywordExtractor()

    community_keywords = {}
    for algorithm, communities_list in communities_dict.items():
        keywords = keyword_extractor.extract_community_keywords(
            communities_list, G, top_k=10)
        community_keywords[algorithm] = keywords
        logger.info(
            f"   • {algorithm.title()}: {len(keywords)} communities processed")

    # Save community keywords to JSON
    keywords_path = output_path / f"{run_id}_community_keywords.json"
    with open(keywords_path, 'w') as f:
        json.dump(community_keywords, f, indent=2,
                  default=lambda x: list(x) if isinstance(x, set) else x)
    log_substep(f"Community keywords saved to {keywords_path}")

    # Log some example keywords
    log_substep("Example community keywords:")
    for algorithm, keywords_dict in community_keywords.items():
        logger.info(f"   • {algorithm.title()}:")
        # Show keywords for first 2 communities
        for comm_id in list(keywords_dict.keys())[:2]:
            keywords = keywords_dict[comm_id]['keywords']
            comm_name = keywords_dict[comm_id]['comm_name']

            # Show top 5 with counts
            top_keywords = [f"{kw[0]}({kw[1]})" for kw in keywords[:5]]
            logger.info(
                f"      Community {comm_id} ({comm_name}): {', '.join(top_keywords)}")
        logger.info("      ...")

    # ====== Generate macro-graph visualizations ======
    builder = MacroGraphBuilder(G, communities_dict)
    for algorithm in communities_dict.keys():
        logger.info(
            f"Generating macro-graph visualizations for {algorithm.title()}...")

        # Create macro-graphs
        macro_w, macro_t = builder.build_macrograph(
            algorithm, max_communities=100)

        # Visualize and save plots
        visualizer = MacroGraphVisualizer(
            macro_w, macro_t, algorithm_name=algorithm.title(), output_dir=output_path)
        visualizer.plot_both()

    return {
        "community_keywords": community_keywords,
        "keywords_path": keywords_path
    }
