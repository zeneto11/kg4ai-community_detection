import ast
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, Tuple

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
from community_detection.visualization.community_visualizer import (
    CommunityVisualizer, SimpleKeywordExtractor)


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


# def create_test_graph() -> Tuple[nx.Graph, Dict]:
    """Create a test graph with realistic titles for keyword extraction testing."""

    # Create a scale-free network (similar to citation patterns)
    citation_graph = nx.barabasi_albert_graph(200, 3, seed=42)
    citation_graph = citation_graph.to_directed()

    # Define research domains and topics for realistic titles
    research_domains = {
        "ai": ["machine learning", "neural networks", "deep learning", "reinforcement learning"],
        "nlp": ["natural language processing", "transformer models", "text generation", "sentiment analysis"],
        "cv": ["computer vision", "object detection", "image segmentation", "convolutional networks"],
        "bio": ["bioinformatics", "genomic analysis", "protein folding", "drug discovery"],
        "data": ["data mining", "big data analytics", "distributed systems", "cloud computing"]
    }

    # Create realistic paper titles with keywords
    def generate_paper_title(domain, topic, technique, application):
        templates = [
            f"A novel approach to {application} using {technique}",
            f"{technique.title()} for {application} in {domain}",
            f"Advances in {topic} for {application} applications",
            f"Comparative study of {technique} methods in {domain}",
            f"Enhancing {application} through {technique} optimization"
        ]
        return random.choice(templates)

    # Add realistic node metadata with titles containing keywords
    for i, node in enumerate(citation_graph.nodes()):
        # Assign nodes to research domains (some nodes belong to multiple domains)
        domains = random.sample(
            list(research_domains.keys()), k=random.randint(1, 2))

        # Build title components from assigned domains
        domain_key = random.choice(domains)
        topic = random.choice(research_domains[domain_key])
        technique = random.choice(
            ["deep learning", "statistical analysis", "optimization", "clustering", "classification"])
        application = random.choice(
            ["healthcare", "finance", "education", "manufacturing", "transportation"])

        citation_graph.nodes[node]['title'] = generate_paper_title(
            domain_key, topic, technique, application)
        citation_graph.nodes[node]['year'] = 2020 + \
            (i % 5)  # Papers from 2020-2024
        citation_graph.nodes[node]['domain'] = domains

    # Remove some edges to make it more citation-like (younger papers don't cite older ones)
    edges_to_remove = []
    for u, v in citation_graph.edges():
        u_year = citation_graph.nodes[u].get('year', 2020)
        v_year = citation_graph.nodes[v].get('year', 2020)
        if u_year < v_year:  # Remove "backwards" citations
            edges_to_remove.append((u, v))

    citation_graph.remove_edges_from(edges_to_remove)

    citation_info = {
        "source": "Simulated Citation Network with Realistic Titles",
        "description": "Directed graph simulating academic paper citations with keyword-rich titles",
        "parameters": {
            "n": 200,
            "temporal_structure": True,
            "research_domains": list(research_domains.keys()),
            "title_generation": "template-based with keywords"
        }
    }

    return citation_graph, citation_info

# def create_test_graph():
    """Create test graph with metadata."""
    # Use built-in test graph
    G = nx.LFR_benchmark_graph(
        n=250, tau1=3, tau2=1.5, mu=0.1,
        average_degree=5, min_community=20,
        seed=42
    )

    # Graph metadata for reporting
    graph_info = {
        "source": "LFR Benchmark",
        "description": "Synthetic graph generated using LFR benchmark for community detection testing"
    }

    return G, graph_info

# def create_test_graph():
    """Create different types of test graphs to demonstrate reporting."""

    citation_graph = nx.barabasi_albert_graph(200, 3, seed=42)
    # Convert to directed and add some realistic citation patterns
    citation_graph = citation_graph.to_directed()

    # Add some node metadata to simulate papers
    for i, node in enumerate(citation_graph.nodes()):
        citation_graph.nodes[node]['title'] = f"Paper_{i:03d}"
        citation_graph.nodes[node]['year'] = 2020 + \
            (i % 5)  # Papers from 2020-2024

    # Remove some edges to make it more citation-like (younger papers don't cite older ones)
    edges_to_remove = []
    for u, v in citation_graph.edges():
        u_year = citation_graph.nodes[u].get('year', 2020)
        v_year = citation_graph.nodes[v].get('year', 2020)
        if u_year < v_year:  # Remove "backwards" citations
            edges_to_remove.append((u, v))

    citation_graph.remove_edges_from(edges_to_remove)

    citation_info = {
        "source": "Simulated Citation Network",
        "description": "Directed graph simulating academic paper citations",
        "parameters": {"n": 200, "temporal_structure": True}
    }

    return citation_graph, citation_info


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

    # Analyze graph
    metrics, G = evaluator.analyze_graph(
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

    # Generate macro-graph visualizations
    visualizer = CommunityVisualizer(G)
    macro_paths = visualizer.generate_all_macro_visualizations(
        communities_dict, output_path, threshold=3
    )

    log_substep("Macro-graph visualizations generated:")
    for algorithm, paths in macro_paths.items():
        logger.info(f"   • {algorithm.title()}:")
        logger.info(f"     - Weighted: {paths['weighted']}")
        logger.info(f"     - Thresholded: {paths['thresholded']}")

    # Generate community keywords
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
            # Show top 5 with counts
            top_keywords = [f"{kw[0]}({kw[1]})" for kw in keywords[:5]]
            logger.info(
                f"      Community {comm_id}: {', '.join(top_keywords)}")

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
