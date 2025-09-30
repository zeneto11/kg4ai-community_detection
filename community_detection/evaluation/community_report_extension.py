# community_detection/evaluation/community_report_extension.py

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

from community_detection.utils.logger import log_substep


class CommunityReportExtension:
    """
    Extension class to append community analysis section to existing graph analysis reports.

    This class reads existing community detection results and appends a comprehensive
    Communities section to the existing markdown report without replacing it.
    """

    def __init__(self, output_dir: Path, run_id: str):
        """
        Initialize the community report extension.

        Args:
            output_dir: Directory containing the run outputs
            run_id: Run identifier for file naming
        """
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.logger = logging.getLogger(__name__)

        # File paths
        self.report_path = self.output_dir / f"{run_id}_report.md"
        self.community_results_path = self.output_dir / \
            f"{run_id}_community_results.json"
        self.community_keywords_path = self.output_dir / \
            f"{run_id}_community_keywords.json"
        self.comparison_csv_path = self.output_dir / f"{run_id}_comparison.csv"
        self.plots_dir = self.output_dir / "plots"

    def extend_report(self) -> bool:
        """
        Extend the existing report with community analysis section.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if required files exist
            if not self._validate_required_files():
                return False

            # Load community data
            community_results = self._load_community_results()
            community_keywords = self._load_community_keywords()
            comparison_data = self._load_comparison_data()

            # Generate community section content
            community_section = self._generate_community_section(
                community_results, community_keywords, comparison_data
            )

            # Append to existing report
            self._append_to_report(community_section)

            log_substep(f"Community section appended to {self.report_path}")
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to extend report with community analysis: {e}")
            return False

    def _validate_required_files(self) -> bool:
        """Validate that all required files exist."""
        required_files = [
            self.report_path,
            self.community_results_path,
            self.community_keywords_path,
            self.comparison_csv_path
        ]

        missing_files = [f for f in required_files if not f.exists()]

        if missing_files:
            self.logger.warning(
                f"Missing required files: {[str(f) for f in missing_files]}")
            return False

        return True

    def _load_community_results(self) -> Dict[str, Any]:
        """Load community detection results."""
        with open(self.community_results_path, 'r') as f:
            return json.load(f)

    def _load_community_keywords(self) -> Dict[str, Any]:
        """Load community keywords."""
        with open(self.community_keywords_path, 'r') as f:
            return json.load(f)

    def _load_comparison_data(self) -> List[Dict[str, Any]]:
        """Load algorithm comparison data."""
        data = []
        with open(self.comparison_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                for key, value in row.items():
                    try:
                        if '.' in value:
                            row[key] = float(value)
                        else:
                            row[key] = int(value)
                    except (ValueError, TypeError):
                        # Keep as string if conversion fails
                        pass
                data.append(row)
        return data

    def _generate_community_section(self, community_results: Dict[str, Any],
                                    community_keywords: Dict[str, Any],
                                    comparison_data: List[Dict[str, Any]]) -> str:
        """Generate the complete community analysis section."""

        section = "\n\n---\n\n"
        section += "# 🏘️ Community Detection Analysis\n\n"
        section += "*Comprehensive analysis of community structures detected by different algorithms*\n\n"

        # Algorithm comparison
        section += self._generate_algorithm_comparison(comparison_data)

        # Detailed analysis per algorithm
        section += self._generate_detailed_analysis(
            community_results, community_keywords)

        # Summary insights
        section += self._generate_summary_insights(
            community_results, comparison_data)

        return section

    def _generate_algorithm_comparison(self, comparison_data: List[Dict[str, Any]]) -> str:
        """Generate algorithm comparison analysis."""

        section = "## 🔍 Algorithm Performance Comparison\n\n"

        # Find best performers
        best_modularity = max(comparison_data, key=lambda x: x['modularity'])
        most_communities = max(
            comparison_data, key=lambda x: x['num_communities'])

        section += "### Key Performance Insights\n\n"
        section += f"- **🏆 Highest Modularity:** {best_modularity['algorithm'].title()} "
        section += f"({best_modularity['modularity']:.4f})\n"
        section += f"- **🔢 Most Communities Found:** {most_communities['algorithm'].title()} "
        section += f"({int(most_communities['num_communities'])} communities)\n"

        section += "\n"

        # Performance metrics table
        section += "### Detailed Performance Metrics\n\n"
        section += "| Algorithm | Communities | Modularity | Homophily* | Avg Conductance | Avg TPR | Avg Clustering |\n"
        section += "|-----------|-------------|------------|-----------|-----------------|---------|----------------|\n"

        for row in comparison_data:
            algorithm = row['algorithm'].title()
            communities = int(row['num_communities'])
            modularity = f"{row['modularity']:.4f}"
            homophily = f"{row['homophily']:.4f}" if 'homophily' in row else "N/A"
            conductance = f"{row['avg_conductance']:.4f}" if 'avg_conductance' in row else "N/A"
            tpr = f"{row['avg_tpr']:.4f}" if 'avg_tpr' in row else "N/A"
            clustering = f"{row['avg_clustering']:.4f}" if 'avg_clustering' in row else "N/A"

            section += f"| **{algorithm}** | {communities} | {modularity} | {homophily} | {conductance} | {tpr} | {clustering} |\n"

        section += "\n"
        return section

    def _generate_detailed_analysis(self, community_results: Dict[str, Any],
                                    community_keywords: Dict[str, Any]) -> str:
        """Generate detailed analysis for each algorithm."""

        section = "## 🔎 Detailed Community Analysis\n\n"

        # Get algorithms from results (excluding metadata)
        algorithms = [k for k in community_results.keys()
                      if k not in ['run_id', 'timestamp', 'graph_info']]

        for algorithm in algorithms:
            section += self._generate_algorithm_details(
                algorithm, community_results[algorithm],
                community_keywords.get(algorithm, {})
            )

        return section

    def _generate_algorithm_details(self, algorithm: str, results: Dict[str, Any],
                                    keywords: Dict[str, Any]) -> str:
        """Generate detailed analysis for a specific algorithm."""

        section = f"### {algorithm.title()} Analysis\n\n"

        # --- Community Size Statistics ---
        metrics = results.get('metrics', {})
        size_stats = metrics.get('community_size_stats', {})
        if size_stats:
            section += "#### Community Size Statistics\n\n"
            section += "| Metric | Value |\n"
            section += "|--------|-------|\n"
            section += f"| **Total Communities** | {size_stats.get('num_communities', 0)} |\n"
            section += f"| **Largest Community** | {size_stats.get('max_size', 0)} nodes |\n"
            section += f"| **Smallest Community** | {size_stats.get('min_size', 0)} nodes |\n"
            section += f"| **Average Size** | {size_stats.get('mean_size', 0):.1f} nodes |\n"
            section += f"| **Median Size** | {size_stats.get('median_size', 0):.1f} nodes |\n"
            section += f"| **Singleton Communities** | {size_stats.get('num_singleton', 0)} |\n"
            section += "\n"

        # --- Community Size Distribution Plot ---
        dist_plot = self.plots_dir / f"{algorithm}_community_size_barplot.png"
        if dist_plot.exists():
            section += "#### Community Size Distribution\n\n"
            section += f"![Community Size Distribution](plots/{dist_plot.name})\n\n"
            section += "_Distribution of community sizes showing the scale and hierarchy of detected communities._\n\n"

        # --- Quality Metrics ---
        section += "#### Quality Metrics\n\n"
        section += "| Metric | Value | Description |\n"
        section += "|--------|-------|-------------|\n"
        for key, name, description in [
            ('modularity', 'Modularity', 'Network division quality (higher = better)'),
            ('homophily', 'Homophily', 'Tendency of similar nodes to cluster'),
            ('avg_conductance', 'Avg Conductance',
             'Community boundary quality (lower = better)'),
            ('avg_tpr', 'Avg TPR', 'True positive rate for community detection'),
            ('avg_clustering', 'Avg Clustering',
             'Average clustering coefficient within communities')
        ]:
            if key in metrics:
                section += f"| **{name}** | {metrics[key]:.4f} | {description} |\n"
        section += "\n"

        # --- Macrograph Plots ---
        weighted_plot = self.plots_dir / f"{algorithm}_macrograph_weighted.png"
        if weighted_plot.exists():
            section += "#### Weighted Community Macrograph\n\n"
            section += f"![Weighted Community Macrograph](plots/{weighted_plot.name})\n\n"
            section += "_Weighted community interaction network showing connection strengths between communities._\n\n"

        thresholded_plot = self.plots_dir / \
            f"{algorithm}_macrograph_thresholded.png"
        if thresholded_plot.exists():
            section += "#### Thresholded Community Macrograph\n\n"
            section += f"![Thresholded Community Macrograph](plots/{thresholded_plot.name})\n\n"
            section += "_Simplified community network showing only significant inter-community connections._\n\n"

        # --- Top Communities Table ---
        if keywords:
            section += self._generate_top_communities_table(keywords, limit=10)

        return section

    def _generate_top_communities_table(self, keywords: Dict[str, Any], limit: int = 10) -> str:
        """Generate table of top communities with their keywords."""

        section = f"#### Top {limit} Communities by Size\n\n"
        section += "| Rank | Community ID | Community Name | Size | Keywords |\n"
        section += "|------|--------------|----------------|------|----------|\n"

        # Sort communities by relevance/size (use community name length as proxy)
        sorted_communities = sorted(
            keywords.items(),
            key=lambda x: x[1].get('node_count', 0),
            reverse=True
        )[:limit]

        for rank, (comm_id, comm_data) in enumerate(sorted_communities, 1):
            comm_name = comm_data.get('comm_name', f'Community {comm_id}')
            size = comm_data.get('node_count', 0)
            keywords_list = comm_data.get('keywords', [])

            # Extract top 5 keywords
            top_keywords = [kw[0]
                            for kw in keywords_list[:5]] if keywords_list else []
            keywords_str = ', '.join(
                top_keywords) if top_keywords else 'No keywords'

            section += f"| {rank} | **C{comm_id}** | **{comm_name}** | `{size}` | `{keywords_str}` |\n"

        section += "\n"
        return section

    def _generate_summary_insights(self, community_results: Dict[str, Any],
                                   comparison_data: List[Dict[str, Any]]) -> str:
        """Generate summary insights and recommendations."""

        section = "## 💡 Summary Insights\n\n"

        # Algorithm recommendations
        section += "### Algorithm Selection Guide\n\n"

        # Analyze results to provide recommendations
        algorithms = [k for k in community_results.keys()
                      if k not in ['run_id', 'timestamp', 'graph_info']]

        recommendations = []

        for algorithm in algorithms:
            results = community_results[algorithm]
            num_communities = results.get('num_communities', 0)
            metrics = results.get('metrics', {})
            modularity = metrics.get('modularity', 0)

            if modularity > 0.4:
                recommendations.append(
                    (algorithm, "High-quality community structure detected", "🏆"))
            elif num_communities > 10:
                recommendations.append(
                    (algorithm, "Good for finding fine-grained communities", "🔍"))
            elif num_communities < 5:
                recommendations.append(
                    (algorithm, "Good for high-level community overview", "🌐"))
            else:
                recommendations.append(
                    (algorithm, "Balanced community detection approach", "⚖️"))

        for algorithm, recommendation, icon in recommendations:
            section += f"- {icon} **{algorithm.title()}:** {recommendation}\n"

        section += "\n"

        # Best practices
        section += "### Key Findings\n\n"

        # Calculate overall statistics
        total_algorithms = len(algorithms)
        modularity_values = [row['modularity'] for row in comparison_data]
        avg_modularity = sum(modularity_values) / len(modularity_values)
        modularity_std = (sum((x - avg_modularity) **
                          2 for x in modularity_values) / len(modularity_values)) ** 0.5

        section += f"- **Consistency:** Tested {total_algorithms} algorithms with average modularity of {avg_modularity:.4f} (±{modularity_std:.4f})\n"

        # Find consensus
        community_counts = [row['num_communities'] for row in comparison_data]
        if len(set(community_counts)) == 1:
            section += f"- **Consensus:** All algorithms found exactly {community_counts[0]} communities\n"
        else:
            section += f"- **Variation:** Community counts range from {min(community_counts)} to {max(community_counts)}\n"

        # Quality assessment
        high_quality_algos = [row['algorithm']
                              for row in comparison_data if row['modularity'] > 0.3]
        if high_quality_algos:
            section += f"- **Quality:** {len(high_quality_algos)} algorithm(s) achieved high modularity (>0.3): {', '.join(high_quality_algos)}\n"

        section += "\n---\n\n"
        section += "*Community analysis completed. Report generated automatically by the community detection pipeline.*\n"

        return section

    def _append_to_report(self, community_section: str):
        """Append the community section to the existing report."""

        # Read existing report
        with open(self.report_path, 'r', encoding='utf-8') as f:
            existing_content = f.read()

        # Check if community section already exists
        if "🏘️ Community Detection Analysis" in existing_content:
            self.logger.warning(
                "Community section already exists in report. Skipping append.")
            return

        # Append new section
        updated_content = existing_content + community_section

        # Write back to file
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
