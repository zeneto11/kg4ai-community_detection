import re
from collections import Counter
from typing import Dict, List, Set, Tuple


class SimpleKeywordExtractor:
    """Simple class to extract most frequent words from node titles."""

    def __init__(self):
        # Basic stop words to filter out
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'list'
        }

    def extract_keywords(self, titles: List[str], top_k: int = 20) -> List[Tuple[str, int]]:
        """
        Extract most frequent words from titles.

        Args:
            titles: List of node titles
            top_k: Number of top keywords to return

        Returns:
            List of tuples (word, count) sorted by frequency
        """
        if not titles:
            return []

        # Convert all titles to strings and filter out empty/None values
        text_titles = []
        for title in titles:
            if title is not None:
                # Convert to string and strip whitespace
                title_str = str(title).strip()
                if title_str:  # Only add non-empty strings
                    text_titles.append(title_str)

        # If no valid titles, return empty list
        if not text_titles:
            return []

        # Combine all titles and make lowercase
        text = ' '.join(text_titles).lower()

        # Extract words (letters only, min 3 characters)
        words = re.findall(r'\b[a-z]{3,}\b', text)

        # Remove stop words
        filtered_words = [
            word for word in words if word not in self.stop_words]

        # Count frequencies and return top k
        word_counts = Counter(filtered_words)
        return word_counts.most_common(top_k)

    def generate_community_name(self, keywords: List[Tuple[str, int]], max_words: int = 4) -> str:
        """
        Generate a simple community name from top keywords.

        Args:
            keywords: List of (word, count) tuples
            max_words: Maximum number of words to include in name

        Returns:
            Community name like "Film, Series, Season, Episodes"
        """
        if not keywords:
            return "Unnamed Community"

        # Get top keyword words (without counts)
        top_words = [kw[0] for kw in keywords[:max_words]]

        # Capitalize each word and join with commas
        capitalized_words = [word.title() for word in top_words]
        return ", ".join(capitalized_words)

    def extract_community_keywords(self, communities: List[Set], graph, top_k: int = 10) -> Dict[int, Dict]:
        """
        Extract keywords for each community from node titles.

        Args:
            communities: List of communities (each is a set of node IDs)
            graph: NetworkX graph containing node titles
            top_k: Number of top keywords to return per community

        Returns:
            Dictionary mapping community index to keyword data
        """
        community_keywords = {}

        for idx, community_set in enumerate(communities):
            # Get titles of all nodes in this community
            titles = []
            for node_id in community_set:
                if graph.has_node(node_id):
                    # Adjust this based on your actual node attribute structure
                    title = graph.nodes[node_id].get('title', '')
                    # Convert to string and add if not empty
                    if title is not None:
                        title_str = str(title).strip()
                        if title_str:
                            titles.append(title_str)

            # Extract keywords for this community
            keywords = self.extract_keywords(titles, top_k=top_k)

            # Generate community name from keywords
            community_name = self.generate_community_name(keywords)

            community_keywords[idx] = {
                'comm_name': community_name,
                'keywords': keywords,
                'node_count': len(community_set),
                'titles_processed': len(titles)
            }

        return community_keywords
