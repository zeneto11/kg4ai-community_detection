# community_detection/methods/kmeans_detector.py

import logging
import numpy as np
import networkx as nx
from typing import Union, Optional, List, Dict, Any
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import umap
from concurrent.futures import ThreadPoolExecutor
import gc

from .base import CommunityDetector


class KMeansDetector(CommunityDetector):
    """Community detection using K-means clustering on node embeddings.

    This detector uses sentence embeddings of node titles/attributes combined with
    UMAP dimensionality reduction and MiniBatchKMeans clustering. Based on the
    evaluation results showing optimal performance with k=50.

    Args:
        k: Number of clusters. Default is 50 based on evaluation results.
        embedding_model: Sentence transformer model name. Default is 'all-MiniLM-L6-v2'.
        umap_components: Number of UMAP components for dimensionality reduction. Default is 15.
        batch_size: Batch size for MiniBatch K-means. Default is 1024.
        random_state: Random seed for reproducibility. Default is 42.
        max_iter: Maximum iterations for K-means. Default is 300.
        n_init: Number of initialization runs. Default is 10.
        auto_select_k: Whether to automatically select optimal k. Default is False.
        k_range: Range of k values to test if auto_select_k is True.
        use_pca: Whether to use PCA before UMAP for faster processing. Default is True.
        pca_components: Number of PCA components if use_pca is True. Default is 50.
        sampling_strategy: Strategy for large datasets ('stratified', 'random', or None). Default is 'random'.
        max_sample_size: Maximum sample size for evaluation. Default is 10000.
        use_fast_metrics: Whether to use approximated metrics for large datasets. Default is True.
        n_jobs: Number of parallel jobs for embedding generation. Default is -1.
    """

    def __init__(
        self,
        k: int = 50,
        embedding_model: str = 'all-MiniLM-L6-v2',
        umap_components: int = 15,
        batch_size: int = 2048,  # Increased default batch size
        random_state: int = 42,
        max_iter: int = 300,
        n_init: int = 5,  # Reduced from 10 for speed
        auto_select_k: bool = False,
        k_range: List[int] = None,
        use_pca: bool = True,  # New parameter
        pca_components: int = 50,  # New parameter
        sampling_strategy: str = 'random',  # New parameter
        max_sample_size: int = 10000,  # New parameter
        use_fast_metrics: bool = True,  # New parameter
        n_jobs: int = -1,  # New parameter
        **kwargs
    ):
        """Initialize the K-means detector."""
        super().__init__(
            k=k, embedding_model=embedding_model, umap_components=umap_components,
            batch_size=batch_size, random_state=random_state, max_iter=max_iter,
            n_init=n_init, auto_select_k=auto_select_k, k_range=k_range, **kwargs
        )

        self.k = k
        self.embedding_model = embedding_model
        self.umap_components = umap_components
        self.batch_size = batch_size
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self.auto_select_k = auto_select_k
        self.k_range = k_range or [30, 40, 50, 60, 70]  # Reduced range

        # New optimization parameters
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.sampling_strategy = sampling_strategy
        self.max_sample_size = max_sample_size
        self.use_fast_metrics = use_fast_metrics
        self.n_jobs = n_jobs

        self.logger = logging.getLogger(__name__)
        self._embeddings = None
        self._reduced_embeddings = None
        self._evaluation_results = None
        self._sentence_model = None  # Cache the model

    def _extract_node_texts(self, graph: nx.Graph) -> List[str]:
        """Extract text content from graph nodes for embedding.

        Args:
            graph: NetworkX graph object

        Returns:
            List of text strings for each node
        """
        texts = []
        for node in graph.nodes():
            # Try to get title or other text attributes
            node_data = graph.nodes[node]
            text = ""

            # Priority order for text extraction
            if 'title' in node_data:
                text = str(node_data['title'])
            elif 'name' in node_data:
                text = str(node_data['name'])
            elif 'label' in node_data:
                text = str(node_data['label'])
            else:
                # Fallback to node ID as string
                text = str(node)

            texts.append(text)

        return texts

    def _load_sentence_model(self):
        """Load and cache the sentence transformer model."""
        if self._sentence_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers package is required. "
                    "Install with 'pip install sentence-transformers'"
                )

            self.logger.info(
                f"Loading sentence transformer model: {self.embedding_model}")
            self._sentence_model = SentenceTransformer(self.embedding_model)

        return self._sentence_model

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate sentence embeddings for node texts.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of embeddings
        """
        model = self._load_sentence_model()

        self.logger.info(
            f"Generating embeddings for {len(texts)} nodes with batch size {self.batch_size}")

        # Use GPU if available, otherwise CPU
        device = "cuda" if hasattr(model, 'device') and 'cuda' in str(
            model.device) else "cpu"
        print(f"Using device: {device}")

        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            device=device,
            normalize_embeddings=True  # Normalize for better clustering
        )

        self.logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings

    def _apply_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply PCA for initial dimensionality reduction before UMAP.

        Args:
            embeddings: High-dimensional embeddings

        Returns:
            PCA-reduced embeddings
        """
        self.logger.info(
            f"Applying PCA to reduce from {embeddings.shape[1]} to {self.pca_components} dimensions")

        pca = PCA(n_components=self.pca_components,
                  random_state=self.random_state)
        pca_embeddings = pca.fit_transform(embeddings)

        self.logger.info(
            f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        return pca_embeddings

    def _reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce embedding dimensions using PCA + UMAP.

        Args:
            embeddings: High-dimensional embeddings

        Returns:
            Reduced embeddings
        """
        try:
            import umap
        except ImportError:
            raise ImportError(
                "umap-learn package is required. "
                "Install with 'pip install umap-learn'"
            )

        # Apply PCA first if requested (much faster than UMAP alone)
        if self.use_pca and embeddings.shape[1] > self.pca_components:
            embeddings = self._apply_pca(embeddings)

        self.logger.info(
            f"Reducing dimensions with UMAP to {self.umap_components} components")

        # Optimize UMAP parameters for speed
        n_neighbors = min(15, max(2, len(embeddings) // 100)
                          )  # Adaptive n_neighbors

        # For very large datasets, use more aggressive sampling
        if len(embeddings) > 100000:
            self.logger.info(
                "Very large dataset detected, using low-dimensional approximation")
            n_neighbors = min(10, max(2, len(embeddings) // 200))

        reducer = umap.UMAP(
            n_components=self.umap_components,
            random_state=self.random_state,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric='cosine',
            n_jobs=1,  # UMAP parallel processing can be unstable
            low_memory=True  # Use memory-efficient version
        )

        reduced_embeddings = reducer.fit_transform(embeddings)
        self.logger.info(
            f"Reduced embeddings shape: {reduced_embeddings.shape}")

        return reduced_embeddings

    def _sample_for_evaluation(self, embeddings: np.ndarray) -> tuple:
        """Sample embeddings for faster evaluation on large datasets.

        Args:
            embeddings: Full embeddings array

        Returns:
            Tuple of (sampled_embeddings, sample_indices)
        """
        if len(embeddings) <= self.max_sample_size:
            return embeddings, np.arange(len(embeddings))

        self.logger.info(
            f"Sampling {self.max_sample_size} points from {len(embeddings)} for evaluation")

        np.random.seed(self.random_state)
        if self.sampling_strategy == 'random':
            sample_indices = np.random.choice(
                len(embeddings), self.max_sample_size, replace=False)
        elif self.sampling_strategy == 'stratified':
            # Simple stratified sampling - divide into chunks and sample from each
            chunk_size = len(embeddings) // self.max_sample_size
            sample_indices = np.arange(0, len(embeddings), chunk_size)[
                :self.max_sample_size]
        else:
            sample_indices = np.arange(self.max_sample_size)

        return embeddings[sample_indices], sample_indices

    def _evaluate_k_value(self, embeddings: np.ndarray, k: int) -> Dict[str, float]:
        """Evaluate K-means clustering for a specific k value.

        Args:
            embeddings: Reduced embeddings
            k: Number of clusters

        Returns:
            Dictionary of evaluation metrics
        """
        # Use sampling for large datasets
        eval_embeddings, sample_indices = self._sample_for_evaluation(
            embeddings)

        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=min(self.batch_size, len(eval_embeddings)),
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=self.n_init,
            # Ensure sufficient init size
            init_size=min(3 * k, len(eval_embeddings))
        )

        # Fit on sample, predict on full dataset
        sample_labels = kmeans.fit_predict(eval_embeddings)
        full_labels = kmeans.predict(embeddings)

        # Calculate evaluation metrics on sample (much faster)
        if self.use_fast_metrics and len(eval_embeddings) > 5000:
            # For very large samples, use even smaller subset for metrics
            metric_size = min(5000, len(eval_embeddings))
            metric_indices = np.random.choice(
                len(eval_embeddings), metric_size, replace=False)
            metric_embeddings = eval_embeddings[metric_indices]
            metric_labels = sample_labels[metric_indices]
        else:
            metric_embeddings = eval_embeddings
            metric_labels = sample_labels

        silhouette = silhouette_score(metric_embeddings, metric_labels)
        calinski_harabasz = calinski_harabasz_score(
            metric_embeddings, metric_labels)
        davies_bouldin = davies_bouldin_score(metric_embeddings, metric_labels)

        return {
            'k': k,
            'silhouette': silhouette,
            'calinski_harabasz': calinski_harabasz,
            'davies_bouldin': davies_bouldin,
            'labels': full_labels,
            'model': kmeans
        }

    def _evaluate_k_parallel(self, embeddings: np.ndarray, k_values: List[int]) -> List[Dict[str, Any]]:
        """Evaluate multiple k values in parallel.

        Args:
            embeddings: Reduced embeddings
            k_values: List of k values to evaluate

        Returns:
            List of evaluation results
        """
        self.logger.info(f"Evaluating k values in parallel: {k_values}")

        with ThreadPoolExecutor(max_workers=min(len(k_values), 4)) as executor:
            futures = [executor.submit(
                self._evaluate_k_value, embeddings, k) for k in k_values]
            results = [future.result() for future in futures]

        return results

    def _select_optimal_k(self, embeddings: np.ndarray) -> int:
        """Automatically select optimal k value based on evaluation metrics.

        Args:
            embeddings: Reduced embeddings

        Returns:
            Optimal k value
        """
        # Use parallel evaluation for speed
        evaluation_results = self._evaluate_k_parallel(
            embeddings, self.k_range)

        # Log results
        for result in evaluation_results:
            self.logger.info(
                f"K={result['k']}: Silhouette={result['silhouette']:.3f}, "
                f"CH={result['calinski_harabasz']:.1f}, "
                f"DB={result['davies_bouldin']:.3f}"
            )

        # Store evaluation results for inspection
        self._evaluation_results = evaluation_results

        # Simplified ranking approach
        best_silhouette = max(evaluation_results,
                              key=lambda x: x['silhouette'])
        best_ch = max(evaluation_results, key=lambda x: x['calinski_harabasz'])
        best_db = min(evaluation_results, key=lambda x: x['davies_bouldin'])

        # Count votes for each k
        votes = {}
        for result in evaluation_results:
            k = result['k']
            votes[k] = 0
            if result == best_silhouette:
                votes[k] += 1
            if result == best_ch:
                votes[k] += 1
            if result == best_db:
                votes[k] += 1

        # Select k with most votes, tie-break with silhouette score
        max_votes = max(votes.values())
        candidates = [k for k, v in votes.items() if v == max_votes]

        if len(candidates) == 1:
            optimal_k = candidates[0]
        else:
            # Tie-break with silhouette score
            candidate_results = [
                r for r in evaluation_results if r['k'] in candidates]
            optimal_k = max(candidate_results,
                            key=lambda x: x['silhouette'])['k']

        self.logger.info(
            f"Selected optimal k: {optimal_k} (votes: {max_votes})")
        return optimal_k

    def fit(self, graph: Union[nx.Graph, nx.DiGraph]) -> 'KMeansDetector':
        """Detect communities using K-means clustering on node embeddings.

        Args:
            graph: NetworkX graph object

        Returns:
            self: The fitted detector
        """
        self.logger.info("Starting optimized K-means community detection")

        # Extract text content from nodes
        texts = self._extract_node_texts(graph)

        # Generate embeddings
        self._embeddings = self._generate_embeddings(texts)

        # Reduce dimensions
        self._reduced_embeddings = self._reduce_dimensions(self._embeddings)

        # Clean up memory
        if self.use_pca:
            del self._embeddings  # Keep only reduced embeddings to save memory
            gc.collect()

        # Determine optimal k
        if self.auto_select_k:
            optimal_k = self._select_optimal_k(self._reduced_embeddings)
        else:
            optimal_k = self.k

        # Final clustering with optimal k
        self.logger.info(
            f"Running final K-means clustering with k={optimal_k}")
        final_result = self._evaluate_k_value(
            self._reduced_embeddings, optimal_k)

        # Create community mapping
        node_list = list(graph.nodes())
        community_map = {
            node_list[i]: int(final_result['labels'][i])
            for i in range(len(node_list))
        }

        # Store results
        self._set_results(community_map=community_map)

        self.logger.info(
            f"K-means clustering completed: {optimal_k} communities found")
        self.logger.info(f"Final metrics - Silhouette: {final_result['silhouette']:.3f}, "
                         f"CH: {final_result['calinski_harabasz']:.1f}, "
                         f"DB: {final_result['davies_bouldin']:.3f}")

        return self

    def get_embeddings(self) -> Optional[np.ndarray]:
        """Get the generated node embeddings.

        Returns:
            Numpy array of original embeddings or None if not fitted
        """
        return self._embeddings

    def get_reduced_embeddings(self) -> Optional[np.ndarray]:
        """Get the UMAP-reduced embeddings.

        Returns:
            Numpy array of reduced embeddings or None if not fitted
        """
        return self._reduced_embeddings

    def get_evaluation_results(self) -> Optional[List[Dict[str, Any]]]:
        """Get the evaluation results for different k values.

        Returns:
            List of evaluation dictionaries or None if auto_select_k was False
        """
        return self._evaluation_results
