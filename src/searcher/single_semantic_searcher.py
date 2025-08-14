import numpy as np

from ..common import registry
from ..indexer import Indexer
from ..semantic_extractor import SemanticExtractor
from .base import Searcher


@registry.register_searcher("single-semantic-search")
class SingleSemanticSearcher(Searcher):
    def __init__(self, extractor: SemanticExtractor, indexer: Indexer):
        super().__init__()
        self.extractor = extractor
        self.indexer = indexer

    def search(self, query, top_k: int = 5, return_idx: bool = False, mode: str = "text"):
        """
        Search top-k results for one or multiple queries.

        Args:
            query (str or list[str]): Single text query or a list of queries
            top_k (int): Number of results to return
            return_idx (bool): If True, return only indices from FAISS. 
                            If False, return mapping entries.

        Returns:
            list: If return_idx=True, returns ndarray of shape (n_queries, top_k)
                If return_idx=False, returns list of list of mapping entries
        """
        if mode == "text":
            # Extract embeddings for one or multiple queries
            query_embed = self.extractor.extract_text_features(
                query).astype(np.float32)
            # Ensure 2D array for FAISS
            if query_embed.ndim == 1:
                query_embed = query_embed[np.newaxis, :]
        elif mode == "image":
            # Extract embeddings for one or multiple queries
            query_embed = self.extractor.extract_image_features(
                query).astype(np.float32)
            # Ensure 2D array for FAISS
            if query_embed.ndim == 1:
                query_embed = query_embed[np.newaxis, :]
        else:
            raise KeyError(f"Unsupported mode: {mode}")

        # Batch search in FAISS
        _, idx = self.indexer.index_gpu.search(
            query_embed, top_k)  # idx: (n_queries, top_k)

        if return_idx:
            return idx

        # Map indices to metadata
        results = [
            [self.indexer.mapping[i] for i in row]
            for row in idx
        ]
        return results
