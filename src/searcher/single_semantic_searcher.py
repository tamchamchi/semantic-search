import numpy as np

from ..common import get_image_from_path, reciprocal_rank_fusion, registry
from ..indexer import Indexer
from ..semantic_extractor import SemanticExtractor
from .base import Searcher


@registry.register_searcher("single-semantic-search")
class SingleSemanticSearcher(Searcher):
    def __init__(self, extractor: SemanticExtractor, indexer: Indexer):
        super().__init__()
        self.extractor = extractor
        self.indexer = indexer

    def search(
        self,
        query,
        top_k: int = 5,
        return_idx: bool = False,
        mode: str = "text",
        use_rrf: bool = True,
    ):
        """
        Perform similarity search on text or image queries using FAISS.

        Args:
            query (Union[str, List[str], Image.Image, List[Image.Image]]):
                - Text query (single string or list of strings), OR
                - Path(s) to image(s) (string or list of strings), OR
                - Pre-loaded PIL Image(s).
            top_k (int, optional):
                Number of top results to return. Defaults to 5.
            return_idx (bool, optional):
                If True → return only FAISS indices.
                If False → return mapping entries (metadata). Defaults to False.
            mode (str, optional):
                "text" → search using text embeddings.  
                "image" → search using image embeddings. Defaults to "text".
            use_rrf (bool, optional):
                Whether to apply Reciprocal Rank Fusion (RRF) on results. Defaults to True.
        Returns:
            Union[np.ndarray, List]:
                - If return_idx=True → FAISS indices, shape (n_queries, top_k).  
                - If return_idx=False → List of results (metadata entries).
        """
        if mode == "text":
            # Extract embeddings from text queries
            query_embed = self.extractor.extract_text_features(
                query).astype(np.float32)

            # Ensure embeddings are always 2D for FAISS
            if query_embed.ndim == 1:
                query_embed = query_embed[np.newaxis, :]

        elif mode == "image":
            images = None

            # Case 1: query is a list of string paths
            if isinstance(query, list) and all(isinstance(q, str) for q in query):
                images = get_image_from_path(query)

            # Case 2: query is a single string path
            elif isinstance(query, str):
                images = get_image_from_path([query])

            # Case 3: query is already a list of PIL Images
            elif isinstance(query, list) and all(hasattr(q, "size") for q in query):
                images = query

            else:
                raise ValueError(
                    f"Unsupported query type for mode=image: {type(query)}"
                )

            # Extract embeddings from images
            query_embed = self.extractor.extract_image_features(
                images
            ).astype(np.float32)

            # Ensure embeddings are always 2D for FAISS
            if query_embed.ndim == 1:
                query_embed = query_embed[np.newaxis, :]

        else:
            raise KeyError(f"Unsupported mode: {mode}")

        # Perform nearest-neighbor search in FAISS
        _, idx = self.indexer.index_gpu.search(query_embed, top_k)

        # Case 1: user only wants FAISS indices
        if return_idx:
            return idx

        # Case 2: apply Reciprocal Rank Fusion to merge multiple query results
        if use_rrf:
            rrf = reciprocal_rank_fusion(idx)
            return [self.indexer.mapping[i[0]] for i in rrf[:top_k]]

        # Case 3: map FAISS indices back to metadata (default)
        results = [
            [self.indexer.mapping[i] for i in row]
            for row in idx
        ]
        return results
