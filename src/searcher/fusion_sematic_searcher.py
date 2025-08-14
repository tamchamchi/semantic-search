from ..common import registry
from ..indexer import Indexer, reciprocal_rank_fusion
from ..semantic_extractor import SemanticExtractor
from .base import Searcher


@registry.register_searcher("fussion-semantic-search")
class FussionSemanticSearch(Searcher):
    def __init__(
        self, extractors: list[SemanticExtractor], indexers: list[Indexer]
    ):
        """
        Initialize a FusionSemanticSearch instance.

        Args:
            extractor_indexer_pairs (List[Tuple[Extractor, Indexer]]):
                A list of (extractor, indexer) pairs.
                Each pair defines one semantic search pipeline whose
                results will be fused together.
        """
        super().__init__()
        self.extractors = extractors
        self.indexers = indexers

    def _run_text_search(self, args) -> list[int]:
        extractor, indexer, query, top_k = args
        # Extract semantic embedding
        embedding = extractor.extract_text_features(query)
        # Search in the indexer
        _, idx = indexer.index_gpu.search(embedding, top_k)
        return idx[0].tolist()

    def _run_image_search(self, args) -> list[int]:
        extractor, indexer, query, top_k = args
        # Extract semantic embedding
        embedding = extractor.extract_image_features(query)
        # Search in the indexer
        _, idx = indexer.index_gpu.search(embedding, top_k)
        return idx[0].tolist()

    def search(self, query, top_k, return_idx: bool = False, mode: str = "text"):
        tasks = [
            (extractor, indexer, query, top_k)
            for extractor, indexer in zip(self.extractors, self.indexers)
        ]

        if mode == "text":
            results = [self._run_text_search(task) for task in tasks]
        elif mode == "image":
            results = [self._run_image_search(task) for task in tasks]
        else:
            raise KeyError(f"Unsupported mode: {mode}")

        fusion_res = reciprocal_rank_fusion(results)

        idxs = [idx for idx, _ in fusion_res[:top_k]]

        if return_idx:
            return [idxs]

        return [
            self.indexers[0].mapping[i] for i in idxs
        ]
