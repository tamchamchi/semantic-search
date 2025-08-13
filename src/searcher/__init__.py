from .base import Searcher
from .single_semantic_searcher import SingleSemanticSearcher
from ..common import registry

__all__ = [
    "Searcher",
    "SingleSemanticSearcher"
]


def load_searcher(name, extractor=None, indexer=None):
    searcher_cls = registry.get_searcher_cls(name)
    return searcher_cls(extractor, indexer)
