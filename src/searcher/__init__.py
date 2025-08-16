from ..common import registry
from .base import Searcher
from .fusion_sematic_searcher import FussionSemanticSearch
from .single_semantic_searcher import SingleSemanticSearcher

__all__ = [
    "Searcher",
    "SingleSemanticSearcher",
    
    "FussionSemanticSearch"

]


def load_searcher(name, extractor=None, indexer=None):
    searcher_cls = registry.get_searcher_cls(name)
    return searcher_cls(extractor, indexer)
