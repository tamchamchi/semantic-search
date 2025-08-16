from .base import Indexer
from .faiss_gpu_index_flat_l2 import FaissGpuIndexFlatL2
from ..semantic_extractor import SemanticExtractor
from ..common import registry

__all__ = [
    "load_indexer",
    "Indexer",

    "FaissGpuIndexFlatL2",
]


def load_indexer(name, extractor: SemanticExtractor):
    semantic_indexer_cls = registry.get_indexer_cls(name)
    return semantic_indexer_cls(extractor)
