from .base import SemanticExtractor
from .align_extractor import AlignExtractor
from .beit3 import Beit3Extractor
from .siglip_extractor import SiglipExtractor
from .siglip2_extractor import Siglip2Extractor
from .clip_extractor import CLIPExtractor
from src.common import registry

__all__ = ["SemanticExtractor", "AlignExtractor",
           "Beit3Extractor", "SiglipExtractor",
           "Siglip2Extractor",
           "CLIPExtractor",
           "load_semantic_extractor"]


def load_semantic_extractor(name, config: dict = {}):
    semantic_extractor_cls = registry.get_semantic_extractor_cls(name)
    if config:
        return semantic_extractor_cls.from_config(config)
    return semantic_extractor_cls()
