from .semantic_extractor import load_semantic_extractor
from .indexer import load_indexer
from .searcher import load_searcher
from .common import setup_paths, registry
from pathlib import Path

setup_paths()

extractor_name = "beit3"
indexer_name = "gpu-index-flat-l2"
searcher_name = "single-semantic-search"

faiss_path = Path(registry.get_path("faiss")) / \
    f"faiss_index_{extractor_name}.faiss"
mapping_path = Path(registry.get_path("embeds")) / \
    f"image_embeddings_{extractor_name}.bin"
mapping_path = Path(registry.get_path("mapping")) / \
    f"mapping_{extractor_name}.json"

extractor = load_semantic_extractor(extractor_name)
indexer = load_indexer(indexer_name, extractor=extractor)

indexer.load(faiss_path, mapping_path)

searcher = load_searcher(searcher_name, extractor, indexer)

query = "a dog"

print(searcher.search(query, top_k=10))
