from .semantic_extractor import load_semantic_extractor
from .indexer import load_indexer
from .searcher import load_searcher
from .common import setup_paths, registry, FAISS_DIR, MAPPING_DIR
from pathlib import Path

setup_paths()

# extractor_name = "beit3"
# indexer_name = "gpu-index-flat-l2"
# searcher_name = "single-semantic-search"

# faiss_path = Path(registry.get_path("faiss")) / \
#     f"faiss_index_{extractor_name}.faiss"
# mapping_path = Path(registry.get_path("embeds")) / \
#     f"image_embeddings_{extractor_name}.bin"
# mapping_path = Path(registry.get_path("mapping")) / \
#     f"mapping_{extractor_name}.json"

# extractor = load_semantic_extractor(extractor_name)
# indexer = load_indexer(indexer_name, extractor=extractor)

# indexer.load(faiss_path, mapping_path)

# searcher = load_searcher(searcher_name, extractor, indexer)

# query = "a dog"

# print(searcher.search(query, top_k=10))

extractor_name = ["align", "apple-clip", "coca-clip"]
indexer_name = ["gpu-index-flat-l2", "gpu-index-flat-l2", "gpu-index-flat-l2"]
searcher_name = "fusion-semantic-search"

extractors = [load_semantic_extractor(name) for name in extractor_name]
indexers = [
    load_indexer(name, extractor) for name, extractor in zip(indexer_name, extractors)
]

for indexer, name in zip(indexers, extractor_name):
    faiss_path = FAISS_DIR / f"faiss_index_{name}.faiss"
    mapping_path = MAPPING_DIR / f"mapping_{name}.json"
    indexer.load(faiss_path, mapping_path)


searcher = load_searcher("fussion-semantic-search", extractors, indexers)

query = "A man in a green collared shirt and sunglasses walks outdoors with a relaxed smile, framed against a backdrop of yellow cable cars gliding over lush green mountains. White support pillars and ornamental plants flank the scene under a clear sky dotted with soft clouds."

print(searcher.search(query, top_k=10))
