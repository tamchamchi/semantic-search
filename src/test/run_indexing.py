from src.indexer.faiss_gpu_index_flat_l2 import FaissGpuIndexFlatL2
from src.semantic_extractor import AlignExtractor
# from src.semantic_extractor import Beit3Extractor
from src import get_aic_data
import os
from pathlib import Path

aic_data_path = get_aic_data()

semantic_dir = Path(os.path.join(aic_data_path, "semantic"))

extractor = AlignExtractor()
# extractor = Beit3Extractor()
indexer = FaissGpuIndexFlatL2(extractor=extractor)

indexer.build(aic_data_path)

query = "A group of people stands near a small temple-like structure by the coast, surrounded by wooden stakes and piles of scattered debris."
print(indexer.search(query))

indexer.save_image_embed(semantic_dir / "image_embeddings_align.bin")
indexer.save_faiss_index(semantic_dir / "faiss_index_align.faiss")
indexer.save_mapping(semantic_dir / "mapping_align.json")
