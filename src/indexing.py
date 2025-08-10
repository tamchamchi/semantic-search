import os
from pathlib import Path

from dotenv import load_dotenv

from src.common import setup_paths
from src.indexer import load_indexer
from src.semantic_extractor import load_semantic_extractor

setup_paths()
load_dotenv()

ACMM_DIR = Path(os.getenv("ACMM_DATA_DIR"))
SEMANTIC_FOLDER = Path(ACMM_DIR, "semantic")
FRAME_DIR = ACMM_DIR / "frames"


def indexing(indexer_name, extractor_name, batch_size: int = 1000):
    mapping_file = SEMANTIC_FOLDER / f"mapping_{extractor_name}.json"
    embed_file = SEMANTIC_FOLDER / f"images_embeddings_{extractor_name}.bin"
    faiss_file = SEMANTIC_FOLDER / f"faiss_index_{extractor_name}.faiss"

    extractor = load_semantic_extractor(extractor_name)
    indexer = load_indexer(indexer_name, extractor=extractor)

    indexer.build(FRAME_DIR, batch_size=batch_size)
    query = "Nighttime scene of a serious railway accident in Dong Nai, Vietnam, where a blue-and-red passenger train has collided with a white pickup truck at a railway crossing."
    print(indexer.search(query, top_k=5))

    # Save
    indexer.save_image_embed(embed_file)
    indexer.save_faiss_index(faiss_file)
    indexer.save_mapping(mapping_file)


if __name__ == "__main__":
    indexer_name = "gpu-index-flat-l2"
    extractor_name = "align"

    indexing(indexer_name, extractor_name, batch_size=128)
