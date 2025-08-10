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
    # indexer.load(faiss_file, mapping_file)
    query = [
        "The image shows a group of competitive cyclists during a race. The cyclist in front, wearing bib number 14, is dressed in a white jersey with blue sleeves, red accents, and blue shorts. He is wearing a white helmet with rainbow stripes and blue sunglasses, and appears focused and determined. Right behind him is another cyclist, bib number 15, wearing a similar team outfit with a white helmet and red sunglasses. Both riders are gripping their handlebars tightly, indicating high speed and effort. The race indicator at the bottom left corner shows they are on lap 14 out of 15.",
        "A group of people stands near a small temple-like structure by the coast, surrounded by wooden stakes and piles of scattered debris."
             ]
    print(indexer.search(query, top_k=5, return_idx=True))
    print(indexer.search(query, top_k=5))
    # Save
    indexer.save_image_embed(embed_file)
    indexer.save_faiss_index(faiss_file)
    indexer.save_mapping(mapping_file)


if __name__ == "__main__":
    indexer_name = "gpu-index-flat-l2"
    extractor_name = "beit3"

    indexing(indexer_name, extractor_name, batch_size=64)
