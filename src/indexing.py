import os
from pathlib import Path
import argparse
from dotenv import load_dotenv

from src.common import setup_paths
from src.indexer import load_indexer
from src.semantic_extractor import load_semantic_extractor

setup_paths()
load_dotenv()

ACMM_DIR = Path(os.getenv("ACMM_DATA_DIR"))
FRAME_DIR = ACMM_DIR / "frames_new"

MAPPING_DIR = Path(os.getenv("NEW_MAPPING_DIR"))
FAISS_DIR = Path(os.getenv("NEW_FAISS_DIR"))
EMBEDS_DIR = Path(os.getenv("NEW_EMBEDS_DIR"))


def indexing(indexer_name, extractor_name, batch_size: int = 1000, fast_test: bool = False):
    mapping_file = MAPPING_DIR / f"mapping_{extractor_name}.json"
    embed_file = EMBEDS_DIR / f"images_embeddings_{extractor_name}.npy"
    faiss_file = FAISS_DIR / f"faiss_index_{extractor_name}.faiss"

    extractor = load_semantic_extractor(extractor_name)
    indexer = load_indexer(indexer_name, extractor=extractor)

    indexer.build(FRAME_DIR, batch_size=batch_size, fast_test=fast_test)

    query = [
        "Two people lie on a vibrant bed of flowers, smiling happily; the one on the left wears glasses and a deep blue shirt with long red hair spread out, while the one on the right wears a bright green shirt with tied-back red hair, both surrounded by colorful blooms and lush greenery under warm sunlight."
    ]
    print(indexer.search(query, top_k=5, return_idx=True))
    print(indexer.search(query, top_k=5))

    indexer.save_image_embed(embed_file)
    indexer.save_faiss_index(faiss_file)
    indexer.save_mapping(mapping_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index images using a semantic extractor and FAISS.")
    parser.add_argument("--indexer", type=str, default="gpu-index-flat-l2",
                        required=True, help="Name of the indexer (e.g., gpu-index-flat-l2)")
    parser.add_argument("--extractor", type=str, default="align",
                        required=True, help="Name of the extractor (e.g., beit3)")
    parser.add_argument("--batch-size", type=int,
                        default=1000, help="Batch size for indexing")
    parser.add_argument("--fast-test", type=bool, default=False)

    args = parser.parse_args()

    indexing(args.indexer, args.extractor, args.batch_size, args.fast_test)
