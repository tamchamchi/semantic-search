import argparse
import json
import multiprocessing as mp
import os
import shutil
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv

from src.common import get_unique_path, setup_paths

# --- Setup paths and load environment variables ---
setup_paths()
load_dotenv()

# Directories
ACMM_DIR = Path(os.getenv("ACMM_DATA_DIR"))
FRAME_DIR = ACMM_DIR / "frames_new"
OUTPUT_DIR = Path(os.getenv("PROCESSED_FRAME_DIR"))

MAPPING_DIR = Path(os.getenv("NEW_MAPPING_DIR"))
EMBEDS_DIR = Path(os.getenv("NEW_EMBEDS_DIR"))
FAISS_DIR = Path(os.getenv("NEW_FAISS_DIR"))

PROCESSED_FAISS_DIR = Path(os.getenv("PROCESSED_FAISS_DIR"))
PROCESSED_EMBEDS_DIR = Path(os.getenv("PROCESSED_EMBEDS_DIR"))
PROCESSED_MAPPING_DIR = Path(os.getenv("PROCESSED_MAPPING_DIR"))
PROCESSED_FRAME_DIR = Path(os.getenv("PROCESSED_FRAME_DIR"))

# --- Sequential filter ---


def sequential_filter(vectors, threshold=0.99):
    """
    Sequential filtering: keep the first frame,
    each new frame is compared only with the last kept frame.
    """
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    kept_indices = [0]
    last_vec = vectors[0]
    for i in range(1, len(vectors)):
        sim = np.dot(last_vec, vectors[i])
        if sim < threshold:
            kept_indices.append(i)
            last_vec = vectors[i]
    return kept_indices


def remove_duplicate_frame(embeddings, mapping, threshold=0.99):
    """
    Keep only embeddings and mapping for the selected frames.
    """
    kept_indices = sequential_filter(embeddings, threshold)
    kept_embeddings = embeddings[kept_indices]
    kept_mapping = [mapping[i] for i in kept_indices]
    return kept_embeddings, kept_mapping


def _copy_one(item, frames_dir, output_dir):
    """Copy one file with better error handling"""
    try:
        src_path = Path(item["path"])
        if not src_path.exists():
            return f"ERROR: File not found: {src_path}"

        # Handle both absolute and relative paths
        if src_path.is_absolute():
            try:
                rel_path = src_path.relative_to(frames_dir)
            except ValueError:
                # If not under frames_dir, use just filename
                rel_path = src_path.name
        else:
            rel_path = src_path
            src_path = frames_dir / src_path

        dst_path = output_dir / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        return f"SUCCESS: {src_path.name}"

    except Exception as e:
        return f"ERROR: Failed to copy {item['path']}: {e}"


def create_processed_frames(kept_mapping, frames_dir, output_dir, num_workers=None):
    """Copy files and return statistics"""
    output_dir.mkdir(parents=True, exist_ok=True)
    if num_workers is None:
        # Cap at 8 to avoid overwhelming system
        num_workers = min(mp.cpu_count(), 8)

    print(f"Copying {len(kept_mapping)} files using {num_workers} workers...")

    tasks = [(item, frames_dir, output_dir) for item in kept_mapping]

    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(_copy_one, tasks)

    # Count results
    success_count = sum(1 for r in results if r.startswith("SUCCESS"))
    error_count = len(results) - success_count

    # Print errors
    for r in results:
        if r.startswith("ERROR"):
            print(r)

    stats = f"Copy completed: {success_count} success, {error_count} errors"
    print(stats)
    return stats

# --- Main run function ---


def run(extractor, threshold=0.99, create_frames_dir=False):
    import time

    print(f"🔍 Processing extractor: {extractor}")
    print(f"📊 Similarity threshold: {threshold}")
    print(f"📁 Copy frames: {create_frames_dir}")

    start_time = time.time()

    embeds_path = EMBEDS_DIR / f"images_embeddings_{extractor}.npy"
    mapping_path = MAPPING_DIR / f"mapping_{extractor}.json"

    embeddings = np.load(embeds_path)
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    print("⏱️  Filtering frames...")
    filter_start = time.time()
    kept_embeddings, kept_mapping = remove_duplicate_frame(
        embeddings, mapping, threshold)
    filter_time = time.time() - filter_start

    reduction_ratio = len(kept_mapping) / len(mapping)
    print(f"✅ Filtering completed in {filter_time:.2f}s")
    print(
        f"📉 Kept {len(kept_mapping)}/{len(mapping)} frames ({reduction_ratio:.2%} retention)")

    # Save embeddings
    processed_embeds_path = get_unique_path(
        PROCESSED_EMBEDS_DIR / f"images_embeddings_{extractor}.npy")
    processed_embeds_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(processed_embeds_path, kept_embeddings.astype(np.float32))
    print(
        f"Saved embeddings to {processed_embeds_path} with shape {kept_embeddings.shape}")

    # Save mapping
    processed_mapping_path = get_unique_path(
        PROCESSED_MAPPING_DIR / f"mapping_{extractor}.json")
    processed_mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(processed_mapping_path, "w", encoding="utf-8") as f:
        json.dump(kept_mapping, f, ensure_ascii=False, indent=4)
    print(f"Saved mapping to {processed_mapping_path}")

    # Save FAISS index
    dim = kept_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(kept_embeddings.astype(np.float32))
    processed_faiss_path = get_unique_path(
        PROCESSED_FAISS_DIR / f"faiss_index_{extractor}.faiss")
    processed_faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(processed_faiss_path))
    print(
        f"Saved FAISS index to {processed_faiss_path} with {len(kept_embeddings)} vectors")

    if create_frames_dir:
        print(create_processed_frames(kept_mapping, FRAME_DIR, OUTPUT_DIR))

    total_time = time.time() - start_time
    print(f"🎉 Total processing time: {total_time:.2f}s")


# --- Argparse ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential filter embeddings and build FAISS index")
    parser.add_argument("--extractor", type=str, required=True,
                        help="Extractor name (e.g., 'siglip_so400m')")
    parser.add_argument("--threshold", type=float, default=0.97,
                        help="Cosine similarity threshold (0.0-1.0)")
    parser.add_argument("--create-frames-dir", action="store_true",
                        help="Create processed frame directory with copied files")

    args = parser.parse_args()

    # Validate threshold range
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("Threshold must be between 0.0 and 1.0")

    try:
        run(args.extractor, threshold=args.threshold,
            create_frames_dir=args.create_frames_dir)
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
