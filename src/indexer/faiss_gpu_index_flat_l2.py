import gc
import json
from pathlib import Path

import faiss
import numpy as np
import rmm
import torch
from PIL import Image
from tqdm import tqdm

from src.common import registry
from src.semantic_extractor import SemanticExtractor

from .base import Indexer
from .utils import parse_frames_info


@registry.register_indexer("gpu-index-flat-l2")
class FaissGpuIndexFlatL2(Indexer):
    def __init__(
        self,
        extractor: SemanticExtractor,
        device: str = "cuda",
    ):
        """
        GPU-based FAISS IndexFlatL2 for exact similarity search.

        Args:
            extractor (SemanticExtractor): Module to extract features for images/text
            device (str): 'cuda' or 'cpu' (defaults to 'cuda' if available)
        """
        super().__init__()
        self.extractor = extractor
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"

    def _init_gpu_flat_index(self, dim: int, pool_size: int = 2**30, device_id: int = 0):
        """
        Initialize a FAISS GpuIndexFlatL2 with RMM memory pool.

        Args:
            dim (int): Dimension of feature vectors
            pool_size (int): Initial RMM memory pool size (default ~1GB)
            device_id (int): GPU device id (default 0)

        Returns:
            faiss.GpuIndexFlatL2: GPU index ready for adding vectors
        """
        try:
            # Initialize an RMM PoolMemoryResource for efficient GPU memory usage
            self.pool = rmm.mr.PoolMemoryResource(
                rmm.mr.CudaMemoryResource(), initial_pool_size=pool_size
            )

            # Set this pool as the current device resource
            rmm.mr.set_per_device_resource(device_id, self.pool)

            # Create FAISS StandardGpuResources
            res = faiss.StandardGpuResources()

            # Disable FAISS default temporary memory (we use RMM instead)
            res.noTempMemory()

            # Create a GPU config (could be extended to select device)
            config = faiss.GpuIndexFlatConfig()

            # Create a CPU IndexFlatL2 (exact search)
            index_cpu = faiss.IndexFlatL2(dim)

            # Transfer the CPU index to GPU
            index_gpu = faiss.GpuIndexFlatL2(res, index_cpu, config)

            return index_gpu

        except Exception as e:
            raise RuntimeError(f"Error creating index: {e}")

    def _get_image_from_path(self, paths: list[str]) -> list[Image.Image]:
        """
        Load images from a list of file paths.

        Args:
            paths (list[str]): List of image file paths

        Returns:
            list[Image.Image]: List of loaded PIL images in RGB format
        """
        images = []
        for path in tqdm(paths, desc="Image Loading..."):
            img = Image.open(path).convert("RGB")
            images.append(img)
        return images

    def build(self, folder_path: str, batch_size: int = 1000, verbose: bool = True):
        """
        Build FAISS index from a folder of images using batch processing to save memory.
        Steps:
        1. Parse mapping info (image paths)
        2. Load images in batches
        3. Extract features for each batch
        4. Initialize GPU FAISS index if not already done
        5. Add features to the index batch by batch
        """

        # 1. Parse mapping info (e.g., list of dicts with image paths)
        self.mapping = parse_frames_info(folder_path)
        print(f"Num images: {len(self.mapping)}")

        # 2. Initialize the FAISS index later (when we know feature dimension)
        self.index_gpu = None

        # Temporary list to store paths for current batch
        batch_paths = []

        for idx, item in enumerate(self.mapping):
            batch_paths.append(item['path'])  # Assume mapping contains 'path'

            # When batch is full or at the last image
            if len(batch_paths) == batch_size or idx == len(self.mapping) - 1:
                # 3. Load current batch of images
                batch_images = self._get_image_from_path(batch_paths)

                # 4. Extract features for this batch
                batch_features = self.extractor.extract_image_features(
                    batch_images, batch_size=64)
                batch_features = batch_features.astype(np.float32)

                # 5. Initialize FAISS index if first batch
                if self.index_gpu is None:
                    dim = batch_features.shape[1]
                    self.index_gpu = self._init_gpu_flat_index(dim=dim)

                # 6. Add batch features to FAISS index
                self.index_gpu.add(batch_features)

                if not hasattr(self, "image_features") or self.image_features is None:
                    self.image_features = batch_features
                else:
                    self.image_features = np.concatenate(
                        [self.image_features, batch_features], axis=0)

                if verbose:
                    print("="*30)
                    print(
                        f"Processed iamges: {len(self.image_features)}/{len(self.mapping)}")
                    print(f"Embed shape: {self.image_features.shape}")
                    print("="*30)

                # 7. Release memory for this batch
                del batch_images, batch_features
                batch_paths.clear()

                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

                gc.collect()

                # 8. Auto Save
                # if auto_save:
                #     self.save_image_embed()

        print(f"Total indexed images: {self.index_gpu.ntotal}")

    def load(self, faiss_path: Path, mapping_path: Path):
        """
        Load FAISS index and mapping from disk.

        Args:
            faiss_path (Path): Path to .faiss file
            mapping_path (Path): Path to mapping.json

        Returns:
            None
        """
        # Load mapping
        with open(mapping_path, "r", encoding="utf-8") as f:
            self.mapping = json.load(f)

        # Load FAISS index (CPU)
        self.index_cpu = faiss.read_index(str(faiss_path))

        co = faiss.GpuClonerOptions()
        co.use_cuvs = True

        # Optionally: move to GPU
        res = faiss.StandardGpuResources()
        self.index_gpu = faiss.index_cpu_to_gpu(res, 0, self.index_cpu, co)

        print(
            f"Loaded FAISS index from {faiss_path} with {self.index_gpu.ntotal} vectors")
        print(
            f"Loaded mapping from {mapping_path} with {len(self.mapping)} items")

    def search(self, query, top_k: int = 5):
        """
        Search top-k images by text query.

        Args:
            query (str or list): Text query
            top_k (int): Number of results to return

        Returns:
            list: Top-k mapping entries
        """
        query_embed = self.extractor.extract_text_features(
            query).astype(np.float32)
        _, idx = self.index_gpu.search(query_embed, top_k)

        docs = [self.mapping[idx] for idx in idx[0]]
        return docs

    def save_image_embed(self, path: Path):
        """
        Save image feature embeddings to binary file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image_features.astype(np.float32).tofile(path)
        print(
            f"Saved embedding to {path} with shape {self.image_features.shape}")

    def save_faiss_index(self, path: Path):
        """
        Save FAISS index to .faiss file.
        """
        # Move GPU index to CPU before saving
        path.parent.mkdir(parents=True, exist_ok=True)
        index_cpu = faiss.index_gpu_to_cpu(self.index_gpu)
        faiss.write_index(index_cpu, str(path))
        print(
            f"Saved FAISS index to {path} with {len(self.image_features)} vectors")

    def save_mapping(self, path: Path):
        """
        Save mapping to JSON file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f, ensure_ascii=False, indent=4)
        print(f"Saved mapping to {path}")
