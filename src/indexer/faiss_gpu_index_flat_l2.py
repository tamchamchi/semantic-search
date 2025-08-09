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

    def _init_gpu_flat_index(
        self,
        dim: int,
        pool_size: int = 2**30,
        device_id: int = 0,
        index_cpu: faiss.Index = None,
    ):
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
            # Init RMM pool
            self.pool = rmm.mr.PoolMemoryResource(
                rmm.mr.CudaMemoryResource(), initial_pool_size=pool_size
            )
            rmm.mr.set_per_device_resource(device_id, self.pool)

            # GPU resources
            res = faiss.StandardGpuResources()
            res.noTempMemory()

            # Clone options
            co = faiss.GpuClonerOptions()
            co.use_cuvs = True

            if index_cpu is None:
                if dim is None:
                    raise ValueError("Need dim if index_cpu is not provided")
                index_cpu = faiss.IndexFlatL2(dim)

            # Move CPU index to GPU
            index_gpu = faiss.index_cpu_to_gpu(res, device_id, index_cpu, co)
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

    def _compute_pool_size(self, embed_size: int, num_embeds: int) -> int:
        return num_embeds * embed_size * 4

    def build(
        self,
        folder_path: str,
        batch_size: int = 1000,
        verbose: bool = True,
        fast_test: bool = False,
    ):
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
        index_cpu = None
        dim = None

        # Temporary list to store paths for current batch
        batch_paths = []

        for idx, item in enumerate(self.mapping):
            batch_paths.append(item["path"])  # Assume mapping contains 'path'

            # When batch is full or at the last image
            if len(batch_paths) == batch_size or idx == len(self.mapping) - 1:
                # 3. Load current batch of images
                batch_images = self._get_image_from_path(batch_paths)

                # 4. Extract features for this batch
                batch_features = self.extractor.extract_image_features(
                    batch_images, batch_size=batch_size
                )
                batch_features = batch_features.astype(np.float32)

                # 5. Initialize FAISS index if first batch
                if index_cpu is None:
                    dim = batch_features.shape[1]
                    index_cpu = faiss.IndexFlatL2(dim)

                # 6. Add batch features to FAISS index
                index_cpu.add(batch_features)

                if not hasattr(self, "image_features") or self.image_features is None:
                    self.image_features = batch_features
                else:
                    self.image_features = np.concatenate(
                        [self.image_features, batch_features], axis=0
                    )

                if verbose:
                    print("=" * 30)
                    print(
                        f"Processed iamges: {len(self.image_features)}/{len(self.mapping)}"
                    )
                    print(f"Embed shape: {self.image_features.shape}")
                    print("=" * 30)

                # 7. Release memory for this batch
                del batch_images, batch_features
                batch_paths.clear()

                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

                gc.collect()

                if fast_test:
                    break

        pool_size = self._compute_pool_size(dim, len(self.mapping))
        self.index_gpu = self._init_gpu_flat_index(
            dim=dim, pool_size=pool_size, index_cpu=index_cpu)

        # Clear index_cpu to free CPU memory
        del index_cpu
        gc.collect()

        print(f"Total indexed images: {self.index_gpu.ntotal}")

    def load(self, faiss_path: Path, mapping_path: Path, pool_size: int = 2**30, device_id: int = 0):
        """
        Load a FAISS index and its corresponding mapping from disk, 
        and move the index to GPU with an RMM-managed memory pool.

        Args:
            faiss_path (Path): Path to the saved FAISS index (.faiss file).
            mapping_path (Path): Path to the mapping JSON file containing metadata.
            pool_size (int, optional): Initial size (in bytes) for the RMM GPU memory pool. 
                Defaults to 1 GB (2**30).
            device_id (int, optional): GPU device ID to load the index onto. Defaults to 0.

        Returns:
            None

        Raises:
            FileNotFoundError: If either `faiss_path` or `mapping_path` does not exist.
            RuntimeError: If there is an error during FAISS index loading or GPU transfer.
        """
        # Load mapping
        with open(mapping_path, "r", encoding="utf-8") as f:
            self.mapping = json.load(f)

        # Load FAISS index (CPU)
        self.index_cpu = faiss.read_index(str(faiss_path))

        self.index_gpu = self._init_gpu_flat_index(
            pool_size=pool_size,
            device_id=device_id,
            index_cpu=self.index_cpu
        )

        print(
            f"Loaded FAISS index from {faiss_path} with {self.index_gpu.ntotal} vectors"
        )
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
