import gc
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.common import registry
from src.semantic_extractor import SemanticExtractor

from .base import Indexer
from ..common import parse_frames_info, get_unique_path
from .rmm_manager import RMMManager


@registry.register_indexer("gpu-index-flat-l2")
class FaissGpuIndexFlatL2(Indexer):
    def __init__(
        self,
        extractor: SemanticExtractor,
        device: str = "cuda",
        use_rmm: bool = True,
    ):
        """
        FAISS IndexFlatL2 running on GPU for exact similarity search.

        Args:
            extractor (SemanticExtractor): Feature extraction module for images/text
            device (str): 'cuda' or 'cpu' (default is 'cuda' if available)
            use_rmm (bool): Whether to use RMM for GPU memory management
        """
        super().__init__()
        self.extractor = extractor
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.use_rmm = use_rmm
        self.device_id = 0  # Default to device 0
        self.rmm_pool_size = None
        self.gpu_resources = None
        self.index_gpu = None
        self.index_cpu = None

    def _init_gpu_flat_index(
        self,
        dim: int,
        device_id: int = 0,
        index_cpu: faiss.Index = None,
    ):
        """
        Initialize a FAISS GPU IndexFlatL2 index.

        Args:
            dim (int): Feature dimension
            device_id (int): GPU device ID
            index_cpu (faiss.Index): Optional prebuilt CPU index to transfer to GPU

        Returns:
            faiss.Index: GPU index
        """
        try:
            # Initialize RMM memory pool if enabled
            if self.use_rmm:
                RMMManager.initialize_pool(device_id, self.rmm_pool_size)

            # Create FAISS GPU resources without temporary memory
            self.gpu_resources = faiss.StandardGpuResources()
            self.gpu_resources.noTempMemory()

            co = faiss.GpuClonerOptions()
            co.use_cuvs = True  # Use cuVS if available for acceleration

            if index_cpu is None:
                if dim is None:
                    raise ValueError("Need dim if index_cpu is not provided")
                index_cpu = faiss.IndexFlatL2(dim)

            # Transfer index from CPU to GPU
            index_gpu = faiss.index_cpu_to_gpu(
                self.gpu_resources,
                device_id,
                index_cpu,
                co
            )
            return index_gpu

        except Exception as e:
            raise RuntimeError(f"Error creating index: {e}")

    def _get_image_from_path(self, paths: list[str]) -> list[Image.Image]:
        """
        Load images from given file paths.

        Args:
            paths (list[str]): Image file paths

        Returns:
            list[Image.Image]: List of RGB PIL images
        """
        images = []
        for path in tqdm(paths, desc="Image Loading..."):
            img = Image.open(path).convert("RGB")
            images.append(img)
        return images

    def _compute_pool_size(self, embed_size: int, num_embeds: int) -> int:
        """Compute RMM pool size in bytes."""
        return num_embeds * embed_size * 4  # 4 bytes per float32 value

    def build(
        self,
        folder_path: str,
        batch_size: int = 1000,
        verbose: bool = True,
        fast_test: bool = False,
    ):
        """
        Build a FAISS index from a folder of images using batch processing.

        Steps:
            1. Parse mapping info (paths & metadata)
            2. Load images in batches
            3. Extract features for each batch
            4. Initialize FAISS index on first batch
            5. Add features batch-by-batch
        """
        # 1. Load image mapping information
        self.mapping = parse_frames_info(folder_path)
        print(f"Num images: {len(self.mapping)}")

        index_cpu = None
        dim = None
        batch_paths = []

        # Process images in batches
        for idx, item in enumerate(self.mapping):
            batch_paths.append(item["path"])

            if len(batch_paths) == batch_size or idx == len(self.mapping) - 1:
                # 2. Load batch of images
                batch_images = self._get_image_from_path(batch_paths)

                # 3. Extract features
                batch_features = self.extractor.extract_image_features(
                    batch_images, batch_size=batch_size
                ).astype(np.float32)

                # 4. Create FAISS CPU index on first batch
                if index_cpu is None:
                    dim = batch_features.shape[1]
                    index_cpu = faiss.IndexFlatL2(dim)

                # 5. Add features to CPU index
                index_cpu.add(batch_features)

                # Store all features for optional saving
                if not hasattr(self, "image_features") or self.image_features is None:
                    self.image_features = batch_features
                else:
                    self.image_features = np.concatenate(
                        [self.image_features, batch_features], axis=0
                    )

                if verbose:
                    print("=" * 30)
                    print(f"Processed images: {len(self.image_features)}/{len(self.mapping)}")
                    print(f"Embed shape: {self.image_features.shape}")
                    print("=" * 30)

                # Free batch memory
                del batch_images, batch_features
                batch_paths.clear()

                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()

                if fast_test:
                    break

        # Compute RMM pool size if enabled
        if self.use_rmm:
            self.rmm_pool_size = self._compute_pool_size(dim, len(self.mapping))

        # Transfer index to GPU
        self.index_gpu = self._init_gpu_flat_index(dim=dim, index_cpu=index_cpu)

        # Free CPU index memory
        del index_cpu
        gc.collect()

        print(f"Total indexed images: {self.index_gpu.ntotal}")

    def load(self, faiss_path: Path, mapping_path: Path, device_id: int = 0):
        """
        Load a saved FAISS index and mapping, then move index to GPU.
        """
        with open(mapping_path, "r", encoding="utf-8") as f:
            self.mapping = json.load(f)

        self.index_cpu = faiss.read_index(str(faiss_path))

        self.index_gpu = self._init_gpu_flat_index(
            dim=self.extractor.get_dim(),
            device_id=device_id,
            index_cpu=self.index_cpu
        )

        print(f"Loaded FAISS index from {faiss_path} with {self.index_gpu.ntotal} vectors")
        print(f"Loaded mapping from {mapping_path} with {len(self.mapping)} items")

    def search(self, query, top_k: int = 5, return_idx: bool = False):
        """
        Search for the top-k results for a given text query or list of queries.
        """
        query_embed = self.extractor.extract_text_features(query).astype(np.float32)
        if query_embed.ndim == 1:
            query_embed = query_embed[np.newaxis, :]

        _, idx = self.index_gpu.search(query_embed, top_k)

        if return_idx:
            return idx

        return [[self.mapping[i] for i in row] for row in idx]

    def save_image_embed(self, path: Path):
        """Save image feature embeddings to a binary file."""
        path = get_unique_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image_features.astype(np.float32).tofile(path)
        print(f"Saved embedding to {path} with shape {self.image_features.shape}")

    def save_faiss_index(self, path: Path):
        """Save FAISS index to a .faiss file."""
        path = get_unique_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        index_cpu = faiss.index_gpu_to_cpu(self.index_gpu)
        faiss.write_index(index_cpu, str(path))
        print(f"Saved FAISS index to {path} with {len(self.image_features)} vectors")

    def save_mapping(self, path: Path):
        """Save mapping metadata to JSON."""
        path = get_unique_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f, ensure_ascii=False, indent=4)
        print(f"Saved mapping to {path}")

    def __del__(self):
        """Release FAISS index and GPU resources on object deletion."""
        if hasattr(self, 'index_gpu') and self.index_gpu is not None:
            try:
                self.index_cpu = faiss.index_gpu_to_cpu(self.index_gpu)
            except Exception as e:
                print(e)
            del self.index_gpu
            self.index_gpu = None

        if hasattr(self, 'index_cpu') and self.index_cpu is not None:
            del self.index_cpu
            self.index_cpu = None

        if hasattr(self, 'gpu_resources') and self.gpu_resources is not None:
            del self.gpu_resources
            self.gpu_resources = None

        if self.use_rmm:
            RMMManager.release_pool(self.device_id)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
