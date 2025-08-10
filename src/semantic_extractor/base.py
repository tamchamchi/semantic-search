from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
from typing import Union
import torch


class SemanticExtractor(ABC):
    @abstractmethod
    def extract_image_features(
        images: Union[list[Image.Image], Image.Image],
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> np.ndarray:
        pass

    @abstractmethod
    def extract_text_features(
        texts: Union[list[str], str],
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> np.ndarray:
        pass

    @abstractmethod
    def get_dim() -> int:
        pass
