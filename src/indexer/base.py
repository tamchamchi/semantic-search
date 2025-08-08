from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path


class Indexer(ABC):
    @abstractmethod
    def build(folder_path: Union[Path, str], faiss_path: Union[Path, str], mapping_path: Union[Path, str]):
        pass

    @abstractmethod
    def load(folder_path: Union[Path, str], faiss_path: Union[Path, str]):
        pass

    @abstractmethod
    def search(query: Union[str, None], top_k: int = 5):
        pass
