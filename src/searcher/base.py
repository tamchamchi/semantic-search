from abc import ABC, abstractmethod


class Searcher(ABC):
    @abstractmethod
    def search(query, top_k: int = 5):
        pass
