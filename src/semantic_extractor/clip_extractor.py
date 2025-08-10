import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

from src.common import registry
from .base import SemanticExtractor


@registry.register_semantic_extractor("clip")
class CLIPExtractor(SemanticExtractor):
    def __init__(self, model_path: str = "openai/clip-vit-large-patch14", device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model_path = model_path
        self.model, self.processor, self.tokenizer = self._load_model()

    def _load_model(self):
        try:
            model = CLIPModel.from_pretrained(self.model_path).to(self.device)
            processor = CLIPProcessor.from_pretrained(self.model_path)
            tokenizer = CLIPTokenizer.from_pretrained(self.model_path)
            model.eval()
            return model, processor, tokenizer
        except Exception as e:
            raise RuntimeError(f"Error loading CLIP model: {e}")

    @torch.no_grad()
    def extract_image_features(self, images, batch_size=32) -> np.ndarray:
        if isinstance(images, Image.Image):
            images = [images]

        features = []

        for i in tqdm(range(0, len(images), batch_size), desc="Image Extracting..."):
            batch = images[i: i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            outputs = self.model.get_image_features(**inputs)
            features.append(outputs.cpu())

        feature_tensors = torch.cat(features, dim=0)
        return feature_tensors.numpy()

    @torch.no_grad()
    def extract_text_features(self, texts, batch_size=32) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        features = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
            outputs = self.model.get_text_features(**inputs)
            features.append(outputs.cpu())

        feature_tensors = torch.cat(features, dim=0)
        return feature_tensors.numpy()

    def get_dim(self) -> int:
        return self.model.config.projection_dim

    @classmethod
    def from_config(cls, config: dict = {}):
        model_path = config.get("model_path", "openai/clip-vit-base-patch32")
        device = config.get("device", "cuda")
        return cls(model_path=model_path, device=device)
