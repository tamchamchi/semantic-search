import numpy as np
import torch
import torch.nn.functional as F
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image
from tqdm import tqdm

from src.common import registry

from .base import SemanticExtractor


@registry.register_semantic_extractor("apple-clip")
class AppleCLIPExtractor(SemanticExtractor):
    def __init__(self, model_path: str = "hf-hub:apple/DFN5B-CLIP-ViT-H-14", tokenizer_path: str = "ViT-H-14", device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model, self.processor, self.tokenizer = self._load_model()

    def _load_model(self):
        try:
            model, processor = create_model_from_pretrained(self.model_path, device=self.device)
            tokenizer = get_tokenizer(self.tokenizer_path)
            model.eval()
            return model, processor, tokenizer
        except Exception as e:
            raise RuntimeError(f"Error loading Apple CLIP model: {e}")

    def _preprocessing_images(self, images: list[Image.Image]) -> torch.tensor:
        batch_tensors = []
        failed_count = 0
        for i, img in enumerate(images):
            try:
                # Apply CLIP preprocessing and add batch dimension
                preprocessed = self.processor(img).unsqueeze(0)
                batch_tensors.append(preprocessed)
            except Exception as e:
                print(f"⚠ Failed to preprocess image {i}: {e}")
                failed_count += 1

        if not batch_tensors:
            print("✗ No images could be preprocessed")
            return []

        if failed_count > 0:
            print(f"⚠ {failed_count}/{len(images)} images failed preprocessing")

        return torch.cat(batch_tensors, dim=0)

    def extract_image_features(self, images, batch_size=32) -> np.ndarray:
        if isinstance(images, Image.Image):
            images = [images]

        features = []

        for i in tqdm(range(0, len(images), batch_size), desc="Image Extracting..."):
            batch = images[i: i + batch_size]
            inputs = self._preprocessing_images(batch).to(self.device)

            with torch.no_grad(), torch.amp.autocast('cuda'):
                outputs = self.model.encode_image(inputs)
                outputs = F.normalize(outputs, dim=-1)
                features.append(outputs.cpu())

        feature_tensors = torch.cat(features, dim=0)
        return feature_tensors.numpy()

    def extract_text_features(self, texts, batch_size=32) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        features = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            inputs = self.tokenizer(
                batch, context_length=self.model.context_length).to(self.device)

            with torch.no_grad(), torch.amp.autocast('cuda'):
                outputs = self.model.encode_text(inputs)
                outputs = F.normalize(outputs, dim=-1)
                features.append(outputs.cpu())

        feature_tensors = torch.cat(features, dim=0)
        return feature_tensors.numpy()

    def get_dim(self) -> int:
        return self.model.visual.output_dim

    @classmethod
    def from_config(cls, config: dict = {}):
        model_path = config.get(
            "model_path", "hf-hub:apple/DFN5B-CLIP-ViT-H-14")
        device = config.get("device", "cuda")
        tokenizer_path = config.get("tokenizer_path", "ViT-H-14")
        return cls(model_path=model_path, tokenizer_path=tokenizer_path, device=device)
