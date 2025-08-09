import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from src.common import registry
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from transformers import XLMRobertaTokenizer
from timm import create_model

from .base import SemanticExtractor


@registry.register_semantic_extractor("beit3")
class Beit3Extractor(SemanticExtractor):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.model = create_model("beit3_large_patch16_384_retrieval")
        self.model = self.model.to(device)
        self.checkpoint = torch.load(
            '/mnt/mmlab2024nas/anhndt/weight/beit3_large_patch16_384_coco_retrieval.pth')
        self.device = device
        self.tokenizer = XLMRobertaTokenizer(
            "/mnt/mmlab2024nas/anhndt/weight/beit3.spm")
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = default_loader

    def process_image(self, image):
        image = self.transform(image).to(
            self.device)  # Apply transforms and move to device
        return image

    def _load_model(self):
        try:
            model = self.model.load_state_dict(self.checkpoint['model'])
            model.eval().to(self.device)
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading Beit3 model: {e}")

    @torch.no_grad()
    def extract_image_features(self, images, batch_size=32) -> np.ndarray:
        if isinstance(images, Image.Image):
            images = [images]

        features = []

        for i in tqdm(range(0, len(images), batch_size), desc="Extracting Image Features"):
            batch_images = images[i: i + batch_size]

            batch_tensors = [self.process_image(img) for img in batch_images]
            batch_tensor = torch.stack(batch_tensors).to(self.device)

            img_embedding, _ = self.model(batch_tensor, only_infer=True)

            img_embedding = torch.nn.functional.normalize(
                img_embedding, dim=-1)

            features.append(img_embedding.cpu())

        # Gộp toàn bộ feature thành 1 tensor
        feature_tensor = torch.cat(features, dim=0)  # (N, D)
        return feature_tensor.numpy()

    @torch.no_grad()
    def extract_text_features(self, texts, batch_size=32) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        features = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]

            # Tokenize
            inputs = self.tokenizer(batch, return_tensors="pt",
                                    padding=True, truncation=True).to(self.device)
            inputs = {k: v for k, v in inputs.items()}
            # Forward
            _, text_embedding = self.model.forward(
                text_description=inputs['input_ids'], only_infer=True)
            # Model co normalize roi
            # text_embedding = torch.nn.functional.normalize(
            #     text_embedding, dim=-1)

            features.append(text_embedding.cpu())

        feature_tensor = torch.cat(features, dim=0)
        return feature_tensor.numpy()

    @classmethod
    def from_config(cls, config: dict = {}):
        # model_path = config.get("model_path")
        device = config.get("device")

        return cls(device=device)
