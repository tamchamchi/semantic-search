import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import XLMRobertaTokenizer
from uniml.beit3.modeling_finetune import beit3_large_patch16_384_retrieval
from uniml.beit3.utils import load_model_and_may_interpolate
from src.common import registry, WEIGHT

from .base import SemanticExtractor


@registry.register_semantic_extractor("beit3")
class Beit3Extractor(SemanticExtractor):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        # Path to pretrained model weights
        self.model_weight_path = WEIGHT / "beit3_large_patch16_384_coco_retrieval.pth"

        # Initialize BEiT3 retrieval model without loading weights yet
        self.model = beit3_large_patch16_384_retrieval(pretrained=False)
        
        # Load weights and interpolate if needed
        load_model_and_may_interpolate(
            str(self.model_weight_path), self.model, model_key='model', model_prefix='')
        
        # Move model to target device and set to evaluation mode
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device

        # Initialize tokenizer with sentencepiece model
        self.tokenizer = XLMRobertaTokenizer(WEIGHT / "beit3.spm")
        
        # Image preprocessing pipeline (Resize → ToTensor)
        self.processor = transforms.Compose([
            transforms.Resize(
                (384, 384), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

    def _preprocessing_images(self, images: list[Image.Image]) -> torch.tensor:
        """
        Preprocesses a list of PIL images into batched tensors.
        Returns a tensor of shape (N, 3, 384, 384) ready for model input.
        """
        batch_tensors = []
        failed_count = 0
        
        for i, img in enumerate(images):
            try:
                # Apply preprocessing and add batch dimension
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

    @torch.no_grad()
    def extract_image_features(self, images, batch_size=32) -> np.ndarray:
        """
        Extracts image embeddings from one or multiple PIL images.
        Returns a NumPy array of shape (N, D).
        """
        if isinstance(images, Image.Image):
            images = [images]

        features = []

        for i in tqdm(range(0, len(images), batch_size), desc="Extracting Image Features"):
            # Select batch
            batch_images = images[i: i + batch_size]
            
            # Preprocess batch and move to device
            batch_tensors = self._preprocessing_images(
                batch_images).to(self.device)

            # Forward pass (image only)
            img_embedding, _ = self.model(batch_tensors, only_infer=True)

            # Normalize embeddings to unit length
            img_embedding = torch.nn.functional.normalize(
                img_embedding, dim=-1)

            features.append(img_embedding.cpu())

        # Concatenate all batch features
        feature_tensor = torch.cat(features, dim=0)  # (N, D)
        return feature_tensor.numpy()

    @torch.no_grad()
    def extract_text_features(self, texts, batch_size=32) -> np.ndarray:
        """
        Extracts text embeddings from a string or list of strings.
        Returns a NumPy array of shape (N, D).
        """
        if isinstance(texts, str):
            texts = [texts]

        features = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]

            # Tokenize and move to device
            input_ids = self.tokenizer(batch, return_tensors="pt",
                                       padding=True, truncation=True)["input_ids"].to(self.device)

            # Forward pass (text only)
            _, outputs = self.model(
                text_description=input_ids, only_infer=True)

            # Normalize embeddings
            outputs /= outputs.norm(dim=-1, keepdim=True)

            features.append(outputs.cpu())

        # Concatenate all batch features
        feature_tensor = torch.cat(features, dim=0)
        return feature_tensor.numpy()

    def get_dim(self) -> int:
        """
        Returns the output embedding dimension.
        """
        pass

    @classmethod
    def from_config(cls, config: dict = {}):
        """
        Creates an instance of the extractor from a configuration dictionary.
        """
        device = config.get("device")
        return cls(device=device)
