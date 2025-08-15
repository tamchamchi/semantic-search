import os
from pathlib import Path

from dotenv import load_dotenv

from .path_loader import setup_paths
from .registry import Registry, registry

load_dotenv()

ACMM_DIR = Path(os.getenv("ACMM_DATA_DIR"))
FRAME_DIR = ACMM_DIR / "frames"

MAPPING_DIR = Path(os.getenv("MAPPING_DIR"))
FAISS_DIR = Path(os.getenv("FAISS_DIR"))
EMBEDS_DIR = Path(os.getenv("EMBEDS_DIR"))
WEIGHT = Path(os.getenv("WEIGHT"))

__all__ = [
    "registry",
    "Registry",

    "setup_paths",

    "ACMM_DIR",
    "FRAME_DIR",
    "MAPPING_DIR",
    "FAISS_DIR",
    "EMBEDS_DIR",
    "WEIGHT"
]
