import os
import sys
from dotenv import load_dotenv
from .registry import registry


def add_env_path(var_name):
    path = os.getenv(var_name)
    if path:
        abs_path = os.path.abspath(os.path.expanduser(path))
        if abs_path not in sys.path:
            sys.path.append(abs_path)


def init_path():
    registry.register_path("external_resource",
                           os.getenv("EXTERNAL_RESOURCE_DIR"))
    registry.register_path("acmm-data", os.getenv("ACMM_DATA_DIR"))
    registry.register_path("weight", os.getenv("WEIGHT"))
    registry.register_path("faiss", os.getenv("FAISS_DIR"))
    registry.register_path("embeds", os.getenv("EMBEDS_DIR"))
    registry.register_path("mapping", os.getenv("MAPPING_DIR"))


def setup_paths():
    load_dotenv()
    init_path()

    # Add paths to sys.path
    for key in ["EXTERNAL_RESOURCE_DIR", "ACMM_DATA_DIR", "WEIGHT"]:
        add_env_path(key)

    # Set Hugging Face cache path
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        try:
            abs_hf_path = os.path.abspath(os.path.expanduser(hf_home))
            os.environ["HF_HOME"] = abs_hf_path
        except Exception as e:
            print(f"[setup_paths] Failed to set HF_HOME: {e}")
