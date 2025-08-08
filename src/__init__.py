import sys
import os
from src.common import registry
from pathlib import Path

os.environ["HF_HOME"] = "/mnt/mmlab2024nas/anhndt/hf_cache"
os.environ["HF_HUB_CACHE"] = os.path.join(os.environ["HF_HOME"], "hub")

def get_aic_data():
    # Setup AIC data
    data_folder_path = "/mnt/mmlab2024nas/anhndt/Batch1/frames"
    # registry path
    registry.register_path("aic-data", data_folder_path)
    sys.path.append(data_folder_path)
    return Path(data_folder_path)
