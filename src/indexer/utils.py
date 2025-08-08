from pathlib import Path
from typing import Union, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import re


def parse_path_info(example: str):
    """
    Parse a single .jpg file path to extract metadata.

    Example input:
        /mnt/.../Videos_L23_a/L23_V001/L23_V001_00026.jpg

    Returns:
        dict with keys:
            - video_folder: e.g., "L23_a"
            - video_id: e.g., "V001"
            - frame_id: e.g., "00026"
            - path: the original file path

    Raises:
        ValueError: if the path does not match the expected format.
    """
    # Regex pattern to extract video folder, video id, and frame id
    pattern = r"Videos_(L\d+_[a-z])/.+?/(L\d+_(V\d+)_(\d+))\.jpg"
    match = re.search(pattern, example)

    if match:
        # Extract and return metadata as a dictionary
        video_folder = match.group(1)  # Example: L23_a
        video_id = match.group(3)      # Example: V001
        frame_id = match.group(4)      # Example: 00026
        return {
            "video_folder": video_folder,
            "video_id": video_id,
            "frame_id": frame_id,
            "path": example
        }
    else:
        # Raise an error if the path format is unexpected
        raise ValueError(f"Invalid path format: {example}")


def parse_frames_info(folder_path: Union[Path, str], max_workers: int = 16) -> List[dict]:
    """
    Recursively scan a folder for .jpg files and parse each into metadata using multithreading.

    Args:
        folder_path: Root folder containing video frame subfolders (Path or string).
        max_workers: Number of threads to use for concurrent scanning.

    Returns:
        List of dictionaries, each containing:
            {
                "video_folder": str,
                "video_id": str,
                "frame_id": str,
                "path": str
            }
    """
    # Convert folder_path to Path object if needed
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)

    # Get first-level subdirectories to parallelize scanning
    subfolders = [p for p in folder_path.iterdir() if p.is_dir()]

    # Container to store all parsed frame metadata
    frames_info = []

    def scan_folder(subfolder: Path) -> List[dict]:
        """
        Recursively scan a single subfolder for .jpg files
        and parse each path into a metadata dictionary.
        """
        return [parse_path_info(str(p)) for p in subfolder.rglob("*.jpg")]

    # Use ThreadPoolExecutor to scan subfolders concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit a scan task for each subfolder
        futures = [executor.submit(scan_folder, subfolder)
                   for subfolder in subfolders]

        # Collect results as tasks complete
        for future in as_completed(futures):
            frames_info.extend(future.result())

    return frames_info

