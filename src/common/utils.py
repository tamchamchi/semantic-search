import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Union

from PIL import Image
from tqdm import tqdm


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
    pattern = r"Videos_(L\d+_[a-z0-9]+)/.+?/(L\d+_(V\d+)_(\d+))"
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


def get_unique_path(path: Path) -> Path:
    """If file exists, append (2), (3), etc. until unique."""
    if not path.exists():
        return path
    counter = 2
    while True:
        new_path = path.with_name(f"{path.stem}({counter}){path.suffix}")
        if not new_path.exists():
            return new_path
        counter += 1


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


def reciprocal_rank_fusion(rank_lists, k_param=60):
    """
    Combine multiple ranked lists into a single ranking using Reciprocal Rank Fusion (RRF).

    RRF assigns a score to each document based on its position (rank) in each ranked list:
        score(d) = Σ [ 1 / (k_param + rank_i(d)) ]
    where rank_i(d) is the 1-based position of document `d` in the i-th ranked list.

    Args:
        rank_lists (List[List[Hashable]]):
            A list of ranked lists, where each ranked list is a sequence of document IDs ordered
            from most to least relevant. Document IDs can be any hashable type (e.g., int, str).
        k_param (int, optional):
            Constant added to the rank position to dampen the effect of lower-ranked documents.
            Defaults to 60, which is common in literature.

    Returns:
        List[Tuple[Hashable, float]]:
            A list of (document_id, fused_score) tuples, sorted in descending order of fused_score.

    Example:
        >>> rankings = [
        ...     [101, 102, 103, 104],
        ...     [103, 101, 105, 102]
        ... ]
        >>> result = reciprocal_rank_fusion(rankings, k_param=60)
        >>> for doc_id, score in result:
        ...     print(doc_id, score)
        101 0.03278688524590164
        103 0.032520325203252036
        102 0.03225806451612903
        104 0.016129032258064516
        105 0.016129032258064516
    """
    scores = {}
    for rank_list in rank_lists:
        for rank_pos, doc_id in enumerate(rank_list):
            scores[doc_id] = scores.get(
                doc_id, 0) + 1 / (k_param + rank_pos + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def get_image_from_path(paths: Union[str, list[str]]) -> list[Image.Image]:
    """
    Load images from given file paths.

    Args:
        paths (list[str]): Image file paths

    Returns:
        list[Image.Image]: List of RGB PIL images
    """
    if isinstance(paths, str):
        paths = [paths]

    images = []
    for path in tqdm(paths, desc="Image Loading..."):
        if not Path(path).exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        img = Image.open(path).convert("RGB")
        images.append(img)
    return images
