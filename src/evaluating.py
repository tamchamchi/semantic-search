from pathlib import Path

import numpy as np
import pandas as pd

from src.indexer import load_indexer
from src.semantic_extractor import load_semantic_extractor


class Evaluator:
    def __init__(self):
        pass

    def _load_gt_csv(self, gt_path: Path) -> pd.DataFrame:
        """
        Load the ground truth CSV file into a DataFrame.
        The CSV should contain information such as video_folder, video_id, start, end frames.
        """
        df = pd.read_csv(gt_path, quotechar='"', skipinitialspace=True)
        return df

    def build(self, mapping_file: Path, faiss_path: Path, extractor_name: str, indexer_name: str):
        """
        Initialize the semantic extractor and indexer, then load the FAISS index and mapping file.

        Args:
            mapping_file: Path to the mapping JSON file (index → metadata).
            faiss_path: Path to the saved FAISS index file.
            extractor_name: Name of the semantic extractor to load.
            indexer_name: Name of the indexer to load.
        """
        self.extrator = load_semantic_extractor(extractor_name)
        self.indexer = load_indexer(indexer_name, self.extrator)

        self.indexer.load(faiss_path, mapping_file)

    def eval(self, gt_file: Path, k_list=[1, 5, 10]):
        """
        Evaluate Recall@K for multiple K values on the given ground truth file.

        Args:
            gt_file: Path to the CSV containing ground truth frame ranges.
            k_list: List of K values for Recall@K computation (default: [1, 5, 10]).

        Returns:
            A dictionary {K: mean recall score} averaged over all queries.
        """
        # Load ground truth and mapping
        df_gt = self._load_gt_csv(gt_file)
        mapping = self.indexer.mapping  # List[dict] from mapping_file.json
        df_map = pd.DataFrame(mapping)
        df_map["frame_id"] = df_map["frame_id"].astype(int)

        # Store recall values for each K
        recalls_per_k = {k: [] for k in k_list}
        max_k = max(k_list)  # Largest K for FAISS search

        # Loop over each ground truth entry
        for _, row in df_gt.iterrows():
            query = row["query"]
            # print(query)

            # Search FAISS index for top max_k results (return only indices)
            indices = self.indexer.search(
                query, top_k=max_k, return_idx=True
            )

            # Build the set of relevant frame indices from GT range
            gt_frames_idx = set(
                df_map.index[
                    (df_map["video_folder"] == row["video_folder"]) &
                    (df_map["video_id"] == row["video_id"]) &
                    (df_map["frame_id"] >= row["start"]) &
                    (df_map["frame_id"] <= row["end"])
                ].tolist()
            )

            total_relevant = len(gt_frames_idx)
            if total_relevant == 0:
                continue  # Skip queries with no relevant frames

            # Compute recall for each K in k_list
            for k in k_list:
                top_k_idx = set(indices[0, :k])
                relevant_found = len(gt_frames_idx & top_k_idx)
                recalls_per_k[k].append(relevant_found / total_relevant)

        # Calculate mean Recall@K across all queries
        mean_recall = {
            k: np.mean(v) if v else 0.0
            for k, v in recalls_per_k.items()
        }
        return mean_recall


def run(extractor_name, indexer_name):
    import os
    from pathlib import Path

    from dotenv import load_dotenv

    from src.common import setup_paths

    setup_paths()
    load_dotenv()

    ACMM_DIR = Path(os.getenv("ACMM_DATA_DIR"))
    SEMANTIC_FOLDER = Path(ACMM_DIR, "semantic")
    GT_FILE = ACMM_DIR / "gt.csv"
    print(GT_FILE)

    mapping_file = SEMANTIC_FOLDER / f"mapping_{extractor_name}.json"
    faiss_file = SEMANTIC_FOLDER / f"faiss_index_{extractor_name}.faiss"

    evaluator = Evaluator()

    evaluator.build(mapping_file, faiss_file, extractor_name, indexer_name)

    score = evaluator.eval(GT_FILE, k_list=[1, 5, 10, 20, 50])

    return score


if __name__ == "__main__":
    score = run("beit3", "gpu-index-flat-l2")
    print(score)
