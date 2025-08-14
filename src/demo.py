import streamlit as st
from PIL import Image
import os
from dotenv import load_dotenv
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.common import setup_paths, FAISS_DIR, MAPPING_DIR
from src.semantic_extractor import load_semantic_extractor
from src.indexer import load_indexer

# Setup path & environment
setup_paths()
load_dotenv()

# === Constants ===
EXTRACTORS = ["align", "siglip", "siglip2",
              "clip", "coca-clip", "apple-clip", "beit3"]


# === Load models/indexers based on extractor name ===


@st.cache_resource
def get_indexer(extractor_name):
    extractor_key = extractor_name.lower()
    extractor = load_semantic_extractor(extractor_key)
    indexer = load_indexer("gpu-index-flat-l2", extractor=extractor)

    index_path = FAISS_DIR / f"faiss_index_{extractor_key}.faiss"
    mapping_path = MAPPING_DIR / f"mapping_{extractor_key}.json"

    indexer.load(index_path, mapping_path)
    return indexer

# === Search function ===


def search_images(query: str, extractor_name: str, top_k: int = 10):
    indexer = get_indexer(extractor_name)
    results = indexer.search(query, top_k=top_k)
    return results[0]


# === UI Sidebar ===
st.sidebar.title("🔍 Image Query")
query = st.sidebar.text_input("Enter your query:")
selected_extractor = st.sidebar.selectbox("Select extractor:", EXTRACTORS)
top_k = st.sidebar.slider(
    "Number of images to show (top_k):", min_value=1, max_value=50, value=10)
# Clear cache khi bấm nút
if st.sidebar.button("Clear indexer cache"):
    get_indexer.clear()
    st.success("Đã xoá cache của indexer.")

# === UI Main ===
st.title("📷 Related Images Viewer")

if query:
    st.write(
        f"### Query: `{query}` using `{selected_extractor}` (Top-{top_k})")
    try:
        infors = search_images(query, selected_extractor, top_k)

        cols = st.columns(3)
        for idx, infor in enumerate(infors):
            with cols[idx % 3]:
                st.image(Image.open(infor["path"]), caption=os.path.basename(
                    infor["path"]), use_container_width=True)
        else:
            st.warning("No images matched your query.")
    except Exception as e:
        st.error(f"Error during search: {e}")
else:
    st.info("Please enter a query to search for images.")
