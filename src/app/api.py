"""
FastAPI application for semantic search with image and text indexing.
Uses FAISS-based similarity search with the Align extractor.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import time

try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as e:
    print(f"FastAPI dependencies not installed: {e}")
    print("Please install with: pip install fastapi uvicorn pydantic")
    raise

from dotenv import load_dotenv

from src.common import FAISS_DIR, MAPPING_DIR, setup_paths
from src.indexer import load_indexer, reciprocal_rank_fusion
from src.searcher import load_searcher
from src.semantic_extractor import load_semantic_extractor

from .schema import (
    ErrorResponse,
    SearchImageRequest,
    SearchResponse,
    SearchResult,
    SearchTextRequest,
)

# Global variables
models = {
    "align": {
        "indexer": None,
        "extractor": None,
        "searcher": None,
        "loaded": False
    },
    "coca-clip": {
        "indexer": None,
        "extractor": None,
        "searcher": None,
        "loaded": False
    },
    "apple-clip": {
        "indexer": None,
        "extractor": None,
        "searcher": None,
        "loaded": False
    }
}


async def initialize_model(model_name):
    """Initialize a specific model if saved files exist"""
    mapping_file = MAPPING_DIR / f"mapping_{model_name}.json"
    faiss_file = FAISS_DIR / f"faiss_index_{model_name}.faiss"
    indexer_name = "gpu-index-flat-l2"

    if mapping_file.exists() and faiss_file.exists():
        try:
            extractor = load_semantic_extractor(model_name)
            indexer = load_indexer(indexer_name, extractor=extractor)
            indexer.load(faiss_file, mapping_file)

            searcher = load_searcher(
                "single-semantic-search", extractor, indexer)

            models[model_name] = {
                "indexer": indexer,
                "extractor": extractor,
                "searcher": searcher,
                "loaded": True
            }
            print(f"✅ Loaded model: {model_name} with indexer: {indexer_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to load model {model_name}: {e}")
    else:
        print(f"⚠️ Index files missing for {model_name}")
    return False


def cleanup_resources():
    """Clean up resources on shutdown"""
    for model_name, model_data in models.items():
        if model_data["indexer"]:
            # Clean up any GPU resources
            try:
                if hasattr(model_data["indexer"], 'pool'):
                    del model_data["indexer"].pool
            except Exception:
                pass
            model_data["indexer"] = None
            model_data["searcher"] = None
            model_data["loaded"] = False


# Initialize FastAPI app
app = FastAPI(
    title="Semantic Search API",
    description="API for semantic search with multiple models and modes",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()
FRAMES_DIR = Path(
    os.getenv("FRAMES_DIR", "/mnt/mmlab2024nas/anhndt/Batch1/frames"))
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

# Mount static files directory
app.mount("/static", StaticFiles(directory=FRAMES_DIR), name="static")


@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    setup_paths()
    load_dotenv()

    # Initialize all available models
    print("\n🚀 Initializing models...")
    for model_name in models.keys():
        await initialize_model(model_name)
    print("✅ All models initialized\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    cleanup_resources()
    print("🛑 Resources cleaned up")


def make_public_url(local_path: str) -> str:
    """Convert local path to public URL"""
    if local_path.startswith(str(FRAMES_DIR)):
        rel_path = local_path[len(str(FRAMES_DIR)) + 1:]
    else:
        rel_path = local_path
    return f"{BASE_URL}/static/{rel_path}"


def get_idx(name, query, top_k):
    indexer = models.get(name)["indexer"]
    extracter = models.get(name)["extractor"]
    embedding = extracter.extract_text_features(query)
    _, idx = indexer.index_gpu.search(embedding, top_k)
    return idx[0].tolist()


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint"""
    loaded_models = [name for name, data in models.items() if data["loaded"]]
    return {
        "message": "Semantic Search API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "loaded_models": loaded_models,
        "frames_dir": str(FRAMES_DIR),
        "base_url": BASE_URL
    }


@app.post("/search_semantic/", response_model=SearchResponse)
async def search_text(request: SearchTextRequest):
    model_data = models.get(request.model)

    # Validate model availability
    if not model_data or not model_data["loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{request.model}' not loaded or not available"
        )

    searcher = model_data["searcher"]
    try:
        start_time = time.time()

        search_results = searcher.search(
            query=request.query,
            top_k=request.top_k
        )
        print(search_results)

        processing_time = time.time() - start_time

        # Return full metadata with public URLs
        query_formatted = []
        for query_result in search_results:
            original_path = query_result.get("path", "")
            public_path = make_public_url(
                original_path) if original_path else ""

            query_formatted.append(
                SearchResult(
                    path=public_path,
                    score=None,
                    metadata=query_result
                )
            )
        print(query_formatted)
        # Handle query type (single or multiple)
        if isinstance(request.query, str):
            total_queries = 1
        else:
            total_queries = len(request.query)

        return SearchResponse(
            query=request.query,
            results=query_formatted,
            model=request.model,
            total_queries=total_queries,
            top_k=request.top_k,
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@app.post("/search_fusion_semantic/", response_model=SearchResponse)
async def search_fusion_text(request: SearchTextRequest):

    mapping = models.get("align")["indexer"].mapping

    try:
        start_time = time.time()
        align_results = get_idx("align", request.query, request.top_k)
        coca_clip_results = get_idx("coca-clip", request.query, request.top_k)
        apple_clip_results = get_idx(
            "apple-clip", request.query, request.top_k)

        rank_lists = [
            align_results,
            coca_clip_results,
            apple_clip_results
        ]

        rrf = reciprocal_rank_fusion(rank_lists)

        fusion_res = [mapping[i[0]] for i in rrf[:request.top_k]]

        processing_time = time.time() - start_time

        # Return full metadata with public URLs
        query_formatted = []
        for query_result in fusion_res:
            original_path = query_result.get("path", "")
            public_path = make_public_url(
                original_path) if original_path else ""

            query_formatted.append(
                SearchResult(
                    path=public_path,
                    score=None,
                    metadata=query_result
                )
            )
        print(query_formatted)
        # Handle query type (single or multiple)
        if isinstance(request.query, str):
            total_queries = 1
        else:
            total_queries = len(request.query)

        return SearchResponse(
            query=request.query,
            results=query_formatted,
            model=request.model,
            total_queries=total_queries,
            top_k=request.top_k,
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )
