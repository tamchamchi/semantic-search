"""
FastAPI application for semantic search with image and text indexing.
Uses FAISS-based similarity search with multiple models.
"""

import os
from pathlib import Path
from typing import Any, Dict
import time

try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from .schema import ErrorResponse, SearchResponse, SearchResult, SearchTextRequest, HealthResponse
except ImportError as e:
    print(f"FastAPI dependencies not installed: {e}")
    print("Please install with: pip install fastapi uvicorn pydantic")
    raise

from dotenv import load_dotenv

from src.common import FAISS_DIR, MAPPING_DIR, setup_paths
from src.indexer import load_indexer, reciprocal_rank_fusion
from src.searcher import load_searcher
from src.semantic_extractor import load_semantic_extractor

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
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8081/")

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
    return f"{BASE_URL}static/{rel_path}"

def get_idx(name, query, top_k):
    """Get search indices for a model"""
    model_data = models.get(name)
    if not model_data or not model_data["loaded"]:
        return None
    
    indexer = model_data["indexer"]
    extractor = model_data["extractor"]
    
    # Extract text features
    embedding = extractor.extract_text_features(query)
    
    # Perform search
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

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    loaded_status = {}
    total_items = {}
    
    for model_name, model_data in models.items():
        loaded_status[model_name] = model_data["loaded"]
        if model_data["loaded"] and model_data["indexer"] and hasattr(model_data["indexer"], 'index_gpu'):
            total_items[model_name] = model_data["indexer"].index_gpu.ntotal
    
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        loaded_models=loaded_status,
        total_indexed_items=total_items
    )

@app.post("/search_text/", response_model=SearchResponse)
async def search_text(request: SearchTextRequest):
    """
    Perform text-based semantic search
    
    Args:
        request (SearchTextRequest): Search request with query and parameters
    
    Returns:
        SearchResponse: Search results with metadata
    """
    # Handle fusion search separately
    if request.model == "fusion":
        return await search_fusion_text(request)
    
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

        # Perform search
        search_results = searcher.search(
            query=request.query,
            top_k=request.top_k
        )

        processing_time = time.time() - start_time

        # Format results
        formatted_results = []
        for item in search_results:
            original_path = item.get("path", "")
            public_path = make_public_url(original_path) if original_path else ""

            formatted_results.append(
                SearchResult(
                    path=public_path,
                    score=None,
                    metadata=item
                )
            )

        # Handle query type (single or multiple)
        if isinstance(request.query, str):
            total_queries = 1
        else:
            total_queries = len(request.query)

        return SearchResponse(
            query=request.query,
            results=formatted_results,
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

@app.post("/search_fusion/", response_model=SearchResponse)
async def search_fusion_text(request: SearchTextRequest):
    """
    Perform fusion search using reciprocal rank fusion
    
    Args:
        request (SearchTextRequest): Search request with query and parameters
    
    Returns:
        SearchResponse: Fused search results
    """
    # Get mapping from align model (assuming all models have same mapping)
    align_model = models.get("align")
    if not align_model or not align_model["loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Align model not loaded, required for fusion"
        )
    
    mapping = align_model["indexer"].mapping

    try:
        start_time = time.time()
        
        # Get results from each model
        align_results = get_idx("align", request.query, request.top_k)
        coca_clip_results = get_idx("coca-clip", request.query, request.top_k)
        apple_clip_results = get_idx("apple-clip", request.query, request.top_k)

        # Check which models are available
        rank_lists = []
        if align_results is not None:
            rank_lists.append(align_results)
        if coca_clip_results is not None:
            rank_lists.append(coca_clip_results)
        if apple_clip_results is not None:
            rank_lists.append(apple_clip_results)
        
        if not rank_lists:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No models available for fusion"
            )

        # Apply reciprocal rank fusion
        rrf = reciprocal_rank_fusion(rank_lists)
        
        # Get top-k results
        fusion_res = [mapping[i[0]] for i in rrf[:request.top_k]]

        processing_time = time.time() - start_time

        # Format results
        formatted_results = []
        for item in fusion_res:
            original_path = item.get("path", "")
            public_path = make_public_url(original_path) if original_path else ""

            formatted_results.append(
                SearchResult(
                    path=public_path,
                    score=None,
                    metadata=item
                )
            )

        return SearchResponse(
            query=request.query,
            results=formatted_results,
            model="fusion",
            total_queries=1 if isinstance(request.query, str) else len(request.query),
            top_k=request.top_k,
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fusion search failed: {str(e)}"
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP_ERROR",
            message=exc.detail,
            detail=f"Status code: {exc.status_code}"
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="INTERNAL_ERROR",
            message="An unexpected error occurred",
            detail=str(exc)
        ).dict()
    )