"""
FastAPI application for semantic search with image and text indexing.
Uses FAISS-based similarity search with the Align extractor.
"""

import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
except ImportError as e:
    print(f"FastAPI dependencies not installed: {e}")
    print("Please install with: pip install fastapi uvicorn pydantic")
    raise

from dotenv import load_dotenv

from src.common import setup_paths
from src.indexer import load_indexer
from src.semantic_extractor import load_semantic_extractor


# Pydantic Models / Schemas
class SearchRequest(BaseModel):
    """Schema for search requests"""
    query: Union[str, List[str]] = Field(
        ...,
        description="Text query or list of text queries for semantic search",
        example="A group of cyclists racing on bicycles"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of top results to return"
    )
    return_idx: bool = Field(
        default=False,
        description="If True, return only indices. If False, return full metadata"
    )

    @validator('query')
    def validate_query(cls, v):
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Query cannot be empty")
        elif isinstance(v, list):
            if not v or not all(isinstance(q, str) and q.strip() for q in v):
                raise ValueError("All queries must be non-empty strings")
        return v


class SearchResult(BaseModel):
    """Schema for individual search result"""
    path: str = Field(..., description="Path to the matched image")
    score: Optional[float] = Field(None, description="Similarity score")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata")


class SearchResponse(BaseModel):
    """Schema for search response"""
    query: Union[str, List[str]] = Field(..., description="Original query")
    results: List[List[SearchResult]] = Field(
        ...,
        description="List of result lists (one per query)"
    )
    total_queries: int = Field(..., description="Number of queries processed")
    top_k: int = Field(..., description="Number of results per query")
    processing_time: float = Field(...,
                                   description="Processing time in seconds")


class HealthResponse(BaseModel):
    """Schema for health check response"""
    status: str = Field(..., description="API health status")
    version: str = Field(..., description="API version")
    indexer_loaded: bool = Field(..., description="Whether indexer is loaded")
    extractor_name: Optional[str] = Field(
        None, description="Current extractor name")
    total_indexed_items: Optional[int] = Field(
        None, description="Total items in index")


class ErrorResponse(BaseModel):
    """Schema for error responses"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")


# Global variables
indexer = None
current_extractor_name = None
current_indexer_name = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    setup_paths()
    load_dotenv()

    # Initialize default indexer and extractor if files exist
    try:
        await initialize_default_indexer()
    except Exception as e:
        print(f"Warning: Could not load default indexer: {e}")

    yield

    # Shutdown
    cleanup_resources()


def cleanup_resources():
    """Clean up resources on shutdown"""
    global indexer
    if indexer:
        # Clean up any GPU resources if applicable
        try:
            if hasattr(indexer, 'pool'):
                del indexer.pool
        except Exception:
            pass
        indexer = None


async def initialize_default_indexer():
    """Initialize default indexer if saved files exist"""
    global indexer, current_extractor_name, current_indexer_name

    ACMM_DIR = Path(os.getenv("ACMM_DATA_DIR", "data"))
    SEMANTIC_FOLDER = Path(ACMM_DIR, "semantic")

    extractor_name = "align"  # Default
    indexer_name = "gpu-index-flat-l2"  # Default

    mapping_file = SEMANTIC_FOLDER / f"mapping_{extractor_name}.json"
    faiss_file = SEMANTIC_FOLDER / f"faiss_index_{extractor_name}.faiss"

    if mapping_file.exists() and faiss_file.exists():
        try:
            extractor = load_semantic_extractor(extractor_name)
            indexer = load_indexer(indexer_name, extractor=extractor)
            indexer.load(faiss_file, mapping_file)

            current_extractor_name = extractor_name
            current_indexer_name = indexer_name
            print(
                f"Loaded default indexer: {indexer_name} with extractor: {extractor_name}")
        except Exception as e:
            print(f"Failed to load default indexer: {e}")
            raise


# Initialize FastAPI app
app = FastAPI(
    title="Semantic Search API (Align)",
    description="API for semantic search using FAISS indexing with the Align extractor",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Semantic Search API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global indexer, current_extractor_name

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        indexer_loaded=indexer is not None,
        extractor_name=current_extractor_name,
        total_indexed_items=indexer.index_gpu.ntotal if indexer else None
    )


@app.post("/search", response_model=SearchResponse)
async def search_images(request: SearchRequest):
    """
    Search for images using semantic similarity.

    - **query**: Text query or list of queries
    - **top_k**: Number of results to return per query (1-100)
    - **return_idx**: Return only indices (True) or full metadata (False)
    """
    global indexer

    if not indexer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Indexer not loaded. Please build index first using /indexing endpoint."
        )

    try:
        import time
        start_time = time.time()

        # Perform search
        search_results = indexer.search(
            query=request.query,
            top_k=request.top_k,
            return_idx=request.return_idx
        )

        processing_time = time.time() - start_time

        # Format response
        if request.return_idx:
            # If return_idx=True, results are indices only
            formatted_results = []
            for query_results in search_results:
                query_formatted = []
                for idx in query_results:
                    query_formatted.append(
                        SearchResult(
                            path=f"index_{idx}",
                            score=None,
                            metadata={"index": int(idx)}
                        )
                    )
                formatted_results.append(query_formatted)
        else:
            # If return_idx=False, results contain full metadata
            formatted_results = []
            for query_results in search_results:
                query_formatted = []
                for item in query_results:
                    query_formatted.append(
                        SearchResult(
                            path=item.get("path", ""),
                            score=None,  # Could add distance scores if available
                            metadata=item
                        )
                    )
                formatted_results.append(query_formatted)

        # Handle single query case
        if isinstance(request.query, str):
            total_queries = 1
        else:
            total_queries = len(request.query)

        return SearchResponse(
            query=request.query,
            results=formatted_results,
            total_queries=total_queries,
            top_k=request.top_k,
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
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
