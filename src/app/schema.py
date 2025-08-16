from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any

# Pydantic Models / Schemas
class SearchResult(BaseModel):
    """Schema for individual search result"""
    path: str = Field(..., description="Path to the matched image")
    score: Optional[float] = Field(None, description="Similarity score")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata")

class SearchResponse(BaseModel):
    """Schema for search response"""
    query: Union[str, List[str]] = Field(..., description="Original query")
    results: List[SearchResult] = Field(
        ...,
        description="List of search results"
    )
    model: str = Field(..., description="Model used for search")
    total_queries: int = Field(..., description="Number of queries processed")
    top_k: int = Field(..., description="Number of results per query")
    processing_time: float = Field(..., description="Processing time in seconds")

class SearchRequest(BaseModel):
    """Request schema for text or image search"""
    query: Union[str, List[str]] = Field(
        ...,
        description="Text query, list of text queries, image path(s), or URL of image",
        example="A group of cyclists racing on bicycles | /data/images/290704.jpg"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=500,
        description="Number of top results to return"
    )
    model: str = Field(
        default="coca-clip",
        description="Model to use: 'align', 'coca-clip', 'apple-clip', or 'fusion'"
    )
    mode: str = Field(
        default="text",
        description="Mode to use: 'text' or 'image'"
    )

class HealthResponse(BaseModel):
    """Schema for health check response"""
    status: str = Field(..., description="API health status")
    version: str = Field(..., description="API version")
    loaded_models: Dict[str, bool] = Field(
        ...,
        description="Status of loaded models"
    )
    total_indexed_items: Dict[str, int] = Field(
        ...,
        description="Total items indexed per model"
    )

class ErrorResponse(BaseModel):
    """Schema for error responses"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
