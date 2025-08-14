from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union, Dict, Any


class SearchTextRequest(BaseModel):
    """Schema for search text requests"""
    query: Union[str, List[str]] = Field(
        ...,
        description="Text query or list of text queries for semantic search",
        example="A group of cyclists racing on bicycles"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=500,
        description="Number of top results to return"
    )
    model: str = Field(
        default="coca-clip",
        description="Search Model: 'coca-clip', 'align', 'apple-clip'"
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
    
class SearchImageRequest(BaseModel):
    """Schema for search Image requests"""
    query: Union[str, List[str]] = Field(
        ...,
        description="Image query or list of Image queries for semantic search",
        example="A group of cyclists racing on bicycles"
    )
    model: str = Field(
        default="coca-clip",
        description="Search Model: 'coca-clip', 'align', 'apple-clip'"
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
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata")

class SearchResponse(BaseModel):
    """Schema for search response"""
    query: Union[str, List[str]] = Field(..., description="Original query")
    results: List[SearchResult] = Field(
        ...,
        description="List of result lists (one per query)"
    )
    total_queries: int = Field(..., description="Number of queries processed")
    top_k: int = Field(..., description="Number of results per query")
    processing_time: float = Field(...,
                                   description="Processing time in seconds")

class ErrorResponse(BaseModel):
    """Schema for error responses"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")

