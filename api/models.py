"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Optional

# --- Response Models ---
class BookRecommendation(BaseModel):
    """Model for a single book recommendation"""
    bookId: str = Field(..., description="Unique book identifier")
    title: str = Field(..., description="Book title")
    author: str = Field(..., description="Book author")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Average rating (0-5)")
    ratingsCount: Optional[int] = Field(None, ge=0, description="Number of ratings")
    genres: Optional[list] = Field(None, description="List of book genres")
    coverImg: Optional[str] = Field(None, description="URL to book cover image")
    score: float = Field(..., ge=0, le=1, description="Similarity score between 0 and 1")

class RecommendationResponse(BaseModel):
    """Response model for book recommendations"""
    input_book: dict = Field(..., description="Information about the input book")
    recommendations: List[BookRecommendation] = Field(..., description="List of recommended books")
    total_recommendations: int = Field(..., ge=0, description="Total number of recommendations returned")

class BookInfo(BaseModel):
    """Model for book information"""
    model_config = {"protected_namespaces": ()}  # Fix Pydantic warning
    
    bookId: str = Field(..., description="Unique book identifier")
    title: str = Field(..., description="Book title")
    author: str = Field(..., description="Book author")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Average rating (0-5)")
    ratingsCount: Optional[int] = Field(None, ge=0, description="Number of ratings")
    reviewsCount: Optional[int] = Field(None, ge=0, description="Number of reviews")
    description: Optional[str] = Field("", description="Book description")
    genres: Optional[list] = Field(None, description="List of book genres")
    coverImg: Optional[str] = Field(None, description="URL to book cover image")
    publishedDate: Optional[str] = Field(None, description="Publication date")

# --- Request Models ---
class RecommendationRequest(BaseModel):
    """Request model for getting recommendations"""
    book_id: int = Field(..., gt=0, description="ID of the book to get recommendations for")
    num_recommendations: Optional[int] = Field(
        5, 
        ge=1, 
        le=50, 
        description="Number of recommendations to return (1-50)"
    )

# --- Status Models ---
class HealthResponse(BaseModel):
    """Health check response model"""
    model_config = {"protected_namespaces": ()}  # Fix Pydantic warning
    
    status: str = Field(..., description="Service status (healthy/unhealthy)")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    total_books: int = Field(..., ge=0, description="Total number of books in the dataset")
    model_directory: str = Field(..., description="Path to the model directory")

class APIInfo(BaseModel):
    """API information response model"""
    message: str = Field(..., description="API welcome message")
    version: str = Field(..., description="API version")
    endpoints: dict = Field(..., description="Available API endpoints")

# --- Error Models ---
class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")

# --- Filter and Search Models ---
class BookFilterRequest(BaseModel):
    """Request model for filtering books"""
    min_rating: Optional[float] = Field(None, ge=0, le=5, description="Minimum rating")
    max_rating: Optional[float] = Field(None, ge=0, le=5, description="Maximum rating")
    genres: Optional[List[str]] = Field(None, description="Filter by genres")
    author: Optional[str] = Field(None, description="Filter by author")
    min_ratings_count: Optional[int] = Field(None, ge=0, description="Minimum number of ratings")
    published_year: Optional[int] = Field(None, description="Filter by publication year")

class SortOption(BaseModel):
    """Sorting options for book lists"""
    field: str = Field("title", description="Field to sort by")
    order: str = Field("asc", pattern="^(asc|desc)$", description="Sort order: asc or desc")

class PaginationParams(BaseModel):
    """Pagination parameters"""
    skip: int = Field(0, ge=0, description="Number of items to skip")
    limit: int = Field(50, ge=1, le=100, description="Maximum number of items to return")

class PaginatedBooksResponse(BaseModel):
    """Paginated books response"""
    books: List[BookInfo] = Field(..., description="List of books")
    total: int = Field(..., ge=0, description="Total number of books available")
    skip: int = Field(..., ge=0, description="Number of items skipped")
    limit: int = Field(..., ge=1, description="Maximum items per page")