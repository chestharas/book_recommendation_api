"""
API route handlers
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .models import (
    BookRecommendation, 
    RecommendationResponse, 
    RecommendationRequest,
    BookInfo, 
    HealthResponse, 
    APIInfo,
    PaginatedBooksResponse
)
from .dependencies import get_model_components

# Create router
router = APIRouter()

def get_recommendations_by_book_id(
    book_id: int, 
    num_recommendations: int = 5,
    df_books: pd.DataFrame = None,
    tfidf_matrix = None
):
    """Get recommendations for a book by its ID"""
    if df_books is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Find the book
    target_book_matches = df_books[df_books['bookId_lookup'] == book_id]
    
    if target_book_matches.empty:
        raise HTTPException(
            status_code=404, 
            detail=f"Book with ID {book_id} not found in the dataset"
        )
    
    selected_book_data = target_book_matches.iloc[0]
    input_book_df_idx = selected_book_data.name  # DataFrame index
    
    # Validate index bounds
    if not (0 <= input_book_df_idx < len(df_books)):
        raise HTTPException(status_code=500, detail="Invalid book index")
    
    if input_book_df_idx >= tfidf_matrix.shape[0]:
        raise HTTPException(status_code=500, detail="Book index out of bounds for TF-IDF matrix")
    
    # Calculate similarities
    input_book_vector = tfidf_matrix[input_book_df_idx]
    cosine_similarities = cosine_similarity(input_book_vector, tfidf_matrix)[0]
    
    # Get top similar books (excluding the input book itself)
    similarity_scores_with_indices = []
    for sim_idx, score in enumerate(cosine_similarities):
        if sim_idx != input_book_df_idx:  # Exclude the input book itself
            similarity_scores_with_indices.append((sim_idx, score))
    
    # Sort by similarity score
    similarity_scores_with_indices = sorted(
        similarity_scores_with_indices, 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Build recommendations
    recommendations = []
    for rec_df_idx, rec_score in similarity_scores_with_indices[:num_recommendations]:
        recommended_book_data = df_books.iloc[rec_df_idx]
        
        # Parse genres if they exist
        genres = None
        if 'genres_list' in recommended_book_data:
            genres = recommended_book_data['genres_list']
        elif 'genres' in recommended_book_data and pd.notna(recommended_book_data['genres']):
            try:
                import ast
                genres = ast.literal_eval(str(recommended_book_data['genres']))
            except:
                genres = [str(recommended_book_data['genres'])]
        
        recommendations.append(BookRecommendation(
            bookId=str(recommended_book_data['bookId']),
            title=recommended_book_data['title'],
            author=recommended_book_data['author'],
            rating=float(recommended_book_data['rating']) if pd.notna(recommended_book_data.get('rating')) else None,
            ratingsCount=int(recommended_book_data['ratingsCount']) if pd.notna(recommended_book_data.get('ratingsCount')) else None,
            genres=genres,
            coverImg=recommended_book_data.get('coverImg') if pd.notna(recommended_book_data.get('coverImg')) else None,
            score=float(rec_score)
        ))
    
    # Input book info
    input_book_info = {
        "bookId": str(selected_book_data['bookId']),
        "title": selected_book_data['title'],
        "author": selected_book_data['author']
    }
    
    return RecommendationResponse(
        input_book=input_book_info,
        recommendations=recommendations,
        total_recommendations=len(recommendations)
    )

# --- Route Handlers ---

@router.get("/", response_model=APIInfo)
async def root():
    """Root endpoint with API information"""
    return APIInfo(
        message="Book Recommendation API",
        version="1.0.0",
        endpoints={
            "get_recommendations": "/recommendations/{book_id}",
            "get_recommendations_post": "/recommendations",
            "get_book_info": "/books/{book_id}",
            "list_books": "/books",
            "health": "/health"
        }
    )

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    df_books, tfidf_vectorizer, tfidf_matrix, model_dir = get_model_components()
    
    model_loaded = all([df_books is not None, tfidf_vectorizer is not None, tfidf_matrix is not None])
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        total_books=len(df_books) if df_books is not None else 0,
        model_directory=str(model_dir)
    )

@router.get("/recommendations/{book_id}", response_model=RecommendationResponse)
async def get_recommendations_get(
    book_id: int, 
    num_recommendations: int = Query(5, ge=1, le=50, description="Number of recommendations")
):
    """Get book recommendations by book ID (GET method)"""
    df_books, _, tfidf_matrix, _ = get_model_components()
    
    return get_recommendations_by_book_id(
        book_id=book_id,
        num_recommendations=num_recommendations,
        df_books=df_books,
        tfidf_matrix=tfidf_matrix
    )

@router.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations_post(request: RecommendationRequest):
    """Get book recommendations by book ID (POST method)"""
    df_books, _, tfidf_matrix, _ = get_model_components()
    
    return get_recommendations_by_book_id(
        book_id=request.book_id,
        num_recommendations=request.num_recommendations,
        df_books=df_books,
        tfidf_matrix=tfidf_matrix
    )

@router.get("/books/{book_id}", response_model=BookInfo)
async def get_book_info(book_id: int):
    """Get information about a specific book"""
    df_books, _, _, _ = get_model_components()
    
    if df_books is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    book_matches = df_books[df_books['bookId_lookup'] == book_id]
    
    if book_matches.empty:
        raise HTTPException(
            status_code=404, 
            detail=f"Book with ID {book_id} not found"
        )
    
    book_data = book_matches.iloc[0]
    
    # Parse genres
    genres = None
    if 'genres_list' in book_data:
        genres = book_data['genres_list']
    elif 'genres' in book_data and pd.notna(book_data['genres']):
        try:
            import ast
            genres = ast.literal_eval(str(book_data['genres']))
        except:
            genres = [str(book_data['genres'])]
    
    return BookInfo(
        bookId=str(book_data['bookId']),
        title=book_data['title'],
        author=book_data['author'],
        rating=float(book_data['rating']) if pd.notna(book_data.get('rating')) else None,
        ratingsCount=int(book_data['ratingsCount']) if pd.notna(book_data.get('ratingsCount')) else None,
        reviewsCount=int(book_data['reviewsCount']) if pd.notna(book_data.get('reviewsCount')) else None,
        description=book_data.get('description', ''),
        genres=genres,
        coverImg=book_data.get('coverImg') if pd.notna(book_data.get('coverImg')) else None,
        publishedDate=book_data.get('publishedDate') if pd.notna(book_data.get('publishedDate')) else None
    )

@router.get("/books", response_model=PaginatedBooksResponse)
async def list_books(
    skip: int = Query(0, ge=0, description="Number of books to skip"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of books to return")
):
    """List all books with pagination"""
    df_books, _, _, _ = get_model_components()
    
    if df_books is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    total_books = len(df_books)
    books_subset = df_books.iloc[skip:skip + limit]
    
    books_list = []
    for _, book_data in books_subset.iterrows():
        # Parse genres
        genres = None
        if 'genres_list' in book_data:
            genres = book_data['genres_list']
        elif 'genres' in book_data and pd.notna(book_data['genres']):
            try:
                import ast
                genres = ast.literal_eval(str(book_data['genres']))
            except:
                genres = [str(book_data['genres'])]
        
        books_list.append(BookInfo(
            bookId=str(book_data['bookId']),
            title=book_data['title'],
            author=book_data['author'],
            rating=float(book_data['rating']) if pd.notna(book_data.get('rating')) else None,
            ratingsCount=int(book_data['ratingsCount']) if pd.notna(book_data.get('ratingsCount')) else None,
            reviewsCount=int(book_data['reviewsCount']) if pd.notna(book_data.get('reviewsCount')) else None,
            description=book_data.get('description', ''),
            genres=genres,
            coverImg=book_data.get('coverImg') if pd.notna(book_data.get('coverImg')) else None,
            publishedDate=book_data.get('publishedDate') if pd.notna(book_data.get('publishedDate')) else None
        ))
    
    return PaginatedBooksResponse(
        books=books_list,
        total=total_books,
        skip=skip,
        limit=limit
    )

# --- Search Routes (Bonus) ---

@router.get("/search/books", response_model=List[BookInfo])
async def search_books(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results")
):
    """Search books by title or author"""
    df_books, _, _, _ = get_model_components()
    
    if df_books is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Simple text search in title and author columns
    query_lower = q.lower()
    mask = (
        df_books['title'].str.lower().str.contains(query_lower, na=False) |
        df_books['author'].str.lower().str.contains(query_lower, na=False)
    )
    
    search_results = df_books[mask].head(limit)
    
    books_list = []
    for _, book_data in search_results.iterrows():
        books_list.append(BookInfo(
            bookId=str(book_data['bookId']),
            title=book_data['title'],
            author=book_data['author'],
            description=book_data.get('description', '')
        ))
    
    return books_list