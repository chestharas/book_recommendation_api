"""
API route handlers
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import pandas as pd
import ast
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
from .dependencies import get_model_components, get_model_stats

# Create router
router = APIRouter()

def safe_parse_genres(genres_data):
    """Safely parse genres from string or list"""
    if pd.isna(genres_data) or genres_data == "":
        return None
    
    if isinstance(genres_data, list):
        return genres_data
    
    if isinstance(genres_data, str):
        try:
            # Try to parse as literal (for lists stored as strings)
            parsed = ast.literal_eval(genres_data)
            if isinstance(parsed, list):
                return parsed
        except:
            # If parsing fails, treat as single genre
            return [genres_data]
    
    return None

def create_book_info(book_data):
    """Create BookInfo object from DataFrame row"""
    return BookInfo(
        bookId=str(book_data['bookId']),
        title=book_data['title'],
        author=book_data['author'],
        rating=float(book_data['rating']) if pd.notna(book_data.get('rating')) else None,
        ratingsCount=int(book_data['ratingsCount']) if pd.notna(book_data.get('ratingsCount')) else None,
        reviewsCount=int(book_data['reviewsCount']) if pd.notna(book_data.get('reviewsCount')) else None,
        description=book_data.get('description', ''),
        genres=safe_parse_genres(book_data.get('genres_list') or book_data.get('genres')),
        coverImg=book_data.get('coverImg') if pd.notna(book_data.get('coverImg')) else None,
        publishedDate=book_data.get('publishedDate') if pd.notna(book_data.get('publishedDate')) else None
    )

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
    input_book_df_idx = selected_book_data.name
    
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
        if sim_idx != input_book_df_idx:
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
        
        recommendations.append(BookRecommendation(
            bookId=str(recommended_book_data['bookId']),
            title=recommended_book_data['title'],
            author=recommended_book_data['author'],
            rating=float(recommended_book_data['rating']) if pd.notna(recommended_book_data.get('rating')) else None,
            ratingsCount=int(recommended_book_data['ratingsCount']) if pd.notna(recommended_book_data.get('ratingsCount')) else None,
            genres=safe_parse_genres(recommended_book_data.get('genres_list') or recommended_book_data.get('genres')),
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
# IMPORTANT: Specific routes MUST come before dynamic routes!

@router.get("/")
async def root():
    """Root endpoint with API information"""
    stats = get_model_stats()
    
    return {
        "message": "Book Recommendation API",
        "version": "1.0.0",
        "total_books": stats.get('total_books', 0),
        "endpoints": {
            "recommendations": "/recommendations/{book_id}",
            "book_info": "/books/{book_id}",
            "list_books": "/books",
            "filter_books": "/books/filter",
            "top_rated": "/books/top-rated",
            "search": "/search/books",
            "genres": "/books/genres",
            "health": "/health"
        }
    }

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

# SPECIFIC BOOK ROUTES - THESE MUST COME BEFORE /books/{book_id}

@router.get("/books/filter", response_model=List[BookInfo])
async def filter_books(
    min_rating: Optional[float] = Query(None, ge=0, le=5),
    max_rating: Optional[float] = Query(None, ge=0, le=5),
    genre: Optional[str] = Query(None),
    author: Optional[str] = Query(None),
    min_ratings_count: Optional[int] = Query(None, ge=0),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100)
):
    """Filter books by various criteria"""
    df_books, _, _, _ = get_model_components()
    
    if df_books is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    filtered_df = df_books.copy()
    
    # Apply filters
    if min_rating is not None and 'rating' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['rating'].fillna(0) >= min_rating]
    
    if max_rating is not None and 'rating' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['rating'].fillna(5) <= max_rating]
    
    if genre is not None:
        if 'genres_list' in filtered_df.columns:
            mask = filtered_df['genres_list'].apply(
                lambda x: genre.lower() in [g.lower() for g in x] if isinstance(x, list) else False
            )
        elif 'genres' in filtered_df.columns:
            mask = filtered_df['genres'].str.contains(genre, case=False, na=False)
        else:
            mask = pd.Series([False] * len(filtered_df))
        filtered_df = filtered_df[mask]
    
    if author is not None:
        filtered_df = filtered_df[filtered_df['author'].str.contains(author, case=False, na=False)]
    
    if min_ratings_count is not None and 'ratingsCount' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['ratingsCount'].fillna(0) >= min_ratings_count]
    
    # Apply pagination
    paginated_df = filtered_df.iloc[skip:skip + limit]
    
    books_list = []
    for _, book_data in paginated_df.iterrows():
        books_list.append(create_book_info(book_data))
    
    return books_list

@router.get("/books/top-rated", response_model=List[BookInfo])
async def get_top_rated_books(
    min_ratings: int = Query(10, ge=1, description="Minimum number of ratings required"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of books to return")
):
    """Get top-rated books"""
    print(f"DEBUG: Received parameters - min_ratings: {min_ratings}, limit: {limit}")
    
    df_books, _, _, _ = get_model_components()
    
    if df_books is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    print(f"DEBUG: DataFrame loaded with {len(df_books)} books")
    print(f"DEBUG: Columns: {df_books.columns.tolist()}")
    
    if 'rating' not in df_books.columns or 'ratingsCount' not in df_books.columns:
        raise HTTPException(status_code=400, detail="Rating data not available")
    
    try:
        # Check data types
        print(f"DEBUG: rating column dtype: {df_books['rating'].dtype}")
        print(f"DEBUG: ratingsCount column dtype: {df_books['ratingsCount'].dtype}")
        print(f"DEBUG: Sample rating values: {df_books['rating'].head().tolist()}")
        print(f"DEBUG: Sample ratingsCount values: {df_books['ratingsCount'].head().tolist()}")
        
        # Ensure numeric conversion
        df_books = df_books.copy()
        df_books['rating'] = pd.to_numeric(df_books['rating'], errors='coerce')
        df_books['ratingsCount'] = pd.to_numeric(df_books['ratingsCount'], errors='coerce')
        
        print(f"DEBUG: After conversion - rating dtype: {df_books['rating'].dtype}")
        print(f"DEBUG: After conversion - ratingsCount dtype: {df_books['ratingsCount'].dtype}")
        
        # Filter books
        mask = (
            (df_books['ratingsCount'].fillna(0) >= min_ratings) & 
            (df_books['rating'].notna()) &
            (df_books['rating'] > 0)
        )
        filtered_books = df_books[mask]
        
        print(f"DEBUG: Found {len(filtered_books)} books matching criteria")
        
        if filtered_books.empty:
            print("DEBUG: No books found matching criteria")
            return []
        
        # Sort books
        top_books = filtered_books.sort_values(
            ['rating', 'ratingsCount'], 
            ascending=[False, False]
        ).head(limit)
        
        print(f"DEBUG: Returning top {len(top_books)} books")
        
        books_list = []
        for _, book_data in top_books.iterrows():
            try:
                book_info = create_book_info(book_data)
                books_list.append(book_info)
            except Exception as e:
                print(f"DEBUG: Error creating book info: {e}")
                continue
        
        print(f"DEBUG: Successfully created {len(books_list)} book objects")
        return books_list
        
    except Exception as e:
        print(f"DEBUG: Exception occurred: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing top-rated books: {str(e)}"
        )

@router.get("/books/{book_id}/genres", response_model=List[str])
async def get_genres_by_book_id(book_id: int):
    """Get genres for a specific book by its ID"""
    df_books, _, _, _ = get_model_components()

    if df_books is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Find the row matching the book ID
    book_row = df_books[df_books['bookId'] == book_id]

    if book_row.empty:
        raise HTTPException(status_code=404, detail=f"Book with ID {book_id} not found")

    # Check which column contains genre list
    if 'genres_list' in book_row.columns and not pd.isna(book_row.iloc[0]['genres_list']):
        genres = book_row.iloc[0]['genres_list']
        if isinstance(genres, str):
            genres = safe_parse_genres(genres)
        return genres if isinstance(genres, list) else []

    elif 'genres' in book_row.columns and not pd.isna(book_row.iloc[0]['genres']):
        genres = safe_parse_genres(book_row.iloc[0]['genres'])
        return genres if genres else []

    raise HTTPException(status_code=404, detail=f"Genres not found for book ID {book_id}")

@router.get("/books/genres", response_model=List[str])
async def get_all_genres():
    """Get list of all available genres"""
    df_books, _, _, _ = get_model_components()
    if df_books is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    all_genres = set()
    
    if 'genres_list' in df_books.columns:
        for genres_list in df_books['genres_list'].dropna():
            if isinstance(genres_list, list):
                all_genres.update(genres_list)
    elif 'genres' in df_books.columns:
        for genres_data in df_books['genres'].dropna():
            parsed_genres = safe_parse_genres(genres_data)
            if parsed_genres:
                all_genres.update(parsed_genres)
    
    return sorted(list(all_genres))

# GENERAL BOOKS LIST ROUTE
@router.get("/books", response_model=List[BookInfo])
async def list_books(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100)
):
    """List all books with pagination"""
    df_books, _, _, _ = get_model_components()
    
    if df_books is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    books_subset = df_books.iloc[skip:skip + limit]
    
    books_list = []
    for _, book_data in books_subset.iterrows():
        books_list.append(create_book_info(book_data))
    
    return books_list

# DYNAMIC ROUTE - THIS MUST COME AFTER ALL SPECIFIC /books/* ROUTES
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
    return create_book_info(book_data)

# RECOMMENDATION ROUTES
@router.get("/recommendations/{book_id}", response_model=RecommendationResponse)
async def get_recommendations(
    book_id: int, 
    num_recommendations: int = Query(5, ge=1, le=50, description="Number of recommendations")
):
    """Get book recommendations by book ID"""
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

# SEARCH ROUTES
@router.get("/search/books", response_model=List[BookInfo])
async def search_books(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100)
):
    """Search books by title or author"""
    df_books, _, _, _ = get_model_components()
    
    if df_books is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    query_lower = q.lower()
    mask = (
        df_books['title'].str.lower().str.contains(query_lower, na=False) |
        df_books['author'].str.lower().str.contains(query_lower, na=False)
    )
    
    search_results = df_books[mask].head(limit)
    
    books_list = []
    for _, book_data in search_results.iterrows():
        books_list.append(create_book_info(book_data))
    
    return books_list