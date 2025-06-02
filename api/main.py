"""
Simple, working FastAPI application with all book data
"""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import joblib
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import uvicorn
from pathlib import Path
import ast

# Suppress warnings
warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- Pydantic Models ---
class BookRecommendation(BaseModel):
    """Model for a single book recommendation"""
    bookId: str
    title: str
    author: str
    rating: Optional[float] = None
    ratingsCount: Optional[int] = None
    genres: Optional[List[str]] = None
    coverImg: Optional[str] = None
    score: float

class RecommendationResponse(BaseModel):
    """Response model for book recommendations"""
    input_book: dict
    recommendations: List[BookRecommendation]
    total_recommendations: int

class BookInfo(BaseModel):
    """Model for complete book information"""
    model_config = {"protected_namespaces": ()}  # Fix Pydantic warning
    
    bookId: str
    title: str
    author: str
    rating: Optional[float] = None
    ratingsCount: Optional[int] = None
    reviewsCount: Optional[int] = None
    description: Optional[str] = ""
    genres: Optional[List[str]] = None
    coverImg: Optional[str] = None
    publishedDate: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    model_config = {"protected_namespaces": ()}  # Fix Pydantic warning
    
    status: str
    model_loaded: bool
    total_books: int
    model_directory: str

# --- FastAPI App ---
app = FastAPI(
    title="Book Recommendation API",
    description="A content-based book recommendation system using TF-IDF and cosine similarity",
    version="1.0.0"
)

# --- Global Variables ---
df_books = None
tfidf_vectorizer = None
tfidf_matrix = None

# --- Constants ---
BASE_DIR = Path(__file__).parent.parent
MODEL_OUTPUT_DIR = BASE_DIR / 'saved_model_components'

def load_model_components():
    """Load all saved model components"""
    global df_books, tfidf_vectorizer, tfidf_matrix
    
    try:
        # Load processed DataFrame
        df_path = MODEL_OUTPUT_DIR / 'processed_books_data.csv'
        df_books = pd.read_csv(df_path)
        print(f"✅ Loaded DataFrame with {len(df_books)} books")
        
        # Load TF-IDF vectorizer
        vectorizer_path = MODEL_OUTPUT_DIR / 'tfidf_vectorizer.joblib'
        tfidf_vectorizer = joblib.load(vectorizer_path)
        print("✅ Loaded TF-IDF vectorizer")
        
        # Load TF-IDF matrix
        matrix_path = MODEL_OUTPUT_DIR / 'tfidf_matrix.npz'
        tfidf_matrix = load_npz(matrix_path)
        print(f"✅ Loaded TF-IDF matrix with shape: {tfidf_matrix.shape}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
        return False
    except Exception as e:
        print(f"❌ Error loading model components: {e}")
        return False

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

def get_recommendations_by_book_id(book_id: int, num_recommendations: int = 5):
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

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Load model components when the app starts"""
    print("🚀 Starting Book Recommendation API...")
    print(f"📁 Model directory: {MODEL_OUTPUT_DIR}")
    
    success = load_model_components()
    if not success:
        print("⚠️  WARNING: Failed to load model components!")
        print("Please run the training script first:")
        print("   python models/train_model.py")
    else:
        print("✅ Model components loaded successfully!")
        if df_books is not None:
            print(f"📚 Total books: {len(df_books)}")

# --- API Endpoints ---

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Book Recommendation API",
        "version": "1.0.0",
        "total_books": len(df_books) if df_books is not None else 0,
        "endpoints": {
            "recommendations": "/recommendations/{book_id}",
            "book_info": "/books/{book_id}",
            "list_books": "/books",
            "filter_books": "/books/filter",
            "top_rated": "/books/top-rated",
            "search": "/search/books",
            "health": "/health"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_loaded = all([df_books is not None, tfidf_vectorizer is not None, tfidf_matrix is not None])
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        total_books=len(df_books) if df_books is not None else 0,
        model_directory=str(MODEL_OUTPUT_DIR)
    )

@app.get("/recommendations/{book_id}", response_model=RecommendationResponse)
async def get_recommendations(
    book_id: int, 
    num_recommendations: int = Query(5, ge=1, le=50, description="Number of recommendations")
):
    """Get book recommendations by book ID"""
    return get_recommendations_by_book_id(book_id, num_recommendations)

@app.get("/books/{book_id}", response_model=BookInfo)
async def get_book_info(book_id: int):
    """Get information about a specific book"""
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

@app.get("/books", response_model=List[BookInfo])
async def list_books(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100)
):
    """List all books with pagination"""
    if df_books is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    books_subset = df_books.iloc[skip:skip + limit]
    
    books_list = []
    for _, book_data in books_subset.iterrows():
        books_list.append(create_book_info(book_data))
    
    return books_list

@app.get("/books/filter", response_model=List[BookInfo])
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

@app.get("/books/top-rated", response_model=List[BookInfo])
async def get_top_rated_books(
    min_ratings: int = Query(10, ge=1),
    limit: int = Query(20, ge=1, le=100)
):
    """Get top-rated books"""
    if df_books is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if 'rating' not in df_books.columns or 'ratingsCount' not in df_books.columns:
        raise HTTPException(status_code=400, detail="Rating data not available")
    
    top_books = df_books[
        (df_books['ratingsCount'].fillna(0) >= min_ratings) & 
        (df_books['rating'].notna())
    ].sort_values(['rating', 'ratingsCount'], ascending=[False, False]).head(limit)
    
    books_list = []
    for _, book_data in top_books.iterrows():
        books_list.append(create_book_info(book_data))
    
    return books_list

@app.get("/search/books", response_model=List[BookInfo])
async def search_books(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100)
):
    """Search books by title or author"""
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

@app.get("/books/genres", response_model=List[str])
async def get_all_genres():
    """Get list of all available genres"""
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

# --- Run the server ---
if __name__ == "__main__":
    print("🚀 Starting Book Recommendation API...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )