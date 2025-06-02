"""
FastAPI dependencies for model management
"""
import pandas as pd
import joblib
from scipy.sparse import load_npz
from pathlib import Path
from typing import Tuple, Optional
from fastapi import HTTPException

# --- Global Variables for Model Components ---
_df_books: Optional[pd.DataFrame] = None
_tfidf_vectorizer = None
_tfidf_matrix = None
_model_loaded: bool = False

# --- Constants ---
BASE_DIR = Path(__file__).parent.parent
MODEL_OUTPUT_DIR = BASE_DIR / 'saved_model_components'
PROCESSED_DF_PATH = 'processed_books_data.csv'
TFIDF_VECTORIZER_PATH = 'tfidf_vectorizer.joblib'
TFIDF_MATRIX_PATH = 'tfidf_matrix.npz'

def load_model_components() -> bool:
    """Load all saved model components"""
    global _df_books, _tfidf_vectorizer, _tfidf_matrix, _model_loaded
    
    if _model_loaded:
        return True
    
    try:
        # Load processed DataFrame
        df_path = MODEL_OUTPUT_DIR / PROCESSED_DF_PATH
        _df_books = pd.read_csv(df_path)
        print(f"Loaded DataFrame with {len(_df_books)} books")
        
        # Load TF-IDF vectorizer
        vectorizer_path = MODEL_OUTPUT_DIR / TFIDF_VECTORIZER_PATH
        _tfidf_vectorizer = joblib.load(vectorizer_path)
        print("Loaded TF-IDF vectorizer")
        
        # Load TF-IDF matrix
        matrix_path = MODEL_OUTPUT_DIR / TFIDF_MATRIX_PATH
        _tfidf_matrix = load_npz(matrix_path)
        print(f"Loaded TF-IDF matrix with shape: {_tfidf_matrix.shape}")
        
        _model_loaded = True
        return True
        
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        print(f"Looking in directory: {MODEL_OUTPUT_DIR}")
        print(f"Make sure to run the training script first!")
        _model_loaded = False
        return False
    except Exception as e:
        print(f"Error loading model components: {e}")
        _model_loaded = False
        return False

def get_model_components() -> Tuple[Optional[pd.DataFrame], object, object, Path]:
    """
    Dependency function to get model components
    Returns: (df_books, tfidf_vectorizer, tfidf_matrix, model_dir)
    """
    if not _model_loaded:
        load_model_components()
    
    return _df_books, _tfidf_vectorizer, _tfidf_matrix, MODEL_OUTPUT_DIR

def require_model_loaded():
    """Dependency that ensures model is loaded or raises an exception"""
    df_books, tfidf_vectorizer, tfidf_matrix, _ = get_model_components()
    
    if df_books is None or tfidf_vectorizer is None or tfidf_matrix is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please ensure the training has been completed and model files exist."
        )
    
    return df_books, tfidf_vectorizer, tfidf_matrix

def get_model_stats() -> dict:
    """Get statistics about the loaded model"""
    df_books, tfidf_vectorizer, tfidf_matrix, model_dir = get_model_components()
    
    if not _model_loaded:
        return {
            "model_loaded": False,
            "error": "Model not loaded"
        }
    
    stats = {
        "model_loaded": True,
        "total_books": len(df_books) if df_books is not None else 0,
        "tfidf_features": len(tfidf_vectorizer.vocabulary_) if tfidf_vectorizer else 0,
        "tfidf_matrix_shape": list(_tfidf_matrix.shape) if _tfidf_matrix is not None else None,
        "model_directory": str(model_dir),
        "files_exist": {
            "dataframe": (model_dir / PROCESSED_DF_PATH).exists(),
            "vectorizer": (model_dir / TFIDF_VECTORIZER_PATH).exists(),
            "matrix": (model_dir / TFIDF_MATRIX_PATH).exists()
        }
    }
    
    if df_books is not None:
        stats.update({
            "sample_book_ids": df_books['bookId_lookup'].head(10).tolist(),
            "columns": list(df_books.columns),
            "memory_usage": df_books.memory_usage(deep=True).sum()
        })
    
    return stats

def reload_model():
    """Force reload the model components"""
    global _model_loaded
    _model_loaded = False
    return load_model_components()

def unload_model():
    """Unload model components from memory"""
    global _df_books, _tfidf_vectorizer, _tfidf_matrix, _model_loaded
    
    _df_books = None
    _tfidf_vectorizer = None
    _tfidf_matrix = None
    _model_loaded = False
    
    print("Model components unloaded from memory")