import os
from pathlib import Path
from typing import Optional

class Settings:
    """Application settings and configuration"""
    
    # Base directories
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    MODEL_DIR = BASE_DIR / 'saved_model_components'
    
    # API settings
    API_HOST: str = os.getenv('API_HOST', '0.0.0.0')
    API_PORT: int = int(os.getenv('API_PORT', '8000'))
    DEBUG: bool = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # Model file names
    PROCESSED_DF_PATH: str = 'processed_books_data.csv'
    TFIDF_VECTORIZER_PATH: str = 'tfidf_vectorizer.joblib'
    TFIDF_MATRIX_PATH: str = 'tfidf_matrix.npz'
    
    # Data file
    BOOKS_CSV_PATH: Path = DATA_DIR / 'books.csv'
    
    # Recommendation settings
    MAX_RECOMMENDATIONS: int = int(os.getenv('MAX_RECOMMENDATIONS', '50'))
    DEFAULT_RECOMMENDATIONS: int = int(os.getenv('DEFAULT_RECOMMENDATIONS', '5'))
    MIN_RECOMMENDATIONS: int = 1
    
    # TF-IDF settings
    TFIDF_MIN_DF: int = int(os.getenv('TFIDF_MIN_DF', '1'))
    TFIDF_NGRAM_RANGE: tuple = (1, 2)
    TFIDF_STOP_WORDS: str = 'english'
    
    # Required columns in the dataset
    REQUIRED_COLUMNS: list = ['bookId', 'title', 'author', 'description']
    
    # Optional columns with default handling
    OPTIONAL_COLUMNS: list = ['rating', 'ratingsCount', 'reviewsCount', 'genres', 'coverImg', 'publishedDate']
    
    # All expected columns
    ALL_COLUMNS: list = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    
    # API metadata
    API_TITLE: str = "Book Recommendation API"
    API_DESCRIPTION: str = "A content-based book recommendation system using TF-IDF and cosine similarity"
    API_VERSION: str = "1.0.0"
    
    @classmethod
    def get_model_file_path(cls, filename: str) -> Path:
        """Get full path to a model file"""
        return cls.MODEL_DIR / filename
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.MODEL_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def validate_environment(cls) -> dict:
        """Validate the environment and return status"""
        status = {
            'data_dir_exists': cls.DATA_DIR.exists(),
            'model_dir_exists': cls.MODEL_DIR.exists(),
            'books_csv_exists': cls.BOOKS_CSV_PATH.exists(),
            'model_files_exist': {
                'dataframe': (cls.MODEL_DIR / cls.PROCESSED_DF_PATH).exists(),
                'vectorizer': (cls.MODEL_DIR / cls.TFIDF_VECTORIZER_PATH).exists(),
                'matrix': (cls.MODEL_DIR / cls.TFIDF_MATRIX_PATH).exists()
            }
        }
        
        status['all_model_files_exist'] = all(status['model_files_exist'].values())
        status['ready_for_training'] = status['data_dir_exists'] and status['books_csv_exists']
        status['ready_for_api'] = status['model_dir_exists'] and status['all_model_files_exist']
        
        return status

# Create a global settings instance
settings = Settings()

# Environment validation function
def check_environment():
    """Check and print environment status"""
    status = settings.validate_environment()
    
    print("=== Environment Status ===")
    print(f"Base directory: {settings.BASE_DIR}")
    print(f"Data directory exists: {status['data_dir_exists']}")
    print(f"Model directory exists: {status['model_dir_exists']}")
    print(f"Books CSV exists: {status['books_csv_exists']}")
    print(f"Model files exist:")
    for file_type, exists in status['model_files_exist'].items():
        print(f"  - {file_type}: {exists}")
    
    print(f"\nReady for training: {status['ready_for_training']}")
    print(f"Ready for API: {status['ready_for_api']}")
    
    if not status['ready_for_training']:
        print("\n⚠️  To prepare for training:")
        if not status['data_dir_exists']:
            print(f"  - Create data directory: {settings.DATA_DIR}")
        if not status['books_csv_exists']:
            print(f"  - Add books.csv to: {settings.BOOKS_CSV_PATH}")
    
    if not status['ready_for_api']:
        print("\n⚠️  To prepare for API:")
        print("  - Run training script: python models/train_model.py")
    
    return status

if __name__ == "__main__":
    check_environment()