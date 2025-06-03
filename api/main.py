"""
FastAPI application entry point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import warnings

from .dependencies import load_model_components, get_model_stats
from .routes import router

# Suppress warnings
warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- FastAPI App ---
app = FastAPI(
    title="Book Recommendation API",
    description="A content-based book recommendation system using TF-IDF and cosine similarity",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Load model components when the app starts"""
    print("🚀 Starting Book Recommendation API...")
    
    success = load_model_components()
    if not success:
        print("⚠️  WARNING: Failed to load model components!")
        print("Please run the training script first:")
        print("   python models/train_model.py")
    else:
        print("✅ Model components loaded successfully!")
        
        # Print model statistics
        stats = get_model_stats()
        if stats.get("model_loaded"):
            print(f"📚 Total books: {stats['total_books']}")
            print(f"🔤 TF-IDF features: {stats['tfidf_features']}")
            print(f"📊 Matrix shape: {stats['tfidf_matrix_shape']}")

# --- Run the server ---
if __name__ == "__main__":
    print("🚀 Starting Book Recommendation API...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )