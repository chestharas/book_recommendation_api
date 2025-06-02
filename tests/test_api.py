import pytest
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings

class TestModel:
    """Test cases for model functionality"""
    
    def setup_method(self):
        """Set up test data"""
        self.sample_data = pd.DataFrame({
            'bookId': ['1', '2', '3', '4', '5'],
            'title': [
                'The Great Gatsby',
                'To Kill a Mockingbird', 
                'Pride and Prejudice',
                '1984',
                'The Catcher in the Rye'
            ],
            'author': [
                'F. Scott Fitzgerald',
                'Harper Lee',
                'Jane Austen', 
                'George Orwell',
                'J.D. Salinger'
            ],
            'description': [
                'A classic American novel about the Jazz Age',
                'A story about racial injustice in the American South',
                'A romantic novel about manners and marriage',
                'A dystopian novel about totalitarian government',
                'A coming-of-age story about teenage alienation'
            ]
        })
    
    def test_data_preprocessing(self):
        """Test data preprocessing steps"""
        df = self.sample_data.copy()
        
        # Test numeric conversion
        df['bookId_numeric'] = pd.to_numeric(df['bookId'], errors='coerce')
        df['bookId_lookup'] = df['bookId_numeric'].astype(int)
        
        assert df['bookId_lookup'].dtype == int
        assert not df['bookId_lookup'].isna().any()
    
    def test_tfidf_vectorization(self):
        """Test TF-IDF vectorization"""
        descriptions = self.sample_data['description'].fillna('')
        
        vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=1,
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        
        assert tfidf_matrix.shape[0] == len(descriptions)
        assert tfidf_matrix.shape[1] > 0
        assert len(vectorizer.vocabulary_) > 0
    
    def test_similarity_calculation(self):
        """Test cosine similarity calculation"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        descriptions = self.sample_data['description'].fillna('')
        
        vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        
        # Calculate similarities for first book
        similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix)[0]
        
        assert len(similarities) == len(descriptions)
        assert similarities[0] == pytest.approx(1.0, rel=1e-5)  # Self-similarity should be 1
        assert all(0 <= sim <= 1 for sim in similarities)  # All similarities should be between 0 and 1
    
    def test_recommendation_logic(self):
        """Test recommendation generation logic"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        df = self.sample_data.copy()
        df['bookId_numeric'] = pd.to_numeric(df['bookId'], errors='coerce')
        df['bookId_lookup'] = df['bookId_numeric'].astype(int)
        
        descriptions = df['description'].fillna('')
        vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        
        # Test recommendation for first book
        input_book_idx = 0
        input_book_vector = tfidf_matrix[input_book_idx]
        cosine_similarities = cosine_similarity(input_book_vector, tfidf_matrix)[0]
        
        # Get recommendations (excluding input book)
        similarity_scores_with_indices = []
        for sim_idx, score in enumerate(cosine_similarities):
            if sim_idx != input_book_idx:
                similarity_scores_with_indices.append((sim_idx, score))
        
        similarity_scores_with_indices = sorted(
            similarity_scores_with_indices, 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Build recommendations
        recommendations = []
        for rec_df_idx, rec_score in similarity_scores_with_indices[:3]:
            recommended_book_data = df.iloc[rec_df_idx]
            recommendations.append({
                'bookId': recommended_book_data['bookId'],
                'title': recommended_book_data['title'],
                'author': recommended_book_data['author'],
                'score': float(rec_score)
            })
        
        assert len(recommendations) <= 3
        assert all('bookId' in rec for rec in recommendations)
        assert all('title' in rec for rec in recommendations)
        assert all('score' in rec for rec in recommendations)
        assert all(isinstance(rec['score'], float) for rec in recommendations)
    
    def test_empty_descriptions(self):
        """Test handling of empty descriptions"""
        df = self.sample_data.copy()
        df.loc[0, 'description'] = ''  # Make first description empty
        
        descriptions = df['description'].fillna('')
        
        vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        
        # Should still work even with empty descriptions
        assert tfidf_matrix.shape[0] == len(descriptions)
    
    def test_settings_validation(self):
        """Test settings and configuration"""
        assert settings.BASE_DIR.exists()
        assert settings.MAX_RECOMMENDATIONS > 0
        assert settings.DEFAULT_RECOMMENDATIONS > 0
        assert len(settings.REQUIRED_COLUMNS) > 0
        assert 'bookId' in settings.REQUIRED_COLUMNS
        assert 'title' in settings.REQUIRED_COLUMNS
        assert 'author' in settings.REQUIRED_COLUMNS
        assert 'description' in settings.REQUIRED_COLUMNS

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])