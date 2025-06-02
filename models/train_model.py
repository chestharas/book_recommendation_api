import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os
import joblib # For saving/loading sklearn objects
from scipy.sparse import save_npz, load_npz # For saving/loading sparse matrices
from pathlib import Path

# --- Constants for filenames ---
BASE_DIR = Path(__file__).parent.parent  # Go up one level from models/ to project root
DATA_DIR = BASE_DIR / 'data'
MODEL_OUTPUT_DIR = BASE_DIR / 'saved_model_components'

PROCESSED_DF_PATH = 'processed_books_data.csv'
TFIDF_VECTORIZER_PATH = 'tfidf_vectorizer.joblib'
TFIDF_MATRIX_PATH = 'tfidf_matrix.npz'

# --- 0. Create output directory ---
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

print(f"Base directory: {BASE_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Model output directory: {MODEL_OUTPUT_DIR}")

# --- 1. Load Data ---
csv_file_path = DATA_DIR / 'books.csv'  # Updated path
try:
    df = pd.read_csv(csv_file_path)
    print(f"Successfully loaded '{csv_file_path}'.")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
except FileNotFoundError:
    print(f"Error: '{csv_file_path}' not found. Please ensure the path is correct and the file exists.")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for file at: {csv_file_path.absolute()}")
    exit()
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")
    exit()

# --- 2. Clean Data ---
print(f"Number of books originally: {len(df)}")

required_columns = ['bookId', 'title', 'author', 'description']
for col in required_columns:
    if col not in df.columns:
        print(f"Error: Essential column '{col}' not found in the CSV. Please check your column names.")
        print(f"Available columns: {list(df.columns)}")
        exit()

df.drop_duplicates(subset=['title', 'author'], keep='first', inplace=True)
print(f"Number of books after deduplication: {len(df)}")

df.reset_index(drop=True, inplace=True) # Crucial for matching TF-IDF matrix rows later

try:
    df['bookId_numeric'] = pd.to_numeric(df['bookId'], errors='coerce')
    df.dropna(subset=['bookId_numeric'], inplace=True)
    df['bookId_lookup'] = df['bookId_numeric'].astype(int)
    print(f"Number of books after numeric ID conversion: {len(df)}")
    # Ensure bookId_lookup is unique if we are going to use it as a primary key for lookup
    # If bookId_lookup is not unique after conversion, we might need to rethink.
    # For now, we assume the first match is fine as done in the original script.
    # If multiple original bookIds map to the same numeric ID, the first one encountered will be used.
except Exception as e:
    print(f"Error processing 'bookId' column: {e}. Ensure 'bookId' contains values that can be interpreted as numbers for lookup.")
    exit()

df['description'] = df['description'].fillna('')

if 'genres' in df.columns:
    def parse_genres(genre_str):
        try:
            if pd.isna(genre_str): return []
            return ast.literal_eval(genre_str)
        except (ValueError, SyntaxError): return []
    df['genres_list'] = df['genres'].apply(parse_genres)
    print("Parsed genres column")
else:
    df['genres_list'] = pd.Series([[] for _ in range(len(df))])
    print("No genres column found, created empty genres_list")

print(f"Final dataset shape: {df.shape}")

# --- 3. TF-IDF Vectorization on Descriptions ---
print("\n--- Creating TF-IDF Vectors ---")
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    min_df=1, # Min_df should ideally be > 1 for robustness, but 1 is fine for small datasets
    ngram_range=(1, 2)
)

if df.empty or df['description'].str.strip().eq('').all():
    print("Error: No descriptions available for TF-IDF.")
    exit()

# Check for empty descriptions
empty_descriptions = df['description'].str.strip().eq('').sum()
print(f"Number of empty descriptions: {empty_descriptions}")

tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])
print(f"Shape of TF-IDF matrix: {tfidf_matrix.shape}")
print(f"Shape of DataFrame: {df.shape}") # Ensure df rows match tfidf_matrix rows
print(f"TF-IDF vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

# --- 4. Recommendation Function (Modified to include bookId in output) ---
def get_recommendations_tfidf_by_index(input_book_idx, df_books, tfidf_matrix_all, num_recommendations=5):
    if not (0 <= input_book_idx < len(df_books)):
        print(f"Error: Invalid book index {input_book_idx}.")
        return []
    if input_book_idx >= tfidf_matrix_all.shape[0]:
        print(f"Error: Book index {input_book_idx} out of bounds for TF-IDF matrix.")
        return []

    input_book_vector = tfidf_matrix_all[input_book_idx]
    cosine_similarities = cosine_similarity(input_book_vector, tfidf_matrix_all)[0]

    similarity_scores_with_indices = []
    for sim_idx, score in enumerate(cosine_similarities):
        if sim_idx != input_book_idx: # Exclude the input book itself
            similarity_scores_with_indices.append((sim_idx, score))

    similarity_scores_with_indices = sorted(similarity_scores_with_indices, key=lambda x: x[1], reverse=True)

    recommendations = []
    for rec_df_idx, rec_score in similarity_scores_with_indices[:num_recommendations]:
        recommended_book_data = df_books.iloc[rec_df_idx]
        recommendations.append({
            'bookId': recommended_book_data['bookId'], # Original bookId for display
            'title': recommended_book_data['title'],
            'author': recommended_book_data['author'],
            'score': float(rec_score) # Ensure score is JSON serializable (float)
        })
    return recommendations

# --- 5. Save Model Components ---
if not df.empty and tfidf_matrix.shape[0] > 0:
    print("\n--- Saving Model Components ---")
    try:
        # Save the processed DataFrame (contains bookId, title, author, bookId_lookup)
        # This DataFrame's index aligns with the rows of tfidf_matrix
        df_save_path = MODEL_OUTPUT_DIR / PROCESSED_DF_PATH
        df.to_csv(df_save_path, index=False) # index=False is important
        print(f"Processed DataFrame saved to '{df_save_path}'")

        # Save the TF-IDF vectorizer
        vectorizer_save_path = MODEL_OUTPUT_DIR / TFIDF_VECTORIZER_PATH
        joblib.dump(tfidf_vectorizer, vectorizer_save_path)
        print(f"TF-IDF Vectorizer saved to '{vectorizer_save_path}'")

        # Save the TF-IDF matrix
        matrix_save_path = MODEL_OUTPUT_DIR / TFIDF_MATRIX_PATH
        save_npz(matrix_save_path, tfidf_matrix)
        print(f"TF-IDF Matrix saved to '{matrix_save_path}'")
        print("--- Model components saved successfully. ---")

        # Print some statistics
        print(f"\n--- Training Summary ---")
        print(f"Total books processed: {len(df)}")
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        print(f"Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
        print(f"Average description length: {df['description'].str.len().mean():.1f} characters")

    except Exception as e:
        print(f"Error saving model components: {e}")
else:
    print("Skipping model saving as DataFrame is empty or TF-IDF matrix could not be generated.")


# --- 6. Test the Model (Optional) ---
def test_recommendations():
    if df.empty or tfidf_matrix.shape[0] == 0:
        print("Cannot test recommendations: no data available")
        return
    
    print("\n--- Testing Recommendations ---")
    # Test with the first book
    test_book = df.iloc[0]
    print(f"Testing with book: '{test_book['title']}' by {test_book['author']}")
    
    recommendations = get_recommendations_tfidf_by_index(
        input_book_idx=0,
        df_books=df,
        tfidf_matrix_all=tfidf_matrix,
        num_recommendations=3
    )
    
    if recommendations:
        print("Top 3 recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['title']} by {rec['author']} (Score: {rec['score']:.4f})")
    else:
        print("No recommendations found")

# Run test
test_recommendations()

# --- 7. Main Interaction Loop (for testing, can be removed if only saving) ---
if not df.empty and tfidf_matrix.shape[0] > 0 :
    print("\n--- Interactive Testing Mode ---")
    print("You can now test the recommendation system!")
    
    while True:
        user_input_id_str = input("\nEnter a bookId to get recommendations (or type 'exit' to quit): ").strip()

        if user_input_id_str.lower() == 'exit':
            print("Exiting recommender.")
            break

        try:
            target_book_id_for_lookup = int(user_input_id_str)
        except ValueError:
            print("Invalid ID format. Please enter a numeric ID.")
            continue

        target_book_matches = df[df['bookId_lookup'] == target_book_id_for_lookup]

        if target_book_matches.empty:
            print(f"Book with ID '{target_book_id_for_lookup}' not found in the dataset.")
            # Show some available IDs
            sample_ids = df['bookId_lookup'].head(10).tolist()
            print(f"Available sample IDs: {sample_ids}")
            continue

        selected_book_data = target_book_matches.iloc[0]
        input_book_original_id = selected_book_data['bookId']
        input_book_title = selected_book_data['title']
        input_book_df_idx = selected_book_data.name # This is the DataFrame index

        print(f"\n--- Getting recommendations for: '{input_book_title}' (ID: {input_book_original_id}) ---")
        recommendations = get_recommendations_tfidf_by_index(
            input_book_idx=input_book_df_idx,
            df_books=df, # The current, in-memory df
            tfidf_matrix_all=tfidf_matrix, # The current, in-memory tfidf_matrix
            num_recommendations=5
        )

        if recommendations:
            print("Recommendations:")
            for i, rec in enumerate(recommendations):
                print(f"{i+1}. ID: {rec['bookId']}, Title: {rec['title']} by {rec['author']} (Similarity: {rec['score']:.4f})")
        else:
            print(f"Found '{input_book_title}', but no other similar books to recommend.")
            print(f"(Debug: Input book DF index: {input_book_df_idx}, Total unique books: {len(df)})")
else:
    print("DataFrame is empty or TF-IDF matrix is empty. Cannot run interaction loop or save components.")

print("\n--- Training Complete ---")
print("You can now run the FastAPI server with: python api/main.py")
print("Or from the project root: uvicorn api.main:app --reload")