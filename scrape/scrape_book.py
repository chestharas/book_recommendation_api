from pathlib import Path
from Scrapper import GoodReadsScraper

# Define base and data directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)

# File paths
LINKS_FILE = DATA_DIR / 'book_links.csv'
BOOKS_FILE = DATA_DIR / 'books.csv'

# Goodreads URL (for completeness)
url = "https://www.goodreads.com/list/show/1.Best_Books_Ever"

# Initialize scraper
scraper = GoodReadsScraper(url)

# Load book links from CSV
if LINKS_FILE.exists():
    scraper.load_book_links_from_csv(file=LINKS_FILE)
else:
    print(f"{LINKS_FILE} not found. Please generate book links first.")
    exit()

# Scrape book details using multithreading
scraper.get_books_multithreaded(max_workers=10)

# Save to books.csv
scraper.books_to_csv(file=BOOKS_FILE)

print(f"✅ Finished scraping. Book details saved to: {BOOKS_FILE}")
