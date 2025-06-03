from Scrapper import GoodReadsScraper
import os
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# --- Constants for directories ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_DIR / "link.csv"

url = "https://www.goodreads.com/list/show/1.Best_Books_Ever"

# --- Custom method to write CSV ---
def save_links_to_csv(links, output_path):
    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Book URL'])
        for link in links:
            writer.writerow([link])

# --- Main logic ---
scrapper = GoodReadsScraper(url)
scrapper.get_book_links()

save_links_to_csv(scrapper.book_links, OUTPUT_FILE)

print(f"Book links saved to {OUTPUT_FILE}")
