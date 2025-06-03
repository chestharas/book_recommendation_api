import os
import csv
import time
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

class GoodReadsScraper:
    def __init__(self, list_url):
        self.book_links = []
        self.books = []
        self.list_url = list_url

    # -------------------------- Book Attribute Parsers --------------------------
    def __get_book_id(self, url):
        last_part = url.split("/")[-1]
        return last_part.split('-')[0] if '-' in last_part else ''.join(filter(str.isdigit, last_part))

    def __get_title(self, soup):
        title_element = soup.find("h1", class_="Text Text__title1")
        return title_element.get_text(strip=True) if title_element else "Unknown Title"

    def __get_author(self, soup):
        author_element = soup.find("a", class_="ContributorLink")
        return author_element.find("span", class_="ContributorLink__name").get_text(strip=True) if author_element else "Unknown Author"

    def __get_rating(self, soup):
        rating_element = soup.find("div", class_="RatingStatistics__rating")
        return rating_element.get_text(strip=True) if rating_element else "N/A"

    def __get_genres(self, soup):
        genres = []
        genres_container = soup.find("div", {"data-testid": "genresList"})
        if genres_container:
            genre_buttons = genres_container.find_all("a", class_="Button--tag")
            genres.extend([btn.get_text(strip=True) for btn in genre_buttons])

            show_all = genres_container.find("a", class_="Button--link")
            if show_all:
                show_all_url = "https://www.goodreads.com" + show_all['href']
                try:
                    response = self.retry_request(show_all_url)
                    show_all_soup = BeautifulSoup(response.text, 'html.parser')
                    additional_genres = show_all_soup.find_all("a", class_="Button--tag")
                    genres.extend([btn.get_text(strip=True) for btn in additional_genres])
                except Exception:
                    pass  # Skip loading additional genres on failure
        return genres

    def __get_description(self, soup):
        desc = soup.find("div", class_="DetailsLayoutRightParagraph__widthConstrained")
        return desc.get_text(strip=True) if desc else "No description available."

    def __get_ratings_count(self, soup):
        count = soup.find("span", {"data-testid": "ratingsCount"})
        return ''.join(filter(str.isdigit, count.get_text(strip=True))) if count else "0"

    def __get_reviews_count(self, soup):
        count = soup.find("span", {"data-testid": "reviewsCount"})
        return ''.join(filter(str.isdigit, count.get_text(strip=True))) if count else "0"

    def __get_first_publish_date(self, soup):
        try:
            pub_info = soup.find("p", {"data-testid": "publicationInfo"}).get_text(strip=True)
            return pub_info.split("First published ")[-1]
        except AttributeError:
            return ""

    def __get_cover_img_url(self, soup):
        img = soup.find("img", class_="ResponsiveImage")
        return img['src'] if img else "https://dryofg8nmyqjw.cloudfront.net/images/no-cover.png"

    # ------------------------ Robust Retry Function ------------------------

    def retry_request(self, url, max_retries=9999, backoff_factor=2):
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                return response
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
                wait_time = min(60, backoff_factor ** retry_count + random.uniform(0, 2))
                print(f"[Retry] Network issue or rate-limited. Retrying in {int(wait_time)}s...")
                time.sleep(wait_time)
                retry_count += 1
        raise ConnectionError(f"Failed to retrieve {url} after {max_retries} retries.")

    # --------------------------- Book Link Scraping ---------------------------
    def get_book_links(self):
        current_url = self.list_url
        page_num = 1

        while current_url:
            print(f"Scraping page {page_num}...")
            response = self.retry_request(current_url)
            soup = BeautifulSoup(response.text, 'html.parser')

            book_entries = soup.find_all('a', class_='bookTitle')
            if not book_entries:
                break

            self.book_links += ["https://www.goodreads.com" + entry['href'] for entry in book_entries]

            next_page = soup.find('a', class_='next_page')
            current_url = "https://www.goodreads.com" + next_page['href'] if next_page else None
            page_num += 1
            time.sleep(1)

        print(f"Scraped {len(self.book_links)} book links.")
        self.links_to_csv("book_links.csv")

    def links_to_csv(self, file):
        with open(file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Book URL"])
            for link in self.book_links:
                writer.writerow([link])
        print(f"Book links saved to {file}.")

    def load_book_links_from_csv(self, file='book_links.csv'):
        try:
            with open(file, mode='r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                next(csv_reader)
                self.book_links = [row[0] for row in csv_reader]
            print(f"Loaded {len(self.book_links)} book links from {file}")
            return True
        except FileNotFoundError:
            print(f"No existing book links file found ({file})")
            return False

    # ---------------------------- Multithreaded Scraping ----------------------------

    def scrape_single_book(self, i, book_url):
        try:
            response = self.retry_request(book_url)
            soup = BeautifulSoup(response.text, 'html.parser')

            book = {
                "bookId": self.__get_book_id(book_url),
                "title": self.__get_title(soup),
                "author": self.__get_author(soup),
                "rating": self.__get_rating(soup),
                "ratingsCount": self.__get_ratings_count(soup),
                "reviewsCount": self.__get_reviews_count(soup),
                "description": self.__get_description(soup),
                "genres": self.__get_genres(soup),
                "coverImg": self.__get_cover_img_url(soup),
                "publishedDate": self.__get_first_publish_date(soup),
            }
            print(f"[{i+1}] Collected: {book['title']} by {book['author']}")
            return book
        except Exception as e:
            print(f"[{i+1}] Failed: {book_url}, Reason: {str(e)}")
            return None

    def get_books_multithreaded(self, max_workers=10):
        if not self.book_links:
            print("No book links loaded.")
            return

        self.books = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.scrape_single_book, i, url): i
                for i, url in enumerate(self.book_links)
            }

            for future in as_completed(futures):
                book = future.result()
                if book:
                    self.books.append(book)

        self.books_to_csv("books.csv")
        print("Scraping complete.")

    def books_to_csv(self, file):
        if not self.books:
            raise ValueError("No books to save.")

        keys = self.books[0].keys()
        file_exists = os.path.isfile(file)
        mode = 'a' if file_exists else 'w'

        with open(file, mode=mode, newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, keys, quoting=csv.QUOTE_NONNUMERIC)
            if not file_exists:
                writer.writeheader()

            if file_exists:
                with open(file, mode='r', encoding='utf-8-sig') as read_f:
                    reader = csv.reader(read_f)
                    next(reader)
                    saved_count = sum(1 for _ in reader)
                writer.writerows(self.books[saved_count:])
            else:
                writer.writerows(self.books)

        print(f"{'Updated' if file_exists else 'Created'} {file} with {len(self.books)} books.")


if __name__ == "__main__":
    list_url = "https://www.goodreads.com/list/show/1.Best_Books_Ever"  # Example URL
    scraper = GoodReadsScraper(list_url)

    scraper.get_book_links()
    
    