"""
scrape.py
---------
Module 1 - Data Pipeline (/data_pipeline)

Scrapes book listings from books.toscrape.com (a public scraping-practice
site, no login/API key/paid tier required) across the first 5 paginated
listing pages of the "All products" catalogue. This reliably yields ~100
books spanning many categories, comfortably clearing the required
>= 60 rows / >= 3 categories.

For each book we capture:
    title, price (raw string as listed, GBP), star_rating (text, e.g. "Three"),
    availability (raw text), category

Output: data_pipeline/raw_books.csv

Run:
    python scrape.py
"""

import csv
import time
import sys
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
CATALOGUE_URL = "https://books.toscrape.com/catalogue/page-{page}.html"
PRODUCT_BASE = "https://books.toscrape.com/catalogue/"
NUM_PAGES = 5
OUTPUT_CSV = "raw_books.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (Zepto-Capstone-Scraper/1.0)"}


def get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def get_category_for_book(product_url: str) -> str:
    """
    Each book's own product page lists its category in the breadcrumb trail
    (Home > Books > <Category> > <Title>). Fetching it per-book is the most
    reliable way to get an accurate category label regardless of which
    listing page the book appeared on.
    """
    soup = get_soup(product_url)
    breadcrumb = soup.select("ul.breadcrumb li a")
    # breadcrumb[0] = Home, breadcrumb[1] = Books, breadcrumb[2] = Category
    if len(breadcrumb) >= 3:
        return breadcrumb[2].get_text(strip=True)
    return "Unknown"


def scrape_listing_page(page_num: int):
    url = CATALOGUE_URL.format(page=page_num)
    soup = get_soup(url)
    articles = soup.select("article.product_pod")

    rows = []
    for art in articles:
        title = art.h3.a["title"].strip()
        price_text = art.select_one("p.price_color").get_text(strip=True)
        # star rating is encoded as a CSS class, e.g. "star-rating Three"
        rating_classes = art.select_one("p.star-rating")["class"]
        star_rating = [c for c in rating_classes if c != "star-rating"][0]
        availability = art.select_one("p.instock.availability").get_text(strip=True)

        relative_link = art.h3.a["href"]
        product_url = PRODUCT_BASE + relative_link.replace("../../../", "")

        rows.append(
            {
                "title": title,
                "price": price_text,
                "star_rating": star_rating,
                "availability": availability,
                "product_url": product_url,
            }
        )
    return rows


def main():
    all_rows = []
    for page in range(1, NUM_PAGES + 1):
        print(f"Scraping listing page {page}/{NUM_PAGES} ...")
        try:
            page_rows = scrape_listing_page(page)
        except requests.RequestException as exc:
            print(f"  Failed to fetch page {page}: {exc}", file=sys.stderr)
            continue
        all_rows.extend(page_rows)
        time.sleep(0.3)  # be polite to the practice site

    print(f"Collected {len(all_rows)} book listings. Fetching category per book...")
    for i, row in enumerate(all_rows, start=1):
        try:
            row["category"] = get_category_for_book(row["product_url"])
        except requests.RequestException as exc:
            print(f"  Failed to fetch category for '{row['title']}': {exc}", file=sys.stderr)
            row["category"] = "Unknown"
        if i % 20 == 0:
            print(f"  {i}/{len(all_rows)} categories resolved")
        time.sleep(0.15)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["title", "price", "star_rating", "availability", "category"]
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(
                {
                    "title": row["title"],
                    "price": row["price"],
                    "star_rating": row["star_rating"],
                    "availability": row["availability"],
                    "category": row["category"],
                }
            )

    print(f"Saved {len(all_rows)} rows to {OUTPUT_CSV}")
    categories = sorted(set(r["category"] for r in all_rows))
    print(f"Categories found ({len(categories)}): {categories}")


if __name__ == "__main__":
    main()
