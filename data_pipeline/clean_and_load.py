"""
clean_and_load.py
------------------
Module 1 - Data Pipeline (/data_pipeline)

Reads raw_books.csv (produced by scrape.py), cleans and types every field,
converts price to INR using the project's fixed baseline rate, and loads
everything into a normalized two-table SQLite database (books.db).

Cleaning rules:
  - price:        strip the "£" symbol -> float column price_gbp
  - star_rating:  map English word ("One".."Five") -> int column rating (1-5)
  - availability: parse free-text ("In stock (19 available)" / "Out of stock")
                   -> bool column in_stock
  - Any row where a field fails to parse is DROPPED (not imputed), because
    for this dataset a parse failure means the scraper picked up a malformed
    / unexpected listing (e.g. a promotional tile), and there is no reliable
    "typical" value to impute for a whole book's price or rating — imputing
    a fabricated price/rating for a specific named book would misrepresent
    that book, whereas dropping the handful of malformed rows (if any) does
    not materially affect the >=60-row / >=3-category requirement given we
    scrape ~100 books. This choice, and the count of rows dropped, is logged
    below and restated in the README.

Currency conversion:
  Fixed, project-defined baseline rate: 1 GBP = 105.50 INR
  This is NOT a live/historical market rate and needs no date reference or
  network lookup - it is simply applied as a constant.

Schema (normalized, two tables, PK/FK relationship):
  categories(category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE)
  books(book_id INTEGER PRIMARY KEY, title TEXT, price_gbp REAL,
        price_inr REAL, rating INTEGER, in_stock INTEGER,
        category_id INTEGER REFERENCES categories(category_id))

Run:
    python clean_and_load.py
"""

import re
import sqlite3
import pandas as pd

RAW_CSV = "raw_books.csv"
DB_PATH = "books.db"
GBP_TO_INR = 105.50  # fixed project-defined baseline rate, see docstring above

RATING_WORD_TO_INT = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def parse_price(raw: str):
    """'£51.77' -> 51.77 (float). Returns None if it cannot be parsed."""
    if not isinstance(raw, str):
        return None
    match = re.search(r"[\d]+\.?\d*", raw.replace(",", ""))
    return float(match.group()) if match else None


def parse_rating(raw: str):
    """'Three' -> 3 (int, 1-5). Returns None if not a recognised word."""
    return RATING_WORD_TO_INT.get(str(raw).strip())


def parse_availability(raw: str):
    """'In stock (19 available)' -> True, 'Out of stock' -> False."""
    if not isinstance(raw, str):
        return None
    return "in stock" in raw.lower() or "available" in raw.lower() and "out of" not in raw.lower()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_gbp"] = df["price"].apply(parse_price)
    df["rating"] = df["star_rating"].apply(parse_rating)
    df["in_stock"] = df["availability"].apply(parse_availability)

    before = len(df)
    bad_mask = df["price_gbp"].isna() | df["rating"].isna() | df["in_stock"].isna() | df["title"].isna()
    n_dropped = int(bad_mask.sum())
    if n_dropped:
        print(f"Dropping {n_dropped} row(s) that failed to parse (of {before}). See README for justification.")
    df = df[~bad_mask].reset_index(drop=True)

    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)
    df["in_stock"] = df["in_stock"].astype(bool)
    df["rating"] = df["rating"].astype(int)

    return df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]


def build_schema(conn: sqlite3.Connection):
    conn.executescript(
        """
        DROP TABLE IF EXISTS books;
        DROP TABLE IF EXISTS categories;

        CREATE TABLE categories (
            category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE books (
            book_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT NOT NULL,
            price_gbp   REAL NOT NULL,
            price_inr   REAL NOT NULL,
            rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
            in_stock    INTEGER NOT NULL,  -- 0/1
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories(category_id)
        );
        """
    )
    conn.commit()


def load_into_db(df: pd.DataFrame, conn: sqlite3.Connection):
    categories = sorted(df["category"].unique())
    cat_df = pd.DataFrame({"category_name": categories})
    cat_df.to_sql("categories", conn, if_exists="append", index=False)

    cat_map = pd.read_sql("SELECT category_id, category_name FROM categories", conn)
    merged = df.merge(cat_map, left_on="category", right_on="category_name", how="left")

    books_df = merged[["title", "price_gbp", "price_inr", "rating", "in_stock", "category_id"]].copy()
    books_df["in_stock"] = books_df["in_stock"].astype(int)
    books_df.to_sql("books", conn, if_exists="append", index=False)


def main():
    raw = pd.read_csv(RAW_CSV)
    print(f"Loaded {len(raw)} raw rows from {RAW_CSV}")

    cleaned = clean_dataframe(raw)
    print(f"{len(cleaned)} rows remain after cleaning, across {cleaned['category'].nunique()} categories")
    assert len(cleaned) >= 60, "Expected at least 60 cleaned rows"
    assert cleaned["category"].nunique() >= 3, "Expected at least 3 categories"

    conn = sqlite3.connect(DB_PATH)
    build_schema(conn)
    load_into_db(cleaned, conn)
    conn.commit()

    n_books = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    n_cats = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
    print(f"Loaded {n_books} books across {n_cats} categories into {DB_PATH}")
    conn.close()


if __name__ == "__main__":
    main()
