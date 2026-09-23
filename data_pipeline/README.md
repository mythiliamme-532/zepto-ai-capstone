# Module 1 — Data Pipeline (`/data_pipeline`)

Scrapes books.toscrape.com, cleans and types the fields, converts price to INR
at a fixed project rate, and loads everything into a normalized two-table
SQLite database that is then queried with both SQL and pandas.

## Setup

```bash
pip install -r requirements.txt
```

## Run, in order

```bash
python scrape.py           # -> raw_books.csv  (scrapes 5 catalogue pages, ~100 books)
python clean_and_load.py   # -> books.db       (cleans, converts currency, loads schema)
python queries.py          # runs the required SQL queries + pandas equivalents, prints output
```

> **Note on this submission's execution environment:** `scrape.py` requires
> live internet access to books.toscrape.com, which the environment used to
> author this code did not have. The script was validated end-to-end against
> a synthetic stand-in dataset shaped exactly like the real scrape output
> (same columns/formats) to confirm the cleaning, currency-conversion, schema,
> and query logic all run correctly and the `pd.read_sql`/`pd.merge`
> equivalence check passes. Run the three scripts above in order, with
> internet access, to regenerate `raw_books.csv` and `books.db` from the real
> site before submitting.

## Design decisions

**Scraping scope.** We scrape the first 5 paginated listing pages of the
"All products" catalogue (`catalogue/page-1.html` … `page-5.html`), which
yields ~100 books across all ~50 of the site's categories — comfortably over
the required 60 rows / 3 categories. Category is read from each book's own
product page breadcrumb (`Home > Books > <Category> > <Title>`) rather than
guessed from the listing page, since the "All products" pages mix categories.

**Field cleaning.**
- `price_gbp`: the `£` symbol and any thousands separators are stripped with
  a regex and the remainder parsed as `float`.
- `rating`: the star rating is exposed on the site as a CSS class
  (`star-rating Three`, etc.) rather than a number; we map the English word
  to an int 1–5 via a lookup dict.
- `in_stock`: parsed from the availability text — `True` when the text
  contains "in stock", `False` for "Out of stock".

**Handling rows that fail to parse: drop, not impute.** If any field fails
to parse for a row, that row is **dropped**, not median-imputed. Rationale:
price and rating are per-book identifying attributes, not measurements with
a natural "typical value" — fabricating a median price or rating for a
specific named book would misrepresent that book in a way a dashboard
consumer (a Zepto analyst) could not detect. Median imputation is
defensible for a genuinely missing *measurement* field, not for a parse
failure on an identity attribute. In practice, books.toscrape.com's markup is
consistent enough that we expect 0 rows dropped in a normal run; the drop
path exists (and was verified) so the pipeline degrades gracefully instead
of crashing on any future markup surprise. The count of dropped rows, if
any, is printed by `clean_and_load.py`.

**Currency conversion.** `price_inr = price_gbp * 105.50`. This is the
project's fixed, artificial baseline rate (1 GBP = 105.50 INR) — a constant,
not a live or dated market rate — so no API call or lookup is required. (The
spec's optional live-rate stretch was not attempted; `price_inr` is fully
correct off the required fixed-rate baseline alone.)

**Schema.**
```sql
categories(category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE)
books(book_id INTEGER PRIMARY KEY, title TEXT, price_gbp REAL, price_inr REAL,
      rating INTEGER, in_stock INTEGER, category_id INTEGER REFERENCES categories(category_id))
```
Two tables, normalized on category to avoid repeating the category string on
every book row, joined via `category_id`.

**Queries (`queries.py`).** Six SQL queries are run, collectively covering
every required clause:
1. `SELECT`/`WHERE`/`ORDER BY`/`LIMIT` — top 10 priciest in-stock books
2. `DISTINCT` — distinct category names
3. `WHERE ... BETWEEN` — books priced INR 1000–2000
4. `WHERE ... IN` — books rated 4 or 5 stars
5. `JOIN` — highly rated books with their category name
6. (bonus) `GROUP BY` aggregate — per-category count/avg price/avg rating

Queries 1 and 5 are re-read into pandas via `pd.read_sql(...)`. Query 5 (the
JOIN) is additionally reproduced purely with `pd.merge(...)` on the two
in-memory DataFrames (no SQL), and the script asserts the two results are
row-for-row identical — demonstrating SQL-side and pandas-side joins agree.
