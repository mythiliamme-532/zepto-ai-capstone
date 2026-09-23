"""
queries.py
----------
Module 1 - Data Pipeline (/data_pipeline)

Runs >= 5 SQL queries against books.db, collectively covering:
  SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, IN/BETWEEN, and a JOIN.

Then:
  - Reads two of these query results into pandas via pd.read_sql
  - Reproduces the JOIN query's result purely with pd.merge on in-memory
    DataFrames (no SQL) and shows the two approaches match.

Run:
    python queries.py
Requires books.db to already exist (run clean_and_load.py first).
"""

import sqlite3
import pandas as pd

DB_PATH = "books.db"


def run_and_print(conn, label, sql):
    print(f"\n--- {label} ---")
    print(sql.strip())
    cur = conn.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    print(f"columns: {cols}")
    for r in rows:
        print(r)
    print(f"({len(rows)} row(s))")
    return cols, rows


def main():
    conn = sqlite3.connect(DB_PATH)

    # 1. SELECT / WHERE / ORDER BY / LIMIT
    q1 = """
    SELECT title, price_inr, rating
    FROM books
    WHERE in_stock = 1
    ORDER BY price_inr DESC
    LIMIT 10;
    """
    run_and_print(conn, "Q1: Top 10 most expensive in-stock books (SELECT/WHERE/ORDER BY/LIMIT)", q1)

    # 2. DISTINCT
    q2 = """
    SELECT DISTINCT category_name
    FROM categories
    ORDER BY category_name;
    """
    run_and_print(conn, "Q2: Distinct category names (DISTINCT)", q2)

    # 3. WHERE ... BETWEEN
    q3 = """
    SELECT title, price_inr
    FROM books
    WHERE price_inr BETWEEN 1000 AND 2000
    ORDER BY price_inr ASC
    LIMIT 15;
    """
    run_and_print(conn, "Q3: Books priced between INR 1000 and 2000 (BETWEEN)", q3)

    # 4. WHERE ... IN
    q4 = """
    SELECT title, rating, category_id
    FROM books
    WHERE rating IN (4, 5)
    ORDER BY rating DESC
    LIMIT 15;
    """
    run_and_print(conn, "Q4: Books rated 4 or 5 stars (IN)", q4)

    # 5. JOIN - top-rated books per category (used again below for read_sql/merge comparison)
    join_sql = """
    SELECT c.category_name AS category, b.title, b.rating, b.price_inr
    FROM books b
    JOIN categories c ON b.category_id = c.category_id
    WHERE b.rating >= 4
    ORDER BY c.category_name, b.rating DESC, b.price_inr DESC
    LIMIT 20;
    """
    run_and_print(conn, "Q5: JOIN - highly rated books with their category name", join_sql)

    # 6. Aggregate example (bonus, not required but useful for the story)
    q6 = """
    SELECT c.category_name AS category,
           COUNT(*) AS n_books,
           ROUND(AVG(b.price_inr), 2) AS avg_price_inr,
           ROUND(AVG(b.rating), 2) AS avg_rating
    FROM books b
    JOIN categories c ON b.category_id = c.category_id
    GROUP BY c.category_name
    ORDER BY n_books DESC;
    """
    run_and_print(conn, "Q6 (bonus): Per-category book count / avg price / avg rating", q6)

    # ---- pd.read_sql for two of the above queries ----
    print("\n\n=== pd.read_sql demonstration ===")
    df_q1 = pd.read_sql(q1, conn)
    print("\npd.read_sql(Q1):")
    print(df_q1)

    df_join_sql = pd.read_sql(join_sql, conn)
    print("\npd.read_sql(Q5 / JOIN query):")
    print(df_join_sql)

    # ---- pd.merge reproduction of the JOIN query, no SQL ----
    print("\n\n=== pd.merge reproduction of Q5 (no SQL) ===")
    books_df = pd.read_sql("SELECT * FROM books", conn)
    categories_df = pd.read_sql("SELECT * FROM categories", conn)

    merged = books_df.merge(categories_df, on="category_id", how="inner")
    merged = merged[merged["rating"] >= 4]
    merged = merged.rename(columns={"category_name": "category"})[
        ["category", "title", "rating", "price_inr"]
    ]
    merged = merged.sort_values(
        by=["category", "rating", "price_inr"], ascending=[True, False, False]
    ).head(20).reset_index(drop=True)

    print(merged)

    # ---- Equivalence check ----
    sql_result = df_join_sql.reset_index(drop=True)
    merge_result = merged.reset_index(drop=True)
    are_equal = sql_result.equals(merge_result)
    print(f"\npd.read_sql JOIN result == pd.merge result (row/column-exact): {are_equal}")
    if not are_equal:
        # Order-independent fallback comparison, useful if LIMIT ties break differently
        sql_sorted = sql_result.sort_values(list(sql_result.columns)).reset_index(drop=True)
        merge_sorted = merge_result.sort_values(list(merge_result.columns)).reset_index(drop=True)
        print(f"Order-independent match: {sql_sorted.equals(merge_sorted)}")

    conn.close()


if __name__ == "__main__":
    main()
