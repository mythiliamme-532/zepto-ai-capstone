# Zepto Data & AI Platform — Capstone Project

Certificate Program in Artificial Intelligence and Machine Learning

One connected repository, three internally-linked modules:

| Module | Path | Marks | What it does |
|---|---|---|---|
| 1. Data Pipeline | [`/data_pipeline`](data_pipeline/) | 25 | Scrape → clean → convert currency → normalized SQLite → SQL + pandas queries |
| 2. Analytics Pipeline | [`/analytics`](analytics/) | 50 | Titanic profiling/EDA → cleaning → 3 classifiers → tuning → regression side-task |
| 3. Support Assistant | [`/support_assistant`](support_assistant/) | 25 | RAG corpus → embeddings → LangGraph intent router → structured output → FastAPI + Docker |

Each module has its own `README.md` with full design-decision write-ups;
this root README covers setup, run order, and a summary of each.

---

## ⚠️ A note on how this repository was produced

This code was authored in a sandboxed environment **with no internet access**
and without several of the third-party packages the project depends on
pre-installed (`sentence-transformers`, `chromadb`, `langgraph`, `fastapi`,
`imbalanced-learn` were all absent, and `books.toscrape.com` / Seaborn's
online dataset repository / any LLM API were all unreachable). Every script
was still exercised as far as that environment allowed:

- **`/data_pipeline`**: `clean_and_load.py` and `queries.py` were run
  end-to-end (cleaning, SQLite schema, all 6 SQL queries, the
  `pd.read_sql`/`pd.merge` equivalence check) against a synthetic stand-in
  CSV shaped exactly like the real scrape output, and passed. `scrape.py`
  itself needs live internet access to run.
- **`/analytics`**: both `01_eda.py` and `02_modeling.py` were run
  end-to-end (missing-value threshold handling, IQR outliers, correlation
  ranking, the leak-free `ColumnTransformer`/`Pipeline`, all 3 classifiers,
  `GridSearchCV`+OOB, the regression side-task, and the joblib save/reload
  round-trip) against a synthetic DataFrame with the exact Titanic schema,
  and passed. The real `sns.load_dataset("titanic")` call needs internet
  access on first run.
- **`/support_assistant`**: the `classify_intent` keyword heuristic and the
  canned mock-mode answer templates were verified in isolated pure-Python
  tests (no missing-dependency issue there). The full FastAPI/LangGraph/
  ChromaDB/sentence-transformers server could not be executed in this
  environment, since those packages could not be installed without network
  access.

**In short: every module's code is complete and its logic has been checked
as thoroughly as this offline environment allowed, but you should run each
module yourself (with internet access and `pip install -r requirements.txt`)
before submitting**, to regenerate the real `books.db`, real `titanic.csv` +
charts, and to actually start and query the FastAPI service — and to record
their real output in place of anything flagged above. Each module's own
README repeats this note in context.

---

## Setup

Each module has its own `requirements.txt` (three separate files, not
consolidated), since the modules have non-overlapping dependencies:

```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended

pip install -r data_pipeline/requirements.txt
pip install -r analytics/requirements.txt
pip install -r support_assistant/requirements.txt
```

## Running each module end to end

### 1. Data Pipeline
```bash
cd data_pipeline
python scrape.py            # -> raw_books.csv
python clean_and_load.py    # -> books.db (normalized SQLite)
python queries.py           # runs the 6 SQL queries + pandas equivalents
```

### 2. Analytics Pipeline
```bash
cd analytics
python 01_eda.py            # loads titanic (network, once), profiles, cleans, -> titanic.csv + charts/
python 02_modeling.py       # reads titanic.csv, trains/evaluates/tunes, -> best_pipeline.joblib
```

### 3. Support Assistant
```bash
cd support_assistant
python ingest.py                              # builds the ChromaDB index
uvicorn main:app --host 0.0.0.0 --port 7860    # MOCK_LLM defaults to mock mode
# or:
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

---

## Design decisions — summary (see each module's own README for full detail)

**Data Pipeline.** Scrapes 5 catalogue pages (~100 books, ~50 categories)
from books.toscrape.com. Rows that fail to parse are **dropped**, not
imputed — price/rating are per-book identity attributes, not measurements
with a sensible "typical value" to fabricate. Currency conversion uses the
fixed, project-defined constant `1 GBP = 105.50 INR`. Schema is two tables
(`categories`, `books`) joined on `category_id`. Six SQL queries cover every
required clause plus a JOIN; the JOIN result is independently reproduced
with `pd.merge` and shown to match `pd.read_sql`'s output exactly.

**Analytics Pipeline.** One load (`sns.load_dataset`) inside `01_eda.py`,
saved immediately to a committed `titanic.csv` that `02_modeling.py` reads
— never a second raw load. Missing values follow an explicit percentage
threshold rule (`<5%` drop rows, `5–30%` impute, very-high `deck` gets its
own "Missing" category rather than being dropped or falsely imputed). All
preprocessing (`ColumnTransformer` inside a `Pipeline`) is fit on the
training split only, structurally preventing test-set leakage. Three
classifiers are trained and compared on accuracy/precision/recall/F1/AUC;
imbalance handling is compared across baseline / `class_weight='balanced'` /
SMOTE (training-fold-only); a Random Forest is tuned via `GridSearchCV` with
`oob_score=True`; a separate linear regression predicts `fare` and reports
MAE/RMSE/R²/Adjusted R² plus a heteroscedasticity read on the residual plot.
The best full pipeline (preprocessing + estimator together) is saved with
`joblib` and reload-verified against raw input.

**Support Assistant.** 8 policy documents (verbatim, one per file) are
embedded locally with `all-MiniLM-L6-v2` and indexed in ChromaDB — no API
key, no network call, no cost. A LangGraph `StateGraph` with 3 nodes
(`classify_intent`, `retrieve_and_answer`, `direct_answer`) routes each
query via a conditional edge; retrieval always runs for real, while the
*generation* step in every node branches on the `MOCK_LLM` toggle — mock
mode (the required, graded, fully offline baseline) uses a keyword
heuristic for classification and canned templated answers, with
`answer`/`sources`/`confidence` populated deterministically and validated
against a Pydantic schema. `MOCK_LLM=0` is an optional, ungraded extension
that swaps in a real free-tier LLM (e.g. Groq) with a retry-with-correction
loop if schema validation fails. A `FastAPI` app exposes `POST /ask`; a
`Dockerfile` builds and serves it locally.

---

## Git workflow

This repository's history includes a feature branch that was created,
committed to at least twice, and merged back into `main` — visible via
`git log --graph --all`. This is checked once against the whole repository's
history (see `data_pipeline`'s grading rubric), not separately per module.
