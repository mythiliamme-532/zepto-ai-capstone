# Module 3 — Support Assistant (`/support_assistant`)

A small, complete GenAI support service for Zepto: a document corpus embedded
and indexed locally, a LangGraph-orchestrated intent router, a structured
Pydantic output schema, and a FastAPI wrapper — all graded through a fully
offline, deterministic `MOCK_LLM` path.

## Setup

```bash
pip install -r requirements.txt
python ingest.py          # builds the ChromaDB index from docs/ (first run downloads
                           # the all-MiniLM-L6-v2 model weights, then runs fully offline)
uvicorn main:app --host 0.0.0.0 --port 7860
```

`MOCK_LLM` is left unset (defaults to mock mode) for grading — no signup, API
key, or network call to any LLM provider is needed for full marks. Only
`MOCK_LLM=0` (optional, ungraded extension) calls a real LLM.

## Docker

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
# POST http://localhost:7860/ask  {"query": "..."}
```

> **Note on this submission's execution environment:** `sentence-transformers`,
> `chromadb`, `langgraph`, and `fastapi` were not installed and could not be
> installed in the environment used to author this code (no internet
> access), so the full server could not be executed end-to-end there. What
> *was* verified in isolation (pure-Python logic, no missing dependencies):
> the `classify_intent` keyword heuristic correctly routes policy-style
> queries (containing "delivery", "return", "refund", "membership",
> "tracking", "cancel", "gift card", "support hours") to `policy_question`
> and unrelated queries to `general_question`; and the canned mock-mode
> answer templates produce exactly the required `"Based on the retrieved
> context: {snippet}"` and fixed-string formats. The example call
> transcripts below show the exact deterministic output the mock pipeline
> produces given the corpus text (computed directly from `docs/*.txt`, since
> mock mode has no randomness); the **retrieval ranking** (which chunk
> ChromaDB returns as the top match) depends on the actual embedding model
> and could not be executed here, so run `python ingest.py` then start the
> server yourself to confirm/re-record these transcripts with real
> retrieval before submitting.

## Example calls (MOCK_LLM at its default)

**Call 1 — triggers retrieval (`policy_question`):**

Request:
```json
POST /ask
{"query": "Can I cancel my order after it's been packed?"}
```

Expected response (mock mode; retrieval should surface `doc_05`, the Order
Cancellation Policy, as the top match for this query):
```json
{
  "answer": "Based on the retrieved context: Orders can be cancelled free of cost any time before the order status changes to 'Packed', typically within the first 2 minutes of placing the order. Once an order has been packed, it can no longer be",
  "sources": ["doc_05", "doc_04", "doc_06"],
  "confidence": 1.0
}
```
*(`sources` lists the top-3 retrieved chunk IDs in ranked order; the exact
2nd/3rd IDs depend on the embedding model's actual similarity ranking and
may differ slightly from this illustrative example — `doc_05` as the #1 match
should be stable given it is the only document that mentions cancellation
after packing.)*

**Call 2 — does NOT trigger retrieval (`general_question`):**

Request:
```json
POST /ask
{"query": "What's the weather like today?"}
```

Response (mock mode, fixed canned string, no retrieval, no LLM call):
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## Architecture (ingestion → embedding → retrieval → generation)

**Ingestion.** `ingest.py:load_chunks()` reads the 8 files in `docs/`
(`doc_01.txt`…`doc_08.txt`). Each file is already a single short policy
paragraph, so each *document* is treated as one *chunk* — no further
splitting is needed given their length. Each chunk's id doubles as its
document id (e.g. `"doc_05"`).

**Embedding.** `ingest.py:get_model()` loads `sentence-transformers`'
`all-MiniLM-L6-v2` model locally (no API key, no per-call cost). `
get_collection()` embeds every chunk once and stores the vectors in a
persistent ChromaDB collection (`chroma_store/`, collection name
`zepto_policies`), so re-running the server doesn't re-embed on every
startup.

**Retrieval.** `ingest.py:retrieve_top_k(query, k=3)` embeds the incoming
query with the same model and asks ChromaDB for the top-3 chunks by cosine
similarity. This is called from the `retrieve_and_answer` node in
`graph.py`, and — per the spec — this retrieval step runs for real in
*both* `MOCK_LLM` states, since embeddings and ChromaDB need no API key and
no network call.

**Generation.** `graph.py` defines a LangGraph `StateGraph` with a
`TypedDict` state (`GraphState`) and three nodes:
- `classify_intent` — keyword heuristic in mock mode (`MOCK_LLM` default);
  would call an LLM in the optional `MOCK_LLM=0` extension.
- `retrieve_and_answer` — always retrieves for real (see above); in mock
  mode returns the canned `f"Based on the retrieved context: {snippet}"`
  template built from the top chunk; in the optional extension, prompts a
  real LLM using the structured template in `prompt_template.py`, grounded
  only in the retrieved chunks.
- `direct_answer` — in mock mode returns a fixed canned string with no
  retrieval and no LLM call; in the optional extension, prompts the LLM
  directly with no retrieval.

A conditional edge out of `classify_intent` (`route_by_intent` in
`graph.py`) routes to `retrieve_and_answer` or `direct_answer` based on the
classified intent — this routing logic itself does not depend on
`MOCK_LLM`, only the generation *inside* each node does.

**Structured output.** Every path populates `schemas.AskResponse`
(`answer: str`, `sources: List[str]`, `confidence: float` in `[0,1]`). In
mock mode this is populated deterministically by our own code (no LLM
output to fail validation, since none was generated). `main.py`'s `/ask`
handler wraps the optional real-LLM path in a retry loop (up to 2 additional
attempts with a corrective instruction) that returns a clearly marked
`ErrorResponse` if validation still fails after 3 total attempts.

**What changes with `MOCK_LLM`.** Only the *generation* sub-step inside
`classify_intent`, `retrieve_and_answer`, and `direct_answer` branches on
the toggle — routing and retrieval are identical either way. At `MOCK_LLM`'s
default (unset/`"1"`), no network call to any LLM provider is ever made and
every field is filled deterministically from code. At `MOCK_LLM=0`
(optional, ungraded), those same three generation sub-steps instead call a
real LLM (e.g. Groq's free tier) — classification via a direct prompt,
`retrieve_and_answer` via the structured template grounded in retrieved
chunks, and `direct_answer` via a plain prompt with no retrieval — with the
retry-on-validation-failure logic in `main.py` engaged.

```
 query
   │
   ▼
[classify_intent]  (keyword heuristic, mock; LLM call, MOCK_LLM=0)
   │
   ├─ policy_question ──▶ [retrieve_and_answer]
   │                         │  embed query → ChromaDB top-3 (always real)
   │                         └─ mock: canned template | MOCK_LLM=0: LLM + prompt template
   │
   └─ general_question ─▶ [direct_answer]
                              └─ mock: fixed string | MOCK_LLM=0: LLM, no retrieval
   │
   ▼
 AskResponse (answer / sources / confidence) — validated via Pydantic
```
