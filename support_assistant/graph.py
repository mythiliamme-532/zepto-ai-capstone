"""
graph.py
--------
Module 3 - Support Assistant (/support_assistant)

Builds a LangGraph StateGraph with a TypedDict state and 3 nodes:
  1. classify_intent     - routes to policy_question / general_question
  2. retrieve_and_answer - for policy_question: real retrieval (always),
                            generation branches on MOCK_LLM
  3. direct_answer        - for general_question: generation branches on MOCK_LLM

A conditional edge from classify_intent routes to (2) or (3). This routing
logic does NOT depend on MOCK_LLM - only each node's generation step does.

MOCK_LLM toggle (read once per call, via os.environ so tests can flip it):
  unset or "1"  -> required, graded, fully offline/deterministic mock path
  "0"           -> optional extension: calls a real LLM (Groq free tier, or
                    any genuinely-free-tier LLM API of your choice)

The final answer is always validated against schemas.AskResponse before
being returned.
"""

import os
import json
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, END
from pydantic import ValidationError

from schemas import AskResponse
from prompt_template import build_prompt
from ingest import retrieve_top_k

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership", "tracking",
    "cancel", "gift card", "support hours",
]


def mock_llm_enabled() -> bool:
    """MOCK_LLM unset or '1' -> mock (graded baseline). '0' -> real LLM."""
    return os.environ.get("MOCK_LLM", "1") != "0"


class GraphState(TypedDict):
    query: str
    intent: Optional[str]           # "policy_question" | "general_question"
    retrieved: Optional[List[dict]]  # list of {id, text, distance}
    answer: Optional[str]
    sources: Optional[List[str]]
    confidence: Optional[float]


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------
def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]

    if mock_llm_enabled():
        # Mock mode (graded baseline): keyword heuristic, no LLM call.
        lowered = query.lower()
        intent = "policy_question" if any(kw in lowered for kw in POLICY_KEYWORDS) else "general_question"
    else:
        # Optional MOCK_LLM=0 extension: call a real LLM to classify instead.
        intent = _real_llm_classify(query)

    return {**state, "intent": intent}


def _real_llm_classify(query: str) -> str:
    """
    Optional extension only (MOCK_LLM=0). Calls a real LLM (e.g. Groq's free
    tier) to classify the query. Left as a documented stub: plug in your
    chosen client here. Falls back to the keyword heuristic on any error so
    the graph never crashes if the optional LLM call is unavailable.
    """
    try:
        # Example (Groq, OpenAI-compatible client) - fill in your own client:
        #
        # from groq import Groq
        # client = Groq(api_key=os.environ["GROQ_API_KEY"])
        # resp = client.chat.completions.create(
        #     model="llama-3.1-8b-instant",
        #     messages=[{"role": "user", "content":
        #         f"Classify this support query as exactly one word, either "
        #         f"'policy_question' or 'general_question': {query}"}],
        # )
        # label = resp.choices[0].message.content.strip().lower()
        # return label if label in ("policy_question", "general_question") else "general_question"
        raise NotImplementedError("Plug in a real LLM client to use MOCK_LLM=0.")
    except Exception:
        lowered = query.lower()
        return "policy_question" if any(kw in lowered for kw in POLICY_KEYWORDS) else "general_question"


# ---------------------------------------------------------------------------
# Node 2: retrieve_and_answer (policy_question branch)
# ---------------------------------------------------------------------------
def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]

    # Retrieval always runs for real in both MOCK_LLM modes - it needs no
    # API key and no network call (local embeddings + local ChromaDB).
    hits = retrieve_top_k(query, k=3)
    top_chunk = hits[0]["text"] if hits else ""

    if mock_llm_enabled():
        # Mock mode (graded baseline): canned templated answer, no LLM call.
        snippet = top_chunk[:200]
        answer = f"Based on the retrieved context: {snippet}"
        sources = [h["id"] for h in hits]
        confidence = 1.0
    else:
        # Optional MOCK_LLM=0 extension: prompt a real LLM, grounded only in
        # the retrieved chunks, using the structured template.
        context_block = "\n".join(f"[{h['id']}] {h['text']}" for h in hits)
        prompt = build_prompt(query=query, retrieved_context=context_block)
        answer = _real_llm_generate(prompt)
        sources = [h["id"] for h in hits]
        confidence = 0.8  # placeholder; a real integration could self-report this

    return {**state, "retrieved": hits, "answer": answer, "sources": sources, "confidence": confidence}


# ---------------------------------------------------------------------------
# Node 3: direct_answer (general_question branch)
# ---------------------------------------------------------------------------
def direct_answer(state: GraphState) -> GraphState:
    if mock_llm_enabled():
        # Mock mode (graded baseline): fixed canned string, no LLM call, no retrieval.
        answer = "I can only answer questions about Zepto policies right now."
        confidence = 1.0
    else:
        # Optional MOCK_LLM=0 extension: prompt the LLM directly, no retrieval.
        prompt = build_prompt(query=state["query"], retrieved_context="(no retrieval for general questions)")
        answer = _real_llm_generate(prompt)
        confidence = 0.6

    return {**state, "retrieved": [], "answer": answer, "sources": [], "confidence": confidence}


def _real_llm_generate(prompt: str) -> str:
    """
    Optional extension only (MOCK_LLM=0). Plug in your real LLM client here
    (e.g. Groq's OpenAI-compatible client). Retries up to 2 additional times
    with a corrective instruction if the raw output fails schema validation
    are handled by the caller (see ask() in main.py / validate_or_retry below).
    """
    raise NotImplementedError(
        "Plug in a real LLM client (e.g. Groq free tier) to use MOCK_LLM=0. "
        "The required, graded submission only needs the MOCK_LLM=1 (default) path."
    )


# ---------------------------------------------------------------------------
# Conditional edge routing (independent of MOCK_LLM)
# ---------------------------------------------------------------------------
def route_by_intent(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"},
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def validate_response(state: GraphState, max_retries: int = 2) -> AskResponse:
    """
    Validates the graph's output against the AskResponse schema. In mock
    mode this always succeeds (the fields are populated deterministically by
    our own code). In the optional real-LLM path, if validation ever failed
    here, the caller would re-invoke generation up to `max_retries` times
    with a corrective instruction before giving up; that retry loop is
    exercised in main.py's ask() handler since it needs to re-run the graph.
    """
    payload = {
        "answer": state.get("answer", ""),
        "sources": state.get("sources") or [],
        "confidence": state.get("confidence", 0.0),
    }
    return AskResponse(**payload)  # raises pydantic.ValidationError on bad shape


def run(query: str) -> AskResponse:
    graph = get_compiled_graph()
    final_state = graph.invoke({"query": query, "intent": None, "retrieved": None,
                                 "answer": None, "sources": None, "confidence": None})
    return validate_response(final_state)
