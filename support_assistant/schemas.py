"""
schemas.py
----------
Module 3 - Support Assistant (/support_assistant)

Pydantic models used by the FastAPI endpoint and enforced as the structured
output schema for every answer the graph produces (mock or real-LLM).
"""

from typing import List
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The customer's question.")


class AskResponse(BaseModel):
    answer: str = Field(..., description="The generated answer text.")
    sources: List[str] = Field(
        default_factory=list,
        description="Chunk/document IDs used to ground the answer. Empty for general_question answers.",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for the answer, 0-1."
    )


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
