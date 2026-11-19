"""Pydantic data contracts and LangGraph shared state definition."""

from typing import Optional, TypedDict
from pydantic import BaseModel, Field


class SearchPlan(BaseModel):
    """Structured output from the Researcher node."""

    queries: list[str] = Field(
        ...,
        min_length=1,
        max_length=3,
        description="2 to 3 crisp, targeted web search queries covering the technical topic.",
    )


class SourceSnippet(BaseModel):
    """A single deduplicated web evidence item."""

    index: int = Field(..., description="1-based citation number, e.g. 1 for [1].")
    title: str
    url: str
    snippet: str


class CritiqueVerdict(BaseModel):
    """Structured audit verdict from the Critic node."""

    passed: bool = Field(
        ...,
        description="True ONLY if every factual claim in the draft is supported by the sources and properly cited.",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Exact claims or numbers from the draft that lack evidence in the provided sources.",
    )
    feedback: str = Field(
        ...,
        description="Specific, actionable revision instructions for the Writer (or brief approval note if passed).",
    )


class ResearchState(TypedDict, total=False):
    """Shared state flowing through the LangGraph Researcher -> Writer -> Critic loop."""

    topic: str
    search_queries: list[str]
    sources: list[SourceSnippet]
    draft: str
    critique: Optional[CritiqueVerdict]
    revision_count: int
    audit_log: list[str]
