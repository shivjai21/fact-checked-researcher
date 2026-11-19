"""Agent nodes for the Researcher -> Writer -> Critic LangGraph pipeline."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.config import settings
from src.schemas import CritiqueVerdict, ResearchState, SearchPlan
from src.tools import format_sources_for_prompt, run_web_search

REASONING_PREFIXES = ("qwen3", "deepseek-r1", "gemma4")


def _build_llm(
    model_name: str,
    temperature: float,
    disable_reasoning: bool = False,
) -> ChatOllama:
    """Create a configured ChatOllama instance for a specific agent role."""
    kwargs: dict = {
        "model": model_name,
        "base_url": settings.ollama_base_url,
        "temperature": temperature,
        "num_ctx": settings.ollama_num_ctx,
    }
    if disable_reasoning and model_name.lower().startswith(REASONING_PREFIXES):
        kwargs["reasoning"] = False
    return ChatOllama(**kwargs)


def researcher_node(state: ResearchState) -> dict:
    """Plan targeted search queries and gather deduplicated web evidence."""
    topic = state["topic"]
    audit_log = list(state.get("audit_log", []))

    llm = _build_llm(
        settings.researcher_model,
        temperature=0.1,
        disable_reasoning=True,
    )
    planner = llm.with_structured_output(SearchPlan, method="json_schema")

    try:
        plan: SearchPlan = planner.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a technical research planner. Generate 2 to 3 concise, "
                        "high-signal web search queries to gather factual engineering "
                        "evidence for the user's topic. Avoid redundant queries."
                    )
                ),
                HumanMessage(content=f"Technical topic: {topic}"),
            ]
        )
        queries = [q.strip() for q in plan.queries if q.strip()] or [topic]
    except Exception as exc:
        audit_log.append(f"[Researcher] Query planner fallback triggered: {exc}")
        queries = [topic]

    sources = run_web_search(
        queries=queries,
        max_results_per_query=settings.search_results_per_query,
    )

    audit_log.append(
        f"[Researcher | {settings.researcher_model}] Ran {len(queries)} queries "
        f"({queries}) -> retrieved {len(sources)} unique sources."
    )

    return {
        "search_queries": queries,
        "sources": sources,
        "revision_count": 0,
        "audit_log": audit_log,
    }


def writer_node(state: ResearchState) -> dict:
    """Draft or revise a concise, strictly cited technical brief."""
    topic = state["topic"]
    sources = state.get("sources", [])
    critique = state.get("critique")
    revision_count = state.get("revision_count", 0)
    audit_log = list(state.get("audit_log", []))

    if not sources:
        audit_log.append(
            f"[Writer | {settings.writer_model}] Skipped drafting: no web sources available."
        )
        return {
            "draft": (
                f"Unable to generate a fact-checked brief for '{topic}' because "
                "web search returned zero verifiable sources."
            ),
            "audit_log": audit_log,
        }

    is_revision = critique is not None
    if is_revision:
        revision_count += 1

    formatted_sources = format_sources_for_prompt(sources)
    llm = _build_llm(settings.writer_model, temperature=0.3)

    system_prompt = (
        "You are a Senior AI Engineer writing a crisp, factual technical brief "
        "(250-350 words). Follow these rules strictly:\n"
        "1. ONLY state facts, metrics, and comparisons explicitly supported by the "
        "numbered SOURCES below.\n"
        "2. Attach inline citations like [1] or [2] to every technical claim.\n"
        "3. Never invent benchmarks, release dates, or external knowledge not in SOURCES.\n"
        "4. Use clear Markdown bullets or short paragraphs."
    )

    if is_revision:
        unsupported = "\n".join(f"- {c}" for c in critique.unsupported_claims) or "- None listed"
        user_prompt = (
            f"TOPIC: {topic}\n\n"
            f"SOURCES:\n{formatted_sources}\n\n"
            f"PREVIOUS DRAFT:\n{state.get('draft', '')}\n\n"
            f"CRITIC FEEDBACK (FIX ALL ISSUES):\n{critique.feedback}\n"
            f"UNSUPPORTED CLAIMS TO REMOVE OR FIX:\n{unsupported}\n\n"
            "Rewrite the technical brief so every claim is 100% grounded in SOURCES with citations."
        )
    else:
        user_prompt = (
            f"TOPIC: {topic}\n\n"
            f"SOURCES:\n{formatted_sources}\n\n"
            "Write the cited technical brief now."
        )

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    draft = str(response.content).strip()

    stage = f"Revision #{revision_count}" if is_revision else "Initial Draft"
    audit_log.append(
        f"[Writer | {settings.writer_model}] Generated {stage} ({len(draft.split())} words)."
    )

    return {
        "draft": draft,
        "revision_count": revision_count,
        "audit_log": audit_log,
    }


def critic_node(state: ResearchState) -> dict:
    """Audit the Writer's draft against raw search snippets and emit a Pydantic verdict."""
    sources = state.get("sources", [])
    draft = state.get("draft", "")
    audit_log = list(state.get("audit_log", []))

    if not sources:
        verdict = CritiqueVerdict(
            passed=False,
            unsupported_claims=["No retrieved sources available to verify claims."],
            feedback="Search returned no sources; cannot verify brief.",
        )
        audit_log.append(f"[Critic | {settings.critic_model}] Auto-failed (0 sources).")
        return {"critique": verdict, "audit_log": audit_log}

    formatted_sources = format_sources_for_prompt(sources)
    llm = _build_llm(
        settings.critic_model,
        temperature=0.0,
        disable_reasoning=True,
    )
    auditor = llm.with_structured_output(CritiqueVerdict, method="json_schema")

    system_prompt = (
        "You are a strict technical fact-checker auditing an AI-generated draft.\n"
        "Compare every factual claim and number in the DRAFT against the numbered SOURCES.\n"
        "- Set passed=True ONLY if all factual claims are grounded in SOURCES and use valid [n] citations.\n"
        "- Set passed=False if ANY claim is hallucinated, exaggerated, contradicts SOURCES, "
        "or cites a source number that does not support that claim.\n"
        "- List exact unsupported claims and give concise, actionable revision feedback."
    )

    user_prompt = (
        f"SOURCES:\n{formatted_sources}\n\n"
        f"DRAFT TO AUDIT:\n{draft}"
    )

    try:
        verdict: CritiqueVerdict = auditor.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
    except Exception as exc:
        verdict = CritiqueVerdict(
            passed=False,
            unsupported_claims=[],
            feedback=(
                f"Structured audit parsing error ({exc}). Ensure every sentence "
                "strictly matches SOURCES with accurate [n] citations."
            ),
        )

    status = "PASSED" if verdict.passed else f"REJECTED ({len(verdict.unsupported_claims)} issues)"
    audit_log.append(f"[Critic | {settings.critic_model}] Verdict: {status} — {verdict.feedback}")

    return {
        "critique": verdict,
        "audit_log": audit_log,
    }
