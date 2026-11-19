"""Web search tool wrapper using ddgs with backend fallback and deduplication."""

import time
from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from src.schemas import SourceSnippet

# Ordered fallback chain of ddgs text backends (comma-separated = queried together)
DEFAULT_BACKEND_FALLBACKS = (
    "duckduckgo,brave,wikipedia",
    "yahoo,startpage,mojeek",
    "all",
)


def _search_with_fallback(
    client: DDGS,
    query: str,
    max_results: int,
    backends: tuple[str, ...] = DEFAULT_BACKEND_FALLBACKS,
) -> list[dict]:
    """Try backend groups in order until one returns non-empty results."""
    for backend in backends:
        try:
            results = client.text(
                query,
                region="us-en",
                max_results=max_results,
                backend=backend,
            )
            if results:
                return results
        except (RatelimitException, TimeoutException, DDGSException) as exc:
            print(f"[search warning] backend='{backend}' failed for '{query}': {exc}")
        except Exception as exc:
            print(f"[search warning] unexpected error on '{backend}': {type(exc).__name__}: {exc}")
    return []


def run_web_search(
    queries: list[str],
    max_results_per_query: int = 4,
    pause_seconds: float = 0.5,
) -> list[SourceSnippet]:
    """Execute search queries via DDGS and return deduplicated, numbered snippets."""
    sources: list[SourceSnippet] = []
    seen_urls: set[str] = set()
    client = DDGS(timeout=20)

    for i, query in enumerate(queries):
        raw_results = _search_with_fallback(client, query, max_results_per_query)

        for item in raw_results:
            url = (item.get("href") or item.get("url") or "").strip()
            body = (item.get("body") or item.get("snippet") or "").strip()
            title = (item.get("title") or "Untitled").strip()

            if not url or not body or url in seen_urls:
                continue

            seen_urls.add(url)
            sources.append(
                SourceSnippet(
                    index=len(sources) + 1,
                    title=title,
                    url=url,
                    snippet=body,
                )
            )

        if i < len(queries) - 1:
            time.sleep(pause_seconds)

    return sources


def format_sources_for_prompt(sources: list[SourceSnippet]) -> str:
    """Render numbered evidence blocks for Writer and Critic prompts."""
    if not sources:
        return "No sources retrieved."

    blocks = [
        f"[{src.index}] {src.title}\nURL: {src.url}\nEvidence: {src.snippet}"
        for src in sources
    ]
    return "\n\n".join(blocks)
