"""CLI entry point for the local multi-agent Fact-Checked Technical Researcher."""

import argparse
from datetime import datetime
from pathlib import Path
import re
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.config import settings
from src.graph import build_research_graph

console = Console()


def _save_report(topic: str, final_state: dict) -> Path:
    """Save the verified brief, citations, and audit log to outputs/."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:50]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"{timestamp}_{slug}.md"

    critique = final_state.get("critique")
    verdict_label = "PASSED" if (critique and critique.passed) else "MAX REVISIONS REACHED"

    sources_md = "\n".join(
        f"- [{s.index}] [{s.title}]({s.url})"
        for s in final_state.get("sources", [])
    )
    audit_md = "\n".join(f"- {entry}" for entry in final_state.get("audit_log", []))

    content = (
        f"# {topic}\n\n"
        f"**Verification Status:** {verdict_label}  \n"
        f"**Revisions Performed:** {final_state.get('revision_count', 0)} / {settings.max_revisions}  \n"
        f"**Models:** Researcher=`{settings.researcher_model}` | "
        f"Writer=`{settings.writer_model}` | Critic=`{settings.critic_model}`\n\n"
        f"## Technical Brief\n\n{final_state.get('draft', '')}\n\n"
        f"## Sources\n\n{sources_md}\n\n"
        f"## Agent Audit Trail\n\n{audit_md}\n"
    )
    filepath.write_text(content, encoding="utf-8")
    return filepath


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local Multi-Agent Fact-Checked Technical Researcher (LangGraph + Ollama)"
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default="Compare vLLM PagedAttention vs SGLang RadixAttention for LLM serving",
        help="Technical topic to research and fact-check.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the verified Markdown brief and audit trail to outputs/.",
    )
    args = parser.parse_args()

    console.print(
        Panel.fit(
            f"[bold cyan]Topic:[/bold cyan] {args.topic}\n"
            f"[bold]Models:[/bold] Researcher=[green]{settings.researcher_model}[/green] | "
            f"Writer=[yellow]{settings.writer_model}[/yellow] | "
            f"Critic=[magenta]{settings.critic_model}[/magenta]\n"
            f"[bold]Max Revisions:[/bold] {settings.max_revisions}",
            title="Fact-Checked Technical Researcher",
        )
    )

    graph = build_research_graph()
    # Hard safety ceiling: researcher (1) + writer/critic pairs (2 * (max_revisions + 1)) + buffer
    recursion_limit = (settings.max_revisions + 1) * 2 + 4

    final_state: dict = {"topic": args.topic}
    for event in graph.stream(
        {"topic": args.topic},
        config={"recursion_limit": recursion_limit},
        stream_mode="updates",
    ):
        for node_name, update in event.items():
            final_state.update(update)
            latest_log = update.get("audit_log", [])[-1] if update.get("audit_log") else node_name
            console.print(f"[dim]• {latest_log}[/dim]")

    critique = final_state.get("critique")
    passed = bool(critique and critique.passed)
    border_color = "green" if passed else "yellow"
    status_title = (
        f"Verified Brief (PASSED after {final_state.get('revision_count', 0)} revision(s))"
        if passed
        else f"Brief Finished (Stopped at max_revisions={settings.max_revisions})"
    )

    console.print()
    console.print(
        Panel(
            Markdown(final_state.get("draft", "")),
            title=status_title,
            border_style=border_color,
        )
    )

    sources = final_state.get("sources", [])
    if sources:
        refs = "\n".join(f"[{s.index}] {s.title} — {s.url}" for s in sources)
        console.print(Panel(refs, title="Retrieved Sources", border_style="cyan"))

    if args.save:
        saved_path = _save_report(args.topic, final_state)
        console.print(f"\n[bold green]Saved report to:[/bold green] {saved_path}")


if __name__ == "__main__":
    main()
