# Fact-Checked Technical Researcher (LangGraph + Multi-Model Ollama)

A local-first, 3-agent technical research pipeline built with **LangGraph** and **Ollama**. Given any engineering topic, the system plans targeted searches, drafts a strictly cited technical brief, and runs a cross-family **Fact-Check Critic** that rejects unsupported claims back to the Writer inside a bounded reflection loop.

## Architecture

```mermaid
flowchart TD
    START([User Topic]) --> R[Researcher Node<br/>qwen3.5:0.8b + DDGS Search]
    R --> W[Writer Node<br/>llama3.2:3b Cited Brief]
    W --> C[Critic Node<br/>qwen3.5:4b Pydantic Audit]
    C -->|passed == False & revisions left| W
    C -->|passed == True or max_revisions hit| END([Verified Brief + Audit Trail])
