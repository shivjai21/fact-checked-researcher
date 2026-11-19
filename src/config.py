#Centralizing .env loading into an immutable, strongly typed Settings object 
# instead of scattering os.getenv() calls across the codebase.

from dataclasses import dataclass

import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    research_model: str = os.getenv("RESEARCHER_MODEL", "qwen3.5:2b")
    writer_model: str = os.getenv("WRITER_MODEL", "llama3.2:3b")
    critic_model: str = os.getenv("CRITIC_MODEL", "qwen3.5:4b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
    max_revisions: int = int(os.getenv("MAX_REVISIONS", "2"))
    search_results_per_query: int = int(os.getenv("SEARCH_RESULTS_PER_QUERY", "4"))


settings = Settings()
