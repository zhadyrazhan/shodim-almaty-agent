"""LlamaIndex RAG agent that recommends things to do in Almaty.

Indexes the structured records from data/sxodim_data.json and answers questions
like "куда сходить на выходных" over them.

Two interchangeable LLM backends:
  openai  (default) - gpt-5-mini via the OpenAI API
  ollama            - a local Qwen model, no API cost, set LLM_BACKEND=ollama

The index is built in-memory from a small JSON file, so startup is a second or
two and there is no vector DB to run.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.query_engine import BaseQueryEngine

from .config import (
    AGENT_MODEL,
    LLM_BACKEND,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    REQUEST_TIMEOUT,
    SXODIM_DATA,
    TOP_K,
    require_openai_key,
)

SYSTEM_PROMPT = """Ты — дружелюбный местный гид по Алматы. Ты советуешь, куда сходить, \
опираясь ТОЛЬКО на афишу ниже.

Как отвечать:
- Предлагай 2-4 конкретных варианта из афиши, каждый с названием и коротким объяснением, почему он подходит.
- Учитывай контекст вопроса: для свидания — атмосферные места, для детей — безопасные и интересные им, для компании — где весело вместе.
- Если в афише нет ничего подходящего, честно скажи об этом и предложи ближайшую альтернативу из списка. Не выдумывай события.
- Отвечай тепло и по-человечески, как советуешь другу, но без навязчивости. 2-5 предложений плюс список вариантов.
"""


def record_to_document(record: dict) -> Document:
    """Flatten one record into searchable text plus filter metadata."""
    parts = [
        record.get("title", ""),
        record.get("category", ""),
        record.get("description", ""),
    ]
    if record.get("good_for"):
        parts.append("Подойдёт для: " + ", ".join(record["good_for"]))
    for field, label in (("date", "Когда"), ("price", "Цена"), ("address", "Адрес")):
        if record.get(field):
            parts.append(f"{label}: {record[field]}")

    return Document(
        text="\n".join(p for p in parts if p),
        metadata={
            "title": record.get("title", ""),
            "kind": record.get("kind", ""),
            "url": record.get("url", ""),
        },
    )


def load_records(path: Path = SXODIM_DATA) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run src/scraper.py then src/extract.py first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def configure_llm() -> None:
    """Point LlamaIndex at whichever backend is selected."""
    if LLM_BACKEND == "ollama":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.ollama import Ollama

        Settings.llm = Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            request_timeout=REQUEST_TIMEOUT,
        )
        # Local embeddings too, so the "ollama" path needs no API key at all.
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    else:
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI

        api_key = require_openai_key()
        Settings.llm = OpenAI(model=AGENT_MODEL, api_key=api_key)
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small", api_key=api_key
        )


@lru_cache(maxsize=1)
def build_query_engine(path: str = str(SXODIM_DATA)) -> BaseQueryEngine:
    """Build the index once and reuse it — cached for the process lifetime."""
    configure_llm()
    records = load_records(Path(path))
    documents = [record_to_document(r) for r in records]
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine(
        similarity_top_k=TOP_K,
        system_prompt=SYSTEM_PROMPT,
    )


def ask(question: str) -> str:
    """Answer one question against the Almaty listings."""
    question = question.strip()
    if not question:
        raise ValueError("empty question")
    return str(build_query_engine().query(question)).strip()


def main() -> None:
    import sys

    questions = sys.argv[1:] or [
        "Куда сходить на выходных?",
        "Посоветуй место для свидания",
        "Куда сводить ребенка?",
    ]
    for q in questions:
        print(f"\nQ: {q}\nA: {ask(q)}")


if __name__ == "__main__":
    main()
