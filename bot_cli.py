"""Simple CLI to query the persisted RAG chain and log basic RAG metrics."""
import csv
import os
from datetime import datetime
from typing import List

from langchain_core.documents import Document

from rag_chain import build_chain, format_sources

METRICS_FILE = "rag_metrics.csv"

def ensure_metrics_file() -> None:
    if os.path.exists(METRICS_FILE):
        return
    with open(METRICS_FILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "timestamp",
                "question",
                "retrieved_docs",
                "unique_sources",
                "response_characters",
                "sources",
            ],
        )
        writer.writeheader()


def log_metrics(
    question: str,
    sources: List[Document],
    answer: str,
) -> None:
    ensure_metrics_file()
    retrieved = len(sources)
    unique_sources = len({doc.metadata.get("source", "desconocido") for doc in sources})
    response_characters = len(answer)
    with open(METRICS_FILE, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "timestamp",
                "question",
                "retrieved_docs",
                "unique_sources",
                "response_characters",
                "sources",
            ],
        )
        writer.writerow(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "question": question,
                "retrieved_docs": retrieved,
                "unique_sources": unique_sources,
                "response_characters": response_characters,
                "sources": format_sources(sources),
            }
        )


def collect_metrics(question: str, answer: str, sources: List[Document]) -> None:
    log_metrics(question=question, sources=sources, answer=answer)


def main() -> None:
    qa_chain = build_chain()
    print("RAG listo. Escribe tu pregunta (o 'exit' para salir).")

    while True:
        try:
            question = input("› ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo…")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "salir"}:
            print("Saliendo…")
            break

        result = qa_chain.invoke({"query": question})
        answer = result.get("result", "")
        sources = result.get("source_documents", [])
        print(f"\nRespuesta:\n{answer}\n")
        collect_metrics(question, answer, sources)


if __name__ == "__main__":
    main()
