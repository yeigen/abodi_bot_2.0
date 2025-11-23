"""Shared utilities to build the RAG chain used across scripts."""
from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "constitucion"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_TOP_K = 25
MODEL_NAME = "llama-3.1-8b-instant"
TEMPERATURE = 0


PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Eres un asistente experto en la Constitución. Usa únicamente el contexto proporcionado.\n"
        "Pregunta: {question}\n"
        "Contexto:\n{context}\n"
        "Instrucciones:\n"
        "- Responde de forma breve y en español.\n"
        "- Cita el artículo completo o una parte significativa del mismo.\n"
        "- Indica siempre la URL del artículo si está disponible en el texto.\n"
    ),
)


def build_chain(top_k: int = DEFAULT_TOP_K) -> RetrievalQA:
    """Load persisted vector store and return a RetrievalQA chain."""
    load_dotenv()
    api_key = os.environ["GROQ_API_KEY"]

    llm = ChatGroq(
        model=MODEL_NAME,
        api_key=api_key,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    retriever = vector_store.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance
    search_kwargs={
        "k": 20,
        "fetch_k": 200,  # Recupera 50, devuelve 20 diversos
        "lambda_mult": 0.5,  # Balance diversidad/relevancia
    },
)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )


def format_sources(sources: list[Document]) -> str:
    seen: list[str] = []
    for doc in sources:
        src = doc.metadata.get("source", "desconocido")
        if src not in seen:
            seen.append(src)
    return ", ".join(seen) if seen else "Sin fuentes registradas"
