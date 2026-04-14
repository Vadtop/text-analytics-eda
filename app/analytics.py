import pandas as pd
import numpy as np
from collections import Counter
import re
import logging

logger = logging.getLogger(__name__)

SAMPLE_TEXTS = [
    "RAG combines retrieval with generation for grounded AI answers",
    "Fine-tuning with LoRA reduces trainable parameters by 90 percent",
    "Docker containers provide reproducible ML deployment environments",
    "ChromaDB stores embeddings for fast semantic similarity search",
    "FastAPI serves ML models as REST APIs with automatic documentation",
    "Qdrant is a vector database optimized for high dimensional search",
    "HuggingFace Transformers library loads pre-trained models for inference",
    "Neo4j graph database stores entities and relationships for GraphRAG",
    "Unsloth accelerates LLM fine-tuning with custom Triton kernels",
    "PyTorch provides GPU accelerated tensor computation for deep learning",
    "PostgreSQL supports full text search and structured data queries",
    "LangChain provides RAG chains that combine retrieval with generation",
    "Python asyncio enables concurrent HTTP requests and batch processing",
    "Sentence transformers generate dense vector representations of text",
    "PEFT library manages parameter efficient fine-tuning with adapters",
    "Cross-encoder rerankers improve retrieval precision after vector search",
    "Guardrails check LLM outputs for PII refusals and length issues",
    "GraphRAG combines knowledge graphs with LLMs for multi-hop reasoning",
    "TF-IDF measures term importance relative to document frequency",
    "Cosine similarity compares vector directions regardless of magnitude",
]


def create_dataframe(texts: list[str] | None = None) -> pd.DataFrame:
    data = texts or SAMPLE_TEXTS
    rows = []
    for i, text in enumerate(data):
        words = text.split()
        rows.append(
            {
                "id": i,
                "text": text,
                "word_count": len(words),
                "char_count": len(text),
                "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
                "has_number": bool(re.search(r"\d", text)),
                "unique_words": len(set(w.lower() for w in words)),
                "lexical_diversity": len(set(w.lower() for w in words))
                / max(len(words), 1),
            }
        )
    return pd.DataFrame(rows)


def compute_statistics(df: pd.DataFrame) -> dict:
    stats = {}
    numeric_cols = [
        "word_count",
        "char_count",
        "avg_word_length",
        "unique_words",
        "lexical_diversity",
    ]

    for col in numeric_cols:
        if col in df.columns:
            stats[col] = {
                "mean": round(float(df[col].mean()), 2),
                "median": round(float(df[col].median()), 2),
                "std": round(float(df[col].std()), 2),
                "min": round(float(df[col].min()), 2),
                "max": round(float(df[col].max()), 2),
            }

    stats["total_texts"] = len(df)
    stats["total_words"] = int(df["word_count"].sum())
    stats["total_unique_words_corpus"] = len(_corpus_vocabulary(df))

    return stats


def _corpus_vocabulary(df: pd.DataFrame) -> set[str]:
    all_words = set()
    for text in df["text"]:
        all_words.update(w.lower() for w in text.split())
    return all_words


def word_frequency(df: pd.DataFrame, top_n: int = 20) -> list[dict]:
    counter = Counter()
    for text in df["text"]:
        words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
        stop_words = {
            "the",
            "and",
            "for",
            "with",
            "is",
            "in",
            "to",
            "of",
            "a",
            "an",
            "on",
            "by",
            "as",
            "or",
            "at",
            "it",
            "from",
            "that",
            "this",
        }
        counter.update(w for w in words if w not in stop_words)

    return [{"word": w, "count": c} for w, c in counter.most_common(top_n)]


def correlation_matrix(df: pd.DataFrame) -> dict:
    numeric_cols = [
        "word_count",
        "char_count",
        "avg_word_length",
        "unique_words",
        "lexical_diversity",
    ]
    existing = [c for c in numeric_cols if c in df.columns]
    if not existing:
        return {}
    corr = df[existing].corr()
    return {
        col: {row: round(float(corr.loc[row, col]), 3) for row in existing}
        for col in existing
    }


def filter_texts(
    df: pd.DataFrame,
    min_words: int = 0,
    max_words: int = 999,
    has_number: bool | None = None,
) -> pd.DataFrame:
    result = df[df["word_count"] >= min_words]
    result = result[result["word_count"] <= max_words]
    if has_number is not None:
        result = result[result["has_number"] == has_number]
    return result


def generate_histogram_data(
    df: pd.DataFrame, column: str = "word_count", bins: int = 10
) -> dict:
    if column not in df.columns:
        return {"error": f"Column {column} not found"}

    values = df[column].values
    counts, edges = np.histogram(values, bins=bins)
    return {
        "column": column,
        "bins": [
            {
                "range": f"{round(edges[i], 1)}-{round(edges[i + 1], 1)}",
                "count": int(counts[i]),
            }
            for i in range(len(counts))
        ],
        "mean": round(float(np.mean(values)), 2),
        "std": round(float(np.std(values)), 2),
    }
