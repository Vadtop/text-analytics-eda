"""Tests for text analytics and pandas/numpy operations."""
import pytest
import numpy as np


def test_word_count():
    """Word count splits on whitespace correctly."""
    text = "RAG combines retrieval with generation for grounded AI answers"
    assert len(text.split()) == 9


def test_char_count():
    """Char count includes spaces."""
    text = "hello world"
    assert len(text) == 11


def test_avg_word_length():
    """Average word length computed correctly."""
    words = ["RAG", "is", "great"]
    avg = np.mean([len(w) for w in words])
    assert round(avg, 2) == round((3 + 2 + 5) / 3, 2)


def test_lexical_diversity():
    """Lexical diversity is unique_words / total_words."""
    text = "the cat sat on the mat"
    words = text.split()
    diversity = len(set(words)) / len(words)
    assert round(diversity, 2) == round(5 / 6, 2)


def test_lexical_diversity_all_unique():
    """All unique words → diversity = 1.0."""
    text = "RAG vector embeddings retrieval"
    words = text.split()
    diversity = len(set(words)) / len(words)
    assert diversity == 1.0


def test_has_number_detection():
    """Numeric content detected correctly."""
    import re
    assert re.search(r"\d", "LoRA reduces params by 90 percent") is not None
    assert re.search(r"\d", "no numbers here") is None


def test_word_frequency_top_n():
    """Top N word frequency returns correct count."""
    from collections import Counter
    texts = ["RAG is fast", "RAG is good", "RAG works well"]
    all_words = " ".join(texts).lower().split()
    freq = Counter(all_words).most_common(3)
    assert freq[0] == ("rag", 3)
    assert len(freq) == 3


def test_stopword_filtering():
    """Common stopwords excluded from frequency."""
    stopwords = {"is", "the", "a", "an", "and", "or", "for", "with", "on"}
    words = ["RAG", "is", "a", "retrieval", "system"]
    filtered = [w for w in words if w.lower() not in stopwords]
    assert "is" not in filtered
    assert "RAG" in filtered
    assert "retrieval" in filtered


def test_correlation_matrix_shape():
    """Correlation matrix is square."""
    import pandas as pd
    df = pd.DataFrame({
        "word_count": [5, 10, 15],
        "char_count": [25, 55, 80],
        "unique_words": [5, 9, 12],
    })
    corr = df.corr()
    assert corr.shape == (3, 3)


def test_histogram_bins():
    """Histogram returns correct number of bins."""
    data = [5, 7, 9, 10, 12, 15, 8, 6]
    counts, edges = np.histogram(data, bins=4)
    assert len(counts) == 4
    assert len(edges) == 5


def test_filter_by_word_count():
    """Filter returns only texts within word range."""
    texts = [
        {"text": "short", "word_count": 1},
        {"text": "medium length text here", "word_count": 4},
        {"text": "this is a longer text with more words in it", "word_count": 10},
    ]
    filtered = [t for t in texts if 3 <= t["word_count"] <= 6]
    assert len(filtered) == 1
    assert filtered[0]["word_count"] == 4


def test_dataframe_head_returns_five():
    """Head returns first 5 rows."""
    import pandas as pd
    df = pd.DataFrame({"id": range(20), "text": [f"text {i}" for i in range(20)]})
    head = df.head(5)
    assert len(head) == 5
    assert head.iloc[0]["id"] == 0
