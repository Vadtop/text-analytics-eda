# Sample Analysis

## Running Analytics on Your Own Texts

### Upload custom texts and analyze
```bash
# Get statistics for loaded texts
curl http://localhost:8000/stats

# Top 15 most frequent words
curl "http://localhost:8000/word-frequency?top_n=15"

# Correlation between text metrics
curl http://localhost:8000/correlation

# Filter: medium-length texts only
curl -X POST http://localhost:8000/filter \
  -H "Content-Type: application/json" \
  -d '{"min_words": 5, "max_words": 12}'

# Histogram of word counts (5 bins)
curl -X POST http://localhost:8000/histogram \
  -H "Content-Type: application/json" \
  -d '{"column": "word_count", "bins": 5}'
```

## Sample Stats Response
```json
{
  "count": 20,
  "word_count": {"mean": 8.4, "median": 8.0, "std": 1.5, "min": 6, "max": 11},
  "char_count": {"mean": 54.3, "median": 53.5, "std": 8.7},
  "lexical_diversity": {"mean": 0.97, "median": 1.0},
  "has_number_pct": 35.0
}
```

## Metrics Explained

| Metric | Formula | Insight |
|--------|---------|---------|
| `word_count` | `len(text.split())` | Text length |
| `lexical_diversity` | `unique_words / word_count` | Vocabulary richness (1.0 = all unique) |
| `avg_word_length` | `mean(len(w) for w in words)` | Complexity indicator |
| `has_number` | `bool(re.search(r"\d", text))` | Numeric content flag |

## Extending with Custom Data

Modify `app/analytics.py` → `SAMPLE_TEXTS` list, or POST your texts to the `/filter` endpoint.
