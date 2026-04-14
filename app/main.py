import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from app.analytics import (
    create_dataframe,
    compute_statistics,
    word_frequency,
    correlation_matrix,
    filter_texts,
    generate_histogram_data,
)

logger = logging.getLogger(__name__)

_df = None


def get_df() -> "pd.DataFrame":
    global _df
    if _df is None:
        _df = create_dataframe()
    return _df


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Text Analytics EDA starting...")
    get_df()
    logger.info(f"Loaded {len(get_df())} sample texts")
    yield


app = FastAPI(
    title="Text Analytics EDA",
    description="Text analytics with pandas, numpy — statistics, word frequency, correlation, histogram",
    version="1.0",
    lifespan=lifespan,
)


class FilterRequest(BaseModel):
    min_words: int = 0
    max_words: int = 999
    has_number: bool | None = None


class HistogramRequest(BaseModel):
    column: str = "word_count"
    bins: int = 10


@app.get("/")
def root():
    return {
        "service": "Text Analytics EDA",
        "version": "1.0",
        "endpoints": {
            "GET /stats": "Compute text statistics (pandas + numpy)",
            "GET /word-frequency": "Top word frequencies (Counter)",
            "GET /correlation": "Correlation matrix of text metrics",
            "GET /histogram": "Histogram data for any column",
            "POST /filter": "Filter texts by criteria",
            "GET /dataframe/head": "First 5 rows of DataFrame",
            "GET /health": "Health check",
        },
    }


@app.get("/stats")
def stats():
    df = get_df()
    return compute_statistics(df)


@app.get("/word-frequency")
def word_freq(top_n: int = 20):
    df = get_df()
    return {"frequencies": word_frequency(df, top_n=top_n)}


@app.get("/correlation")
def correlation():
    df = get_df()
    return {"matrix": correlation_matrix(df)}


@app.post("/histogram")
def histogram(req: HistogramRequest):
    df = get_df()
    return generate_histogram_data(df, column=req.column, bins=req.bins)


@app.post("/filter")
def filter_endpoint(req: FilterRequest):
    df = get_df()
    result = filter_texts(
        df, min_words=req.min_words, max_words=req.max_words, has_number=req.has_number
    )
    return {"count": len(result), "texts": result["text"].tolist()}


@app.get("/dataframe/head")
def dataframe_head():
    df = get_df()
    return {
        "rows": df.head().to_dict(orient="records"),
        "columns": list(df.columns),
        "shape": list(df.shape),
    }


@app.get("/health")
def health():
    df = get_df()
    return {"status": "ok", "texts_loaded": len(df)}
