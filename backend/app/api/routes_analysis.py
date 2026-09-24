import io
import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd

from backend.app.config import TEMP_EXPORTS_DIR
from backend.app.schemas.requests import FeedbackAnalysisRequest
from backend.app.schemas.responses import (
    SingleAnalysisResponse,
    BulkAnalysisResponse,
    BulkItemResult,
    DatasetValidationResponse
)
from backend.app.services.topic_engine import topic_engine
from backend.app.services.sentiment_engine import sentiment_engine
from backend.app.services.explainability_engine import explainability_engine
from backend.app.services.dataset_validator import dataset_validator

router = APIRouter(prefix="/analyze", tags=["Analysis"])


@router.post("/single", response_model=SingleAnalysisResponse)
def analyze_single_feedback(request: FeedbackAnalysisRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # 1. Topic inference
    topic_res = topic_engine.predict_single(text)

    # 2. Sentiment inference
    sentiment_res = sentiment_engine.predict_single(text)

    # 3. Token attributions & explainability
    token_weights = explainability_engine.explain_text(text)

    return SingleAnalysisResponse(
        original_text=text,
        clean_sentiment_text=sentiment_res["clean_sentiment_text"],
        clean_topic_text=topic_res["clean_topic_text"],
        sentiment=sentiment_res["sentiment"],
        sentiment_confidence=sentiment_res["confidence"],
        sentiment_probabilities=sentiment_res["probabilities"],
        dominant_topic=topic_res["dominant_topic"],
        topic_label=topic_res["topic_label"],
        topic_confidence=topic_res["confidence"],
        topic_probabilities=topic_res["probabilities"],
        top_topic_keywords=topic_res["top_keywords"],
        token_attributions=token_weights
    )


@router.post("/validate", response_model=DatasetValidationResponse)
async def validate_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    content = await file.read()
    return dataset_validator.validate_csv(content)


@router.post("/bulk", response_model=BulkAnalysisResponse)
async def analyze_bulk_csv(
    file: UploadFile = File(...),
    text_column: Optional[str] = Form(None)
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty.")

    # Determine target text column
    if not text_column or text_column not in df.columns:
        # Auto-detect text column
        candidates = [
            c for c in df.columns 
            if any(k in c.lower() for k in ["feedback", "text", "comment", "review", "recommendation", "liked"])
            or df[c].dtype == object
        ]
        text_column = candidates[0] if candidates else df.columns[0]

    # Clean missing texts
    texts = df[text_column].fillna("").astype(str).tolist()

    # Batch Topic and Sentiment Predictions
    topic_results = topic_engine.predict_batch(texts)
    sentiment_results = sentiment_engine.predict_batch(texts)

    results_list = []
    sent_breakdown = {"positive": 0, "neutral": 0, "negative": 0}
    top_breakdown = {}

    predicted_sentiments = []
    sentiment_confidences = []
    dominant_topics = []
    topic_labels = []
    topic_confidences = []

    for i, text_val in enumerate(texts):
        s_res = sentiment_results[i]
        t_res = topic_results[i]

        sent = s_res["sentiment"]
        s_conf = s_res["confidence"]
        top_id = t_res["dominant_topic"]
        t_label = t_res["topic_label"]
        t_conf = t_res["confidence"]

        sent_breakdown[sent] = sent_breakdown.get(sent, 0) + 1
        top_breakdown[t_label] = top_breakdown.get(t_label, 0) + 1

        predicted_sentiments.append(sent)
        sentiment_confidences.append(s_conf)
        dominant_topics.append(top_id)
        topic_labels.append(t_label)
        topic_confidences.append(t_conf)

        if i < 200:  # Include preview items for response
            results_list.append(BulkItemResult(
                id=i + 1,
                text=text_val[:200] + ("..." if len(text_val) > 200 else ""),
                sentiment=sent,
                sentiment_confidence=s_conf,
                dominant_topic=top_id,
                topic_label=t_label,
                topic_confidence=t_conf
            ))

    # Append results into dataframe and save temporary export CSV (no database)
    df["predicted_sentiment"] = predicted_sentiments
    df["sentiment_confidence"] = sentiment_confidences
    df["dominant_topic"] = dominant_topics
    df["topic_label"] = topic_labels
    df["topic_confidence"] = topic_confidences

    download_token = str(uuid.uuid4())
    temp_file_path = TEMP_EXPORTS_DIR / f"analysis_{download_token}.csv"
    df.to_csv(temp_file_path, index=False)

    return BulkAnalysisResponse(
        total_processed=len(texts),
        sentiment_breakdown=sent_breakdown,
        topic_breakdown=top_breakdown,
        results=results_list,
        download_token=download_token
    )


@router.get("/export/{token}")
def export_analyzed_csv(token: str):
    file_path = TEMP_EXPORTS_DIR / f"analysis_{token}.csv"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Analyzed CSV export not found or expired.")

    def iterfile():
        with open(file_path, mode="rb") as f:
            yield from f

    return StreamingResponse(
        iterfile(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=student_feedback_analyzed_{token[:8]}.csv"}
    )
