import io
import json
import uuid
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import StreamingResponse
import pandas as pd

from app.config import ENRICHED_DATA_PATH, MODEL_METRICS_PATH, TEMP_EXPORTS_DIR
from app.schemas.requests import FeedbackRequest, BatchFeedbackRequest
from app.schemas.responses import (
    SingleAnalysisResponse,
    BulkAnalysisResponse,
    BulkItemResult,
    DashboardSummaryResponse,
    TopicsListResponse,
    TopicDetail,
    ModelPerformanceResponse,
    ConfusionMatrixData,
    DatasetValidationResponse
)
from app.services.combined_nlp_service import combined_nlp_service
from app.services.topic_engine import topic_engine
from app.services.sentiment_engine import sentiment_engine
from app.services.explainability_engine import explainability_engine
from app.services.dataset_validator import dataset_validator

router = APIRouter()


# -------------------------------------------------------------
# 1. Health Endpoint
# -------------------------------------------------------------
@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "student-feedback-intelligence-backend",
        "version": "1.0.0",
        "models_loaded": topic_engine.model is not None and sentiment_engine.model is not None,
        "database": "none (in-memory / file-based)"
    }


# -------------------------------------------------------------
# 2. Single Feedback Analysis Endpoint
# -------------------------------------------------------------
@router.post("/analyze", response_model=SingleAnalysisResponse)
def analyze_feedback(request: FeedbackRequest):
    return combined_nlp_service.analyze_single(request.text)


# Also support /analyze/single for backward-compatibility
@router.post("/analyze/single", response_model=SingleAnalysisResponse)
def analyze_feedback_single(request: FeedbackRequest):
    return combined_nlp_service.analyze_single(request.text)


# -------------------------------------------------------------
# 3. Batch Feedback Analysis Endpoint (JSON or CSV)
# -------------------------------------------------------------
@router.post("/analyze/batch", response_model=BulkAnalysisResponse)
async def analyze_batch_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    text_column: Optional[str] = Form(None)
):
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type and file is not None:
        content = await file.read()
        res = combined_nlp_service.process_csv_upload(content, text_column)
        
        # Save temporary CSV for streaming download (no database)
        download_token = str(uuid.uuid4())
        temp_file = TEMP_EXPORTS_DIR / f"analysis_{download_token}.csv"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(res["csv_content"])

        return BulkAnalysisResponse(
            total_processed=res["total_processed"],
            sentiment_breakdown=res["sentiment_breakdown"],
            topic_breakdown=res["topic_breakdown"],
            results=[BulkItemResult(**r) for r in res["results"]],
            download_token=download_token
        )

    # Parse JSON payload
    try:
        body = await request.json()
        texts = body.get("texts", [])
        if not texts:
            raise HTTPException(status_code=400, detail="Please provide a non-empty 'texts' list or upload a CSV file.")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid request payload. Expected JSON {'texts': [...]} or CSV file.")

    batch_res = combined_nlp_service.analyze_batch_texts(texts)
    download_token = str(uuid.uuid4())
    
    # Save temporary CSV
    df = pd.DataFrame(batch_res["results"])
    temp_file = TEMP_EXPORTS_DIR / f"analysis_{download_token}.csv"
    df.to_csv(temp_file, index=False)

    return BulkAnalysisResponse(
        total_processed=batch_res["total_processed"],
        sentiment_breakdown=batch_res["sentiment_breakdown"],
        topic_breakdown=batch_res["topic_breakdown"],
        results=[BulkItemResult(**r) for r in batch_res["results"][:200]],
        download_token=download_token
    )


# Bulk upload alias for UI
@router.post("/analyze/bulk", response_model=BulkAnalysisResponse)
async def analyze_bulk_alias(
    file: UploadFile = File(...),
    text_column: Optional[str] = Form(None)
):
    content = await file.read()
    res = combined_nlp_service.process_csv_upload(content, text_column)
    
    download_token = str(uuid.uuid4())
    temp_file = TEMP_EXPORTS_DIR / f"analysis_{download_token}.csv"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(res["csv_content"])

    return BulkAnalysisResponse(
        total_processed=res["total_processed"],
        sentiment_breakdown=res["sentiment_breakdown"],
        topic_breakdown=res["topic_breakdown"],
        results=[BulkItemResult(**r) for r in res["results"]],
        download_token=download_token
    )


@router.post("/analyze/validate", response_model=DatasetValidationResponse)
async def validate_dataset_route(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    content = await file.read()
    return dataset_validator.validate_csv(content)


@router.get("/analyze/export/{token}")
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


# -------------------------------------------------------------
# 4. Dashboard Endpoint
# -------------------------------------------------------------
@router.get("/dashboard", response_model=DashboardSummaryResponse)
@router.get("/dashboard/summary", response_model=DashboardSummaryResponse)
def get_dashboard():
    if not ENRICHED_DATA_PATH.exists():
        raise HTTPException(status_code=500, detail="Enriched dataset not found. Run training script first.")

    df = pd.read_csv(ENRICHED_DATA_PATH)
    total = len(df)

    sentiment_counts = df['predicted_sentiment'].value_counts().to_dict()
    for s in ["positive", "neutral", "negative"]:
        if s not in sentiment_counts:
            sentiment_counts[s] = 0

    sentiment_pcts = {
        k: round((v / total) * 100, 1) if total > 0 else 0.0
        for k, v in sentiment_counts.items()
    }

    topic_counts = df['topic_label'].value_counts().to_dict()
    topic_pcts = {
        k: round((v / total) * 100, 1) if total > 0 else 0.0
        for k, v in topic_counts.items()
    }

    cross = pd.crosstab(df['topic_label'], df['predicted_sentiment']).fillna(0)
    matrix = []
    for topic_label, row in cross.iterrows():
        pos = int(row.get('positive', 0))
        neu = int(row.get('neutral', 0))
        neg = int(row.get('negative', 0))
        matrix.append({
            "topic": topic_label,
            "positive": pos,
            "neutral": neu,
            "negative": neg,
            "total": pos + neu + neg
        })

    label_to_id = {label: tid for tid, label in topic_engine.labels.items()}

    if matrix:
        sorted_pos = sorted(matrix, key=lambda x: x["positive"], reverse=True)
        top_pos_item = sorted_pos[0]
        top_pos_topic = top_pos_item["topic"]
        pos_count = top_pos_item["positive"]
        pos_tot = top_pos_item["total"]
        pos_pct = round((pos_count / pos_tot) * 100, 1) if pos_tot > 0 else 0.0
        pos_id = label_to_id.get(top_pos_topic)
        pos_keywords = topic_engine.get_top_keywords(pos_id, top_n=4) if pos_id is not None else []
        pos_kw_str = ", ".join(pos_keywords) if pos_keywords else "N/A"
        top_pos_insight = f"{top_pos_topic} recorded the highest positive response ({pos_pct}% positive across {pos_tot} evaluations) driven by terms: {pos_kw_str}."

        sorted_neg = sorted(matrix, key=lambda x: x["negative"], reverse=True)
        top_neg_item = sorted_neg[0]
        top_neg_topic = top_neg_item["topic"]
        neg_count = top_neg_item["negative"]
        neg_tot = top_neg_item["total"]
        neg_pct = round((neg_count / neg_tot) * 100, 1) if neg_tot > 0 else 0.0
        neg_id = label_to_id.get(top_neg_topic)
        neg_keywords = topic_engine.get_top_keywords(neg_id, top_n=4) if neg_id is not None else []
        neg_kw_str = ", ".join(neg_keywords) if neg_keywords else "N/A"
        top_neg_insight = f"{top_neg_topic} shows the highest concentration of negative feedback ({neg_pct}% negative across {neg_tot} evaluations) associated with terms: {neg_kw_str}."
    else:
        top_pos_topic = "N/A"
        top_neg_topic = "N/A"
        top_pos_insight = "No feedback data available."
        top_neg_insight = "No feedback data available."

    avg_conf = round(float(df['sentiment_confidence'].mean() * 100), 1) if 'sentiment_confidence' in df.columns else 0.0

    return DashboardSummaryResponse(
        total_feedbacks=total,
        sentiment_distribution=sentiment_counts,
        sentiment_percentages=sentiment_pcts,
        topic_distribution=topic_counts,
        topic_percentages=topic_pcts,
        topic_sentiment_matrix=matrix,
        key_metrics={
            "avg_confidence": avg_conf,
            "top_positive_topic": top_pos_topic,
            "top_improvement_area": top_neg_topic,
            "top_positive_insight": top_pos_insight,
            "top_improvement_insight": top_neg_insight,
            "positive_rate": sentiment_pcts.get("positive", 0.0),
            "negative_rate": sentiment_pcts.get("negative", 0.0)
        }
    )


# -------------------------------------------------------------
# 5. Topics Endpoint
# -------------------------------------------------------------
@router.get("/topics", response_model=TopicsListResponse)
def get_topics():
    if not ENRICHED_DATA_PATH.exists():
        raise HTTPException(status_code=500, detail="Dataset not found. Run training script first.")

    df = pd.read_csv(ENRICHED_DATA_PATH)
    total_feedbacks = len(df)

    topics_list = []
    for topic_id, label in topic_engine.labels.items():
        top_words = topic_engine.get_top_keywords(topic_id, top_n=10)
        word_weights = topic_engine.get_topic_word_weights(topic_id, top_n=10)

        topic_df = df[df['dominant_topic'] == topic_id] if 'dominant_topic' in df.columns else pd.DataFrame()
        topic_count = len(topic_df)
        pct = round((topic_count / total_feedbacks) * 100, 1) if total_feedbacks > 0 else 0.0

        sent_counts = topic_df['predicted_sentiment'].value_counts().to_dict() if not topic_df.empty else {}
        for s in ["positive", "neutral", "negative"]:
            if s not in sent_counts:
                sent_counts[s] = 0

        sample_quotes = []
        if not topic_df.empty:
            for _, row in topic_df.head(4).iterrows():
                sample_quotes.append({
                    "text": str(row.get("text", "")),
                    "sentiment": str(row.get("predicted_sentiment", "neutral")),
                    "confidence": round(float(row.get("topic_probability", 0.85)), 2)
                })

        topics_list.append(TopicDetail(
            topic_id=topic_id,
            label=label,
            total_feedbacks=topic_count,
            percentage=pct,
            top_words=top_words,
            word_weights=word_weights,
            sentiment_breakdown=sent_counts,
            representative_samples=sample_quotes
        ))

    return TopicsListResponse(topics=topics_list)


# -------------------------------------------------------------
# 6. Sentiment Endpoint
# -------------------------------------------------------------
@router.get("/sentiment")
def get_sentiment_details():
    if not ENRICHED_DATA_PATH.exists():
        raise HTTPException(status_code=500, detail="Dataset not found. Run training script first.")

    df = pd.read_csv(ENRICHED_DATA_PATH)
    total = len(df)
    sent_counts = df['predicted_sentiment'].value_counts().to_dict()
    for s in ["positive", "neutral", "negative"]:
        if s not in sent_counts:
            sentiment_counts[s] = 0

    sent_pcts = {
        k: round((v / total) * 100, 1) if total > 0 else 0.0
        for k, v in sent_counts.items()
    }

    feature_imp = explainability_engine.get_global_feature_importance(top_n=10)

    return {
        "total_analyzed": total,
        "distribution": sent_counts,
        "percentages": sent_pcts,
        "top_features": {
            "positive": feature_imp.get("positive", []),
            "negative": feature_imp.get("negative", [])
        }
    }


# -------------------------------------------------------------
# 7. Models / Performance Endpoint
# -------------------------------------------------------------
@router.get("/models", response_model=ModelPerformanceResponse)
@router.get("/performance", response_model=ModelPerformanceResponse)
def get_models_performance():
    if not MODEL_METRICS_PATH.exists():
        raise HTTPException(status_code=500, detail="Model metrics file not found. Run training script first.")

    with open(MODEL_METRICS_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    feature_imp = explainability_engine.get_global_feature_importance(top_n=10)

    return ModelPerformanceResponse(
        models_benchmark=metrics_data["models_benchmark"],
        best_model=metrics_data.get("best_model", "Logistic Regression"),
        confusion_matrix=ConfusionMatrixData(
            labels=metrics_data["confusion_matrix"]["labels"],
            matrix=metrics_data["confusion_matrix"]["matrix"]
        ),
        topic_metrics=metrics_data.get("topic_metrics", {}),
        feature_importance=feature_imp
    )
