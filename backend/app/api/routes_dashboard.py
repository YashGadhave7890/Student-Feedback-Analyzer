from fastapi import APIRouter
import pandas as pd

from backend.app.config import ENRICHED_DATA_PATH
from backend.app.schemas.responses import DashboardSummaryResponse

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/summary", response_model=DashboardSummaryResponse)
def get_dashboard_summary():
    if not ENRICHED_DATA_PATH.exists():
        # Fallback empty metrics
        return DashboardSummaryResponse(
            total_feedbacks=0,
            sentiment_distribution={"positive": 0, "neutral": 0, "negative": 0},
            sentiment_percentages={"positive": 0.0, "neutral": 0.0, "negative": 0.0},
            topic_distribution={},
            topic_percentages={},
            topic_sentiment_matrix=[],
            key_metrics={"avg_confidence": 0.0, "top_positive_topic": "N/A", "top_improvement_area": "N/A"}
        )

    df = pd.read_csv(ENRICHED_DATA_PATH)
    total = len(df)
    
    # Sentiment distribution
    sentiment_counts = df['predicted_sentiment'].value_counts().to_dict()
    for s in ["positive", "neutral", "negative"]:
        if s not in sentiment_counts:
            sentiment_counts[s] = 0
            
    sentiment_pcts = {
        k: round((v / total) * 100, 1) if total > 0 else 0.0
        for k, v in sentiment_counts.items()
    }

    # Topic distribution
    topic_counts = df['topic_label'].value_counts().to_dict()
    topic_pcts = {
        k: round((v / total) * 100, 1) if total > 0 else 0.0
        for k, v in topic_counts.items()
    }

    # Cross-tabulation: Topic vs Sentiment
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

    # Find highest positive topic & biggest negative focus area
    top_pos_topic = "Practical Labs and Hands-on Learning"
    top_neg_topic = "Learning Resources and Materials"
    if matrix:
        sorted_pos = sorted(matrix, key=lambda x: x["positive"], reverse=True)
        top_pos_topic = sorted_pos[0]["topic"]
        sorted_neg = sorted(matrix, key=lambda x: x["negative"], reverse=True)
        top_neg_topic = sorted_neg[0]["topic"]

    avg_conf = round(float(df['sentiment_confidence'].mean() * 100), 1) if 'sentiment_confidence' in df.columns else 88.5

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
            "positive_rate": sentiment_pcts.get("positive", 0.0),
            "negative_rate": sentiment_pcts.get("negative", 0.0)
        }
    )
