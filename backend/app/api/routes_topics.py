from fastapi import APIRouter
import pandas as pd

from backend.app.config import ENRICHED_DATA_PATH
from backend.app.schemas.responses import TopicsListResponse, TopicDetail
from backend.app.services.topic_engine import topic_engine

router = APIRouter(prefix="/topics", tags=["Topics"])


@router.get("", response_model=TopicsListResponse)
def get_all_topics():
    df = pd.read_csv(ENRICHED_DATA_PATH) if ENRICHED_DATA_PATH.exists() else pd.DataFrame()
    total_feedbacks = len(df)

    topics_list = []
    for topic_id, label in topic_engine.labels.items():
        top_words = topic_engine.get_top_keywords(topic_id, top_n=10)
        word_weights = topic_engine.get_topic_word_weights(topic_id, top_n=10)

        # Topic distribution stats from baseline CSV
        if not df.empty and 'dominant_topic' in df.columns:
            topic_df = df[df['dominant_topic'] == topic_id]
            topic_count = len(topic_df)
            pct = round((topic_count / total_feedbacks) * 100, 1) if total_feedbacks > 0 else 0.0

            sent_counts = topic_df['predicted_sentiment'].value_counts().to_dict()
            for s in ["positive", "neutral", "negative"]:
                if s not in sent_counts:
                    sent_counts[s] = 0

            # Representative quotes
            sample_quotes = []
            if not topic_df.empty:
                for _, row in topic_df.head(4).iterrows():
                    sample_quotes.append({
                        "text": str(row.get("text", "")),
                        "sentiment": str(row.get("predicted_sentiment", "neutral")),
                        "confidence": round(float(row.get("topic_probability", 0.85)), 2)
                    })
        else:
            topic_count = 0
            pct = 0.0
            sent_counts = {"positive": 0, "neutral": 0, "negative": 0}
            sample_quotes = []

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
