import io
import uuid
from typing import List, Dict, Any, Optional
import pandas as pd
from fastapi import HTTPException

from app.services.preprocessor import clean_text_basic, clean_text_for_sentiment
from app.services.topic_engine import topic_engine
from app.services.sentiment_engine import sentiment_engine
from app.services.explainability_engine import explainability_engine
from app.services.dataset_validator import dataset_validator
from app.config import ENRICHED_DATA_PATH


class CombinedNLPService:
    """
    Combined NLP service that coordinates topic modelling, sentiment analysis,
    and explainability for both single and batch feedback processing.
    """

    def analyze_single(self, text: str) -> Dict[str, Any]:
        text_str = text.strip() if text else ""
        if not text_str:
            raise HTTPException(status_code=400, detail="Feedback text cannot be empty.")

        # 1. Topic inference
        topic_res = topic_engine.predict_single(text_str)

        # 2. Sentiment inference
        sentiment_res = sentiment_engine.predict_single(text_str)

        # 3. Explainability / Token attributions
        tokens = explainability_engine.explain_text(text_str)

        return {
            "original_text": text_str,
            "clean_sentiment_text": sentiment_res["clean_sentiment_text"],
            "clean_topic_text": topic_res["clean_topic_text"],
            "sentiment": sentiment_res["sentiment"],
            "sentiment_confidence": sentiment_res["confidence"],
            "sentiment_probabilities": sentiment_res["probabilities"],
            "dominant_topic": topic_res["dominant_topic"],
            "topic_label": topic_res["topic_label"],
            "topic_confidence": topic_res["confidence"],
            "topic_probabilities": topic_res["probabilities"],
            "top_topic_keywords": topic_res["top_keywords"],
            "token_attributions": tokens,
        }

    def analyze_batch_texts(self, texts: List[str]) -> Dict[str, Any]:
        if not texts:
            raise HTTPException(status_code=400, detail="Texts list cannot be empty.")

        topic_results = topic_engine.predict_batch(texts)
        sentiment_results = sentiment_engine.predict_batch(texts)

        results_list = []
        sent_breakdown = {"positive": 0, "neutral": 0, "negative": 0}
        topic_breakdown = {}

        for i, text_val in enumerate(texts):
            s_res = sentiment_results[i]
            t_res = topic_results[i]

            sent = s_res["sentiment"]
            s_conf = s_res["confidence"]
            top_id = t_res["dominant_topic"]
            t_label = t_res["topic_label"]
            t_conf = t_res["confidence"]

            sent_breakdown[sent] = sent_breakdown.get(sent, 0) + 1
            topic_breakdown[t_label] = topic_breakdown.get(t_label, 0) + 1

            results_list.append({
                "id": i + 1,
                "text": text_val,
                "sentiment": sent,
                "sentiment_confidence": s_conf,
                "dominant_topic": top_id,
                "topic_label": t_label,
                "topic_confidence": t_conf,
            })

        return {
            "total_processed": len(texts),
            "sentiment_breakdown": sent_breakdown,
            "topic_breakdown": topic_breakdown,
            "results": results_list,
        }

    def process_csv_upload(self, content: bytes, text_column: Optional[str] = None) -> Dict[str, Any]:
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty.")

        # Determine target text column
        if not text_column or text_column not in df.columns:
            text_column = dataset_validator.find_best_text_column(df)
            if not text_column:
                raise HTTPException(
                    status_code=400,
                    detail="Please select a feedback/text column containing natural-language responses."
                )

        # Validate the chosen text column
        is_valid, err_msg = dataset_validator.validate_text_column(df, text_column)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail="Please select a feedback/text column containing natural-language responses."
            )

        texts = df[text_column].fillna("").astype(str).tolist()
        batch_res = self.analyze_batch_texts(texts)

        # Append predictions
        df["predicted_sentiment"] = [r["sentiment"] for r in batch_res["results"]]
        df["sentiment_confidence"] = [r["sentiment_confidence"] for r in batch_res["results"]]
        df["dominant_topic"] = [r["dominant_topic"] for r in batch_res["results"]]
        df["topic_label"] = [r["topic_label"] for r in batch_res["results"]]
        df["topic_confidence"] = [r["topic_confidence"] for r in batch_res["results"]]

        # Return CSV string for streaming download
        out_buffer = io.StringIO()
        df.to_csv(out_buffer, index=False)
        csv_string = out_buffer.getvalue()

        return {
            "total_processed": batch_res["total_processed"],
            "sentiment_breakdown": batch_res["sentiment_breakdown"],
            "topic_breakdown": batch_res["topic_breakdown"],
            "results": batch_res["results"][:200],
            "csv_content": csv_string,
        }


combined_nlp_service = CombinedNLPService()
