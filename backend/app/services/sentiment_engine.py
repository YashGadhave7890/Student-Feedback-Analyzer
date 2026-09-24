from typing import List, Dict, Any
import joblib
import numpy as np

from app.config import SENTIMENT_MODEL_PATH, SENTIMENT_VECTORIZER_PATH
from app.services.preprocessor import clean_text_for_sentiment


class SentimentEngine:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.classes = []
        self._load()

    def _load(self):
        if SENTIMENT_MODEL_PATH.exists() and SENTIMENT_VECTORIZER_PATH.exists():
            self.model = joblib.load(SENTIMENT_MODEL_PATH)
            self.vectorizer = joblib.load(SENTIMENT_VECTORIZER_PATH)
            self.classes = list(self.model.classes_)
        else:
            raise FileNotFoundError("Sentiment artifacts missing. Run training script first.")

    def predict_single(self, text: str) -> Dict[str, Any]:
        cleaned = clean_text_for_sentiment(text)
        if not cleaned:
            return {
                "sentiment": "neutral",
                "confidence": 0.34,
                "probabilities": {"positive": 0.33, "neutral": 0.34, "negative": 0.33},
                "clean_sentiment_text": ""
            }

        vec = self.vectorizer.transform([cleaned])
        pred = self.model.predict(vec)[0]
        
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(vec)[0]
            confidence = float(max(proba))
            prob_dict = {
                str(cls_name): round(float(p), 4)
                for cls_name, p in zip(self.model.classes_, proba)
            }
        else:
            confidence = 0.85
            prob_dict = {cls_name: 1.0 if cls_name == pred else 0.0 for cls_name in self.classes}

        return {
            "sentiment": str(pred).lower(),
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "clean_sentiment_text": cleaned
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        cleaned_list = [clean_text_for_sentiment(t) for t in texts]
        vecs = self.vectorizer.transform(cleaned_list)
        preds = self.model.predict(vecs)
        
        if hasattr(self.model, "predict_proba"):
            proba_matrix = self.model.predict_proba(vecs)
        else:
            proba_matrix = None

        results = []
        for idx, pred in enumerate(preds):
            confidence = float(np.max(proba_matrix[idx])) if proba_matrix is not None else 0.85
            results.append({
                "sentiment": str(pred).lower(),
                "confidence": round(confidence, 4)
            })
        return results


# Singleton instance
sentiment_engine = SentimentEngine()
