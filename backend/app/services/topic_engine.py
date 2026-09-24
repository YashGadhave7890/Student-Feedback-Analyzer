import json
from typing import List, Dict, Any
import joblib
import numpy as np

from app.config import TOPIC_MODEL_PATH, TOPIC_VECTORIZER_PATH, TOPIC_LABELS_PATH
from app.services.preprocessor import clean_text_basic


class TopicEngine:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.labels = {}
        self._load()

    def _load(self):
        if TOPIC_MODEL_PATH.exists() and TOPIC_VECTORIZER_PATH.exists():
            self.model = joblib.load(TOPIC_MODEL_PATH)
            self.vectorizer = joblib.load(TOPIC_VECTORIZER_PATH)
        else:
            raise FileNotFoundError("Topic modeling artifacts are missing. Run training script first.")

        if TOPIC_LABELS_PATH.exists():
            with open(TOPIC_LABELS_PATH, "r", encoding="utf-8") as f:
                raw_labels = json.load(f)
                self.labels = {int(k): v for k, v in raw_labels.items()}
        else:
            self.labels = {i: f"Topic {i+1}" for i in range(5)}

    def predict_single(self, text: str) -> Dict[str, Any]:
        cleaned = clean_text_basic(text)
        if not cleaned:
            return {
                "dominant_topic": 0,
                "topic_label": self.labels.get(0, "Topic 1"),
                "confidence": 0.2,
                "probabilities": {self.labels.get(i, f"Topic {i+1}"): 0.2 for i in range(len(self.labels))},
                "top_keywords": self.get_top_keywords(0),
                "clean_topic_text": ""
            }

        vec = self.vectorizer.transform([cleaned])
        probs = self.model.transform(vec)[0]
        topic_id = int(np.argmax(probs))
        confidence = float(probs[topic_id])

        prob_dict = {
            self.labels.get(i, f"Topic {i+1}"): round(float(p), 4)
            for i, p in enumerate(probs)
        }

        return {
            "dominant_topic": topic_id,
            "topic_label": self.labels.get(topic_id, f"Topic {topic_id+1}"),
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "top_keywords": self.get_top_keywords(topic_id),
            "clean_topic_text": cleaned
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        cleaned_list = [clean_text_basic(t) for t in texts]
        vecs = self.vectorizer.transform(cleaned_list)
        probs_matrix = self.model.transform(vecs)
        
        results = []
        for probs in probs_matrix:
            topic_id = int(np.argmax(probs))
            confidence = float(probs[topic_id])
            results.append({
                "dominant_topic": topic_id,
                "topic_label": self.labels.get(topic_id, f"Topic {topic_id+1}"),
                "confidence": round(confidence, 4)
            })
        return results

    def get_top_keywords(self, topic_id: int, top_n: int = 10) -> List[str]:
        feature_names = self.vectorizer.get_feature_names_out()
        topic_components = self.model.components_[topic_id]
        top_indices = topic_components.argsort()[-top_n:][::-1]
        return [feature_names[i] for i in top_indices]

    def get_topic_word_weights(self, topic_id: int, top_n: int = 12) -> List[Dict[str, Any]]:
        feature_names = self.vectorizer.get_feature_names_out()
        topic_components = self.model.components_[topic_id]
        norm_weights = topic_components / topic_components.sum()
        top_indices = topic_components.argsort()[-top_n:][::-1]
        
        return [
            {"word": str(feature_names[i]), "weight": round(float(norm_weights[i]), 4)}
            for i in top_indices
        ]


# Singleton instance
topic_engine = TopicEngine()
