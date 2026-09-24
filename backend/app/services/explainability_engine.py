from typing import List, Dict, Any
import numpy as np

from app.services.sentiment_engine import sentiment_engine
from app.services.preprocessor import clean_text_basic, STOP_WORDS


class ExplainabilityEngine:
    def __init__(self):
        self._feature_names = None
        self._coef_dict = {}
        self._init_weights()

    def _init_weights(self):
        if sentiment_engine.model is not None and sentiment_engine.vectorizer is not None:
            self._feature_names = sentiment_engine.vectorizer.get_feature_names_out()
            
            if hasattr(sentiment_engine.model, "coef_"):
                coefs = sentiment_engine.model.coef_
                classes = list(sentiment_engine.model.classes_)
                
                for idx, cls_name in enumerate(classes):
                    self._coef_dict[cls_name] = {
                        self._feature_names[i]: float(coefs[idx][i])
                        for i in range(len(self._feature_names))
                    }

    def explain_text(self, text: str) -> List[Dict[str, Any]]:
        tokens = [w for w in clean_text_basic(text).split() if w not in STOP_WORDS]
        if not tokens:
            return []

        pos_dict = self._coef_dict.get("positive", {})
        neg_dict = self._coef_dict.get("negative", {})
        
        token_attributions = []
        for word in tokens:
            pos_weight = pos_dict.get(word, 0.0)
            neg_weight = neg_dict.get(word, 0.0)
            net_weight = pos_weight - neg_weight
            
            if net_weight > 0.15:
                tag = "positive"
            elif net_weight < -0.15:
                tag = "negative"
            else:
                tag = "neutral"
                
            token_attributions.append({
                "word": word,
                "weight": round(float(net_weight), 3),
                "sentiment_tag": tag
            })
            
        return token_attributions

    def get_global_feature_importance(self, top_n: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        result = {"positive": [], "negative": [], "neutral": []}
        
        for cls_name, weights in self._coef_dict.items():
            sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:top_n]
            result[cls_name] = [
                {"term": term, "coefficient": round(weight, 4)}
                for term, weight in sorted_items if weight > 0
            ]
            
        return result


explainability_engine = ExplainabilityEngine()
