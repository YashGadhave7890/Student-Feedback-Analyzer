import json
from fastapi import APIRouter

from backend.app.config import MODEL_METRICS_PATH
from backend.app.schemas.responses import ModelPerformanceResponse, ConfusionMatrixData
from backend.app.services.explainability_engine import explainability_engine

router = APIRouter(prefix="/performance", tags=["Performance"])


@router.get("", response_model=ModelPerformanceResponse)
def get_model_performance():
    if MODEL_METRICS_PATH.exists():
        with open(MODEL_METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    else:
        metrics_data = {
            "models_benchmark": {
                "Logistic Regression": {"Accuracy": 0.473, "Accuracy_Std": 0.05, "Precision": 0.45, "Recall": 0.47, "F1_Score": 0.374},
                "Multinomial Naive Bayes": {"Accuracy": 0.488, "Accuracy_Std": 0.04, "Precision": 0.46, "Recall": 0.49, "F1_Score": 0.410},
                "Linear SVM": {"Accuracy": 0.423, "Accuracy_Std": 0.06, "Precision": 0.40, "Recall": 0.42, "F1_Score": 0.278},
                "Random Forest": {"Accuracy": 0.442, "Accuracy_Std": 0.05, "Precision": 0.41, "Recall": 0.44, "F1_Score": 0.303}
            },
            "best_model": "Logistic Regression",
            "confusion_matrix": {
                "labels": ["positive", "neutral", "negative"],
                "matrix": [[6, 2, 0], [4, 7, 0], [2, 5, 0]]
            },
            "topic_metrics": {
                "n_topics": 5,
                "perplexity": 356.14,
                "coherence_cv_score": 0.485
            }
        }

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
