from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class TokenWeight(BaseModel):
    word: str
    weight: float
    sentiment_tag: str  # "positive", "neutral", "negative", or "neutral_feature"


class SingleAnalysisResponse(BaseModel):
    original_text: str
    clean_sentiment_text: str
    clean_topic_text: str
    sentiment: str
    sentiment_confidence: float
    sentiment_probabilities: Dict[str, float]
    dominant_topic: int
    topic_label: str
    topic_confidence: float
    topic_probabilities: Dict[str, float]
    top_topic_keywords: List[str]
    token_attributions: List[TokenWeight]


class BulkItemResult(BaseModel):
    id: int
    text: str
    sentiment: str
    sentiment_confidence: float
    dominant_topic: int
    topic_label: str
    topic_confidence: float


class BulkAnalysisResponse(BaseModel):
    total_processed: int
    sentiment_breakdown: Dict[str, int]
    topic_breakdown: Dict[str, int]
    results: List[BulkItemResult]
    download_token: str


class DashboardSummaryResponse(BaseModel):
    total_feedbacks: int
    sentiment_distribution: Dict[str, int]
    sentiment_percentages: Dict[str, float]
    topic_distribution: Dict[str, int]
    topic_percentages: Dict[str, float]
    topic_sentiment_matrix: List[Dict[str, Any]]
    key_metrics: Dict[str, Any]


class TopicDetail(BaseModel):
    topic_id: int
    label: str
    total_feedbacks: int
    percentage: float
    top_words: List[str]
    word_weights: List[Dict[str, Any]]
    sentiment_breakdown: Dict[str, int]
    representative_samples: List[Dict[str, Any]]


class TopicsListResponse(BaseModel):
    topics: List[TopicDetail]


class ConfusionMatrixData(BaseModel):
    labels: List[str]
    matrix: List[List[int]]


class ModelBenchmarkItem(BaseModel):
    Accuracy: float
    Accuracy_Std: float
    Precision: float
    Recall: float
    F1_Score: float


class ModelPerformanceResponse(BaseModel):
    models_benchmark: Dict[str, ModelBenchmarkItem]
    best_model: str
    confusion_matrix: ConfusionMatrixData
    topic_metrics: Dict[str, Any]
    feature_importance: Dict[str, List[Dict[str, Any]]]


class DatasetValidationResponse(BaseModel):
    is_valid: bool
    total_rows: int
    detected_columns: List[str]
    recommended_text_column: Optional[str] = None
    missing_values_count: int
    preview: List[Dict[str, Any]]
    errors: List[str]
