import os
import sys
from pathlib import Path

# Resolve base directories
BASE_DIR = Path(__file__).resolve().parent.parent  # .../backend
PROJECT_ROOT = BASE_DIR.parent

# Add directories to sys.path so imports work regardless of launch directory
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Data directory resolution (checks both backend/data and root data/)
if (BASE_DIR / "data").exists():
    DATA_DIR = BASE_DIR / "data"
elif (PROJECT_ROOT / "data").exists():
    DATA_DIR = PROJECT_ROOT / "data"
else:
    DATA_DIR = BASE_DIR / "data"

MODELS_DIR = BASE_DIR / "trained_models"

# Dataset paths
RAW_DATA_PATH = DATA_DIR / "202511-ft_bi1_bi2_course_evaluation.csv"
ENRICHED_DATA_PATH = DATA_DIR / "course_evals_with_topics_and_sentiments.csv"

# Model paths
TOPIC_MODEL_PATH = MODELS_DIR / "topic_model_lda.pkl"
TOPIC_VECTORIZER_PATH = MODELS_DIR / "topic_vectorizer.pkl"
SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_classifier.pkl"
SENTIMENT_VECTORIZER_PATH = MODELS_DIR / "sentiment_tfidf_vectorizer.pkl"
TOPIC_LABELS_PATH = MODELS_DIR / "topic_labels.json"
MODEL_METRICS_PATH = MODELS_DIR / "model_metrics.json"

# In-memory storage directory for temporary batch exports (no database used)
TEMP_EXPORTS_DIR = BASE_DIR / "temp_exports"
TEMP_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
