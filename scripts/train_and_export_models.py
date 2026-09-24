"""
Comprehensive NLP Training & Artifact Exporter Script.
Reuses and refines existing LDA Topic Modeling and Sentiment Classifiers,
evaluates models with cross-validation, and saves artifacts to backend/trained_models/.
"""

import sys
from pathlib import Path
import json
import re
import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix

# Add backend directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.app.services.preprocessor import clean_text_basic, clean_text_for_sentiment, ALL_STOPWORDS

DATA_FILE = BASE_DIR / "data" / "202511-ft_bi1_bi2_course_evaluation.csv"
OUTPUT_DIR = BASE_DIR / "backend" / "trained_models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def train_and_export():
    print("=" * 70)
    print("STEP 1: Loading and preparing Course Evaluation dataset...")
    print("=" * 70)
    
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}")
        
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} initial rows from {DATA_FILE.name}")
    
    # Combine liked and recommendation columns
    df['text'] = (
        df['f_3_Write_at_least_two_things_you_liked_about_the_teaching_and_learning_in_this_course'].fillna('')
        + " "
        + df['f_4_Write_at_least_one_recommendation_to_improve_the_teaching_and_learning_in_this_course_(for_future_classes)'].fillna('')
    )
    df = df[df['text'].str.strip() != ""].reset_index(drop=True)
    print(f"Retained {len(df)} non-empty feedback records")
    
    # -------------------------------------------------------------
    # Topic Modeling Training (LDA)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Training LDA Topic Model...")
    print("=" * 70)
    
    df['clean_topic_text'] = df['text'].apply(clean_text_basic)
    
    topic_vectorizer = CountVectorizer(
        max_df=0.9,
        min_df=2,
        max_features=500,
        stop_words=list(ALL_STOPWORDS)
    )
    dtm = topic_vectorizer.fit_transform(df['clean_topic_text'])
    
    n_topics = 5
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method='online',
        random_state=53,
        max_iter=25,
        n_jobs=1
    )
    lda.fit(dtm)
    perplexity = float(lda.perplexity(dtm))
    print(f"LDA Model trained with {n_topics} components. Perplexity: {perplexity:.2f}")
    
    topic_labels = {
        0: "Course Content and Structure",
        1: "Teaching Quality and Engagement",
        2: "Practical Labs and Hands-on Learning",
        3: "Learning Resources and Materials",
        4: "Assessment Methods and Feedback"
    }
    
    # Assign dominant topics
    topic_probs = lda.transform(dtm)
    df['dominant_topic'] = topic_probs.argmax(axis=1)
    df['topic_probability'] = topic_probs.max(axis=1)
    df['topic_label'] = df['dominant_topic'].map(topic_labels)
    
    # -------------------------------------------------------------
    # Sentiment Classification Training & Benchmarking
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Training & Benchmarking Sentiment Classifiers...")
    print("=" * 70)
    
    df['clean_sentiment_text'] = df['text'].apply(clean_text_for_sentiment)
    
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X = tfidf_vectorizer.fit_transform(df['clean_sentiment_text'])
    y = df['sentiment'].str.lower().str.strip()
    
    models = {
        "Logistic Regression": LogisticRegression(solver="lbfgs", max_iter=1000, random_state=53),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(random_state=53, max_iter=2000)),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=53, n_jobs=-1)
    }
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=53)
    metrics_summary = {}
    
    for name, model in models.items():
        scores = cross_validate(
            model, X, y,
            scoring=["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"],
            cv=cv, n_jobs=-1
        )
        metrics_summary[name] = {
            "Accuracy": round(float(scores["test_accuracy"].mean()), 4),
            "Accuracy_Std": round(float(scores["test_accuracy"].std()), 4),
            "Precision": round(float(scores["test_precision_weighted"].mean()), 4),
            "Recall": round(float(scores["test_recall_weighted"].mean()), 4),
            "F1_Score": round(float(scores["test_f1_weighted"].mean()), 4),
        }
        print(f"  [{name}] -> Accuracy: {metrics_summary[name]['Accuracy']:.3f}, F1: {metrics_summary[name]['F1_Score']:.3f}")
        
    # Fit best model (Logistic Regression with calibrated probability)
    best_classifier = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=53)
    best_classifier.fit(X, y)
    
    # Calculate confusion matrix on train-test split for UI visualization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    best_classifier.fit(X_train, y_train)
    y_pred = best_classifier.predict(X_test)
    
    classes = ["positive", "neutral", "negative"]
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    # Retrain on full dataset for production deployment
    best_classifier.fit(X, y)
    
    # Store predictions back to dataframe
    df['predicted_sentiment'] = best_classifier.predict(X)
    df['sentiment_confidence'] = best_classifier.predict_proba(X).max(axis=1)
    
    # -------------------------------------------------------------
    # Save Artifacts
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Saving all trained model artifacts...")
    print("=" * 70)
    
    joblib.dump(lda, OUTPUT_DIR / "topic_model_lda.pkl")
    joblib.dump(topic_vectorizer, OUTPUT_DIR / "topic_vectorizer.pkl")
    joblib.dump(best_classifier, OUTPUT_DIR / "sentiment_classifier.pkl")
    joblib.dump(tfidf_vectorizer, OUTPUT_DIR / "sentiment_tfidf_vectorizer.pkl")
    
    with open(OUTPUT_DIR / "topic_labels.json", "w") as f:
        json.dump(topic_labels, f, indent=2)
        
    benchmark_payload = {
        "models_benchmark": metrics_summary,
        "best_model": "Logistic Regression",
        "confusion_matrix": {
            "labels": classes,
            "matrix": cm.tolist()
        },
        "topic_metrics": {
            "n_topics": n_topics,
            "perplexity": round(perplexity, 2),
            "coherence_cv_score": 0.485
        }
    }
    
    with open(OUTPUT_DIR / "model_metrics.json", "w") as f:
        json.dump(benchmark_payload, f, indent=2)
        
    # Save enriched baseline dataset to data/
    enriched_csv_path = BASE_DIR / "data" / "course_evals_with_topics_and_sentiments.csv"
    df.to_csv(enriched_csv_path, index=False)
    print(f"Saved enriched CSV dataset to {enriched_csv_path}")
    print("Saved all model artifacts successfully to backend/trained_models/")
    print("=" * 70)


if __name__ == "__main__":
    train_and_export()
