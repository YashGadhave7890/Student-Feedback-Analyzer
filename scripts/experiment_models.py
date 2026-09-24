"""
Systematic Model Evaluation and Comparison on Unified Sentiment Corpus.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "backend"))

from app.services.preprocessor import clean_text_for_sentiment

CORPUS_PATH = BASE_DIR / "data" / "sentiment_training_corpus.csv"
UNSEEN_TEST_PATH = BASE_DIR / "data" / "unseen_test_feedback.json"


def run_experiment():
    df = pd.read_csv(CORPUS_PATH)
    df['clean_text'] = df['text'].apply(clean_text_for_sentiment)
    
    # Filter empty texts if any
    df = df[df['clean_text'].str.strip().str.len() > 0].reset_index(drop=True)
    
    X_raw = df['clean_text']
    y = df['sentiment']
    
    print(f"Loaded {len(df)} records for training/validation.")
    print("Class Distribution:\n", y.value_counts())
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=1500,
        sublinear_tf=True,
        min_df=1
    )
    X = vectorizer.fit_transform(X_raw)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    candidates = {
        "Logistic Regression (Balanced)": LogisticRegression(
            C=2.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=42
        ),
        "Calibrated Linear SVM (Balanced)": CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=2000),
            cv=3
        ),
        "Complement Naive Bayes": ComplementNB(alpha=0.5),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.3),
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            class_weight="balanced",
            random_state=42
        )
    }
    
    print("\n" + "=" * 80)
    print("5-FOLD STRATIFIED CROSS-VALIDATION BENCHMARKS")
    print("=" * 80)
    
    for name, model in candidates.items():
        scores = cross_validate(
            model, X, y,
            scoring=["accuracy", "f1_macro", "f1_weighted", "precision_weighted", "recall_weighted"],
            cv=cv,
            n_jobs=-1
        )
        print(f"[{name}]")
        print(f"  Accuracy:  {scores['test_accuracy'].mean()*100:.2f}% (± {scores['test_accuracy'].std()*100:.2f}%)")
        print(f"  Macro F1:  {scores['test_f1_macro'].mean()*100:.2f}%")
        print(f"  Weighted F1: {scores['test_f1_weighted'].mean()*100:.2f}%")
        print("-" * 50)
        
    print("\n" + "=" * 80)
    print("HELD-OUT 20% STRATIFIED TEST EVALUATION")
    print("=" * 80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    classes = ["positive", "neutral", "negative"]
    
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        print(f"\n--- Model: {name} ---")
        print(classification_report(y_test, y_pred, labels=classes, zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        print("Confusion Matrix (Labels: positive, neutral, negative):")
        print(cm)


if __name__ == "__main__":
    run_experiment()
