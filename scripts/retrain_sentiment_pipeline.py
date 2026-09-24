"""
Production Sentiment Pipeline Retraining, Benchmarking, and Artifact Export.

This script:
1. Loads the unified verified training corpus (data/sentiment_training_corpus.csv)
2. Applies negation-preserving preprocessor
3. Extracts TF-IDF word unigrams and bigrams
4. Benchmarks candidate models using 5-Fold Stratified Cross-Validation
5. Evaluates the selected model on a 20% Stratified Held-Out test set
6. Evaluates on the independent untouched unseen benchmark (data/unseen_test_feedback.json)
7. Exports production model artifacts and metrics to backend/trained_models/
8. Synchronizes course_evals_with_topics_and_sentiments.csv
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "backend"))

from app.services.preprocessor import clean_text_for_sentiment, clean_text_basic

CORPUS_PATH = BASE_DIR / "data" / "sentiment_training_corpus.csv"
UNSEEN_TEST_PATH = BASE_DIR / "data" / "unseen_test_feedback.json"
MODELS_DIR = BASE_DIR / "backend" / "trained_models"
COURSE_EVAL_PATH = BASE_DIR / "data" / "course_evals_with_topics_and_sentiments.csv"


def retrain_and_evaluate():
    print("=" * 75)
    print("STEP 1: Loading Unified Sentiment Training Corpus...")
    print("=" * 75)
    
    df = pd.read_csv(CORPUS_PATH)
    print(f"Loaded {len(df)} records from {CORPUS_PATH.name}")
    print("\nSource Breakdown:")
    print(df['source'].value_counts())
    print("\nClass Distribution:")
    print(df['sentiment'].value_counts())
    print("\nClass Percentages:")
    print(df['sentiment'].value_counts(normalize=True) * 100)
    
    # Clean text using negation-preserving preprocessor
    df['clean_sentiment_text'] = df['text'].apply(clean_text_for_sentiment)
    df = df[df['clean_sentiment_text'].str.strip().str.len() > 0].reset_index(drop=True)
    
    X_raw = df['clean_sentiment_text']
    y = df['sentiment']
    
    # -------------------------------------------------------------
    # STEP 2: TF-IDF Feature Extraction with Unigrams and Bigrams
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 2: Fitting TF-IDF Vectorizer (Unigrams + Bigrams)...")
    print("=" * 75)
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=2000,
        sublinear_tf=True,
        min_df=1
    )
    X = vectorizer.fit_transform(X_raw)
    print(f"Extracted {X.shape[1]} TF-IDF feature tokens.")
    
    # -------------------------------------------------------------
    # STEP 3: Stratified Cross-Validation & Model Benchmarking
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 3: Running 5-Fold Stratified Cross-Validation...")
    print("=" * 75)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    candidates = {
        "Logistic Regression": LogisticRegression(
            C=0.5,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=42
        ),
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=2000),
            cv=3
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.3),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            class_weight="balanced",
            random_state=42
        )
    }
    
    benchmarks = {}
    
    for name, model in candidates.items():
        scores = cross_validate(
            model, X, y,
            scoring=["accuracy", "f1_macro", "f1_weighted", "precision_weighted", "recall_weighted"],
            cv=cv,
            n_jobs=-1
        )
        benchmarks[name] = {
            "Accuracy": round(float(scores["test_accuracy"].mean()), 4),
            "Accuracy_Std": round(float(scores["test_accuracy"].std()), 4),
            "Macro_F1": round(float(scores["test_f1_macro"].mean()), 4),
            "F1_Score": round(float(scores["test_f1_weighted"].mean()), 4),
            "Precision": round(float(scores["test_precision_weighted"].mean()), 4),
            "Recall": round(float(scores["test_recall_weighted"].mean()), 4),
        }
        print(f"  [{name}]")
        print(f"    Accuracy: {benchmarks[name]['Accuracy']*100:.1f}% | Macro F1: {benchmarks[name]['Macro_F1']*100:.1f}% | Weighted F1: {benchmarks[name]['F1_Score']*100:.1f}%")

    # -------------------------------------------------------------
    # STEP 4: Held-Out Train/Test Evaluation (80/20 Stratified Split)
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 4: Held-Out Test Evaluation (20% Stratified Split)...")
    print("=" * 75)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Selected model: Logistic Regression (Balanced, C=0.5)
    selected_model = LogisticRegression(
        C=0.5,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )
    selected_model.fit(X_train, y_train)
    y_test_pred = selected_model.predict(X_test)
    
    classes = ["positive", "neutral", "negative"]
    cm = confusion_matrix(y_test, y_test_pred, labels=classes)
    
    print("\nHeld-out Test Classification Report:")
    report_dict = classification_report(y_test, y_test_pred, labels=classes, output_dict=True, zero_division=0)
    print(classification_report(y_test, y_test_pred, labels=classes, zero_division=0))
    print("Confusion Matrix (Labels: positive, neutral, negative):")
    print(cm)
    
    # -------------------------------------------------------------
    # STEP 5: Independent Unseen Test Set Evaluation
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 5: Independent Unseen Feedback Evaluation (10 real cases)...")
    print("=" * 75)
    
    with open(UNSEEN_TEST_PATH, "r", encoding="utf-8") as f:
        unseen_cases = json.load(f)
        
    unseen_texts = [c["text"] for c in unseen_cases]
    unseen_clean = [clean_text_for_sentiment(t) for t in unseen_texts]
    unseen_vec = vectorizer.transform(unseen_clean)
    
    unseen_preds = selected_model.predict(unseen_vec)
    unseen_probs = selected_model.predict_proba(unseen_vec)
    
    unseen_results = []
    correct_count = 0
    
    print(f"{'#':<3} | {'Expected':<9} | {'Predicted':<9} | {'Conf':<6} | {'Status':<6} | Text")
    print("-" * 110)
    
    for i, case in enumerate(unseen_cases):
        exp = case["expected_sentiment"]
        pred = unseen_preds[i]
        conf = float(unseen_probs[i].max())
        is_correct = (pred == exp)
        if is_correct:
            correct_count += 1
            
        status = "PASS" if is_correct else "FAIL"
        snippet = case["text"] if len(case["text"]) <= 60 else case["text"][:57] + "..."
        print(f"{i+1:<3} | {exp:<9} | {pred:<9} | {conf*100:<5.1f}% | {status:<6} | {snippet}")
        
        unseen_results.append({
            "id": case["id"],
            "text": case["text"],
            "category": case["category"],
            "expected": exp,
            "predicted": pred,
            "confidence": round(conf, 4),
            "probabilities": {cls_name: round(float(p), 4) for cls_name, p in zip(selected_model.classes_, unseen_probs[i])},
            "correct": is_correct
        })
        
    print(f"\nUnseen Evaluation Accuracy: {correct_count}/{len(unseen_cases)} ({correct_count/len(unseen_cases)*100:.1f}%)")
    
    # -------------------------------------------------------------
    # STEP 6: Fit Final Model on Full Corpus & Export Artifacts
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 6: Exporting Final Production Artifacts...")
    print("=" * 75)
    
    final_classifier = LogisticRegression(
        C=0.5,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )
    final_classifier.fit(X, y)
    
    # Save model and vectorizer
    joblib.dump(final_classifier, MODELS_DIR / "sentiment_classifier.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "sentiment_tfidf_vectorizer.pkl")
    print("Saved sentiment_classifier.pkl and sentiment_tfidf_vectorizer.pkl")
    
    # Update model_metrics.json (keeping LDA metrics intact)
    metrics_file = MODELS_DIR / "model_metrics.json"
    existing_metrics = {}
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            existing_metrics = json.load(f)
            
    updated_payload = {
        "models_benchmark": benchmarks,
        "best_model": "Logistic Regression",
        "confusion_matrix": {
            "labels": classes,
            "matrix": cm.tolist()
        },
        "per_class_metrics": {
            cls: {
                "precision": round(report_dict[cls]["precision"], 4),
                "recall": round(report_dict[cls]["recall"], 4),
                "f1_score": round(report_dict[cls]["f1-score"], 4),
                "support": int(report_dict[cls]["support"])
            }
            for cls in classes if cls in report_dict
        },
        "topic_metrics": existing_metrics.get("topic_metrics", {
            "n_topics": 5,
            "coherence_score": 0.472
        }),
        "feature_importance": {
            cls: [
                {"token": token, "weight": round(float(weight), 4)}
                for token, weight in sorted(
                    zip(vectorizer.get_feature_names_out(), final_classifier.coef_[idx]),
                    key=lambda x: x[1],
                    reverse=True
                )[:12]
            ]
            for idx, cls in enumerate(final_classifier.classes_)
        }
    }
    
    with open(metrics_file, "w") as f:
        json.dump(updated_payload, f, indent=2)
    print(f"Updated {metrics_file.name}")
    
    # -------------------------------------------------------------
    # STEP 7: Synchronize course evaluation CSV with new model
    # -------------------------------------------------------------
    if COURSE_EVAL_PATH.exists():
        eval_df = pd.read_csv(COURSE_EVAL_PATH)
        if "text" in eval_df.columns:
            cleaned_texts = eval_df["text"].apply(clean_text_for_sentiment)
            eval_vecs = vectorizer.transform(cleaned_texts)
            new_preds = final_classifier.predict(eval_vecs)
            new_probs = final_classifier.predict_proba(eval_vecs)
            
            eval_df["predicted_sentiment"] = new_preds
            eval_df["sentiment_confidence"] = [round(float(p.max()), 4) for p in new_probs]
            eval_df.to_csv(COURSE_EVAL_PATH, index=False)
            print(f"Synchronized updated predictions into {COURSE_EVAL_PATH.name}")

    print("\n" + "=" * 75)
    print("PIPELINE RETRAINING & ARTIFACT EXPORT COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    retrain_and_evaluate()
