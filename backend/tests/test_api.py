import io
import pytest
from fastapi.testclient import TestClient
import pandas as pd

from backend.app.main import app

client = TestClient(app)


def test_get_health():
    """Test GET /api/health"""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True
    assert "service" in data


def test_post_analyze_single():
    """Test POST /api/analyze with single feedback text"""
    payload = {"text": "The practical lab sessions with Docker were very engaging and clear."}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["positive", "neutral", "negative"]
    assert "sentiment_confidence" in data
    assert "sentiment_probabilities" in data
    assert "dominant_topic" in data
    assert "topic_label" in data
    assert "top_topic_keywords" in data
    assert len(data["top_topic_keywords"]) > 0
    assert "token_attributions" in data


def test_post_analyze_empty_and_invalid_text():
    """Test POST /api/analyze with invalid or whitespace text"""
    response = client.post("/api/analyze", json={"text": "   "})
    assert response.status_code == 400

    response2 = client.post("/api/analyze", json={"text": ""})
    assert response2.status_code == 422 or response2.status_code == 400


def test_post_analyze_special_characters():
    """Test POST /api/analyze with symbols and numbers"""
    payload = {"text": "100% loved the labs! @Docker & #PowerBI were 10/10 :)"}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["positive", "neutral", "negative"]


def test_post_analyze_batch_json():
    """Test POST /api/analyze/batch with JSON list of texts"""
    payload = {
        "texts": [
            "Great lectures and helpful practical exercises.",
            "Slides were too long and confusing during revision.",
            "Normal pace and standard content."
        ]
    }
    response = client.post("/api/analyze/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 3
    assert "sentiment_breakdown" in data
    assert "topic_breakdown" in data
    assert len(data["results"]) == 3
    assert "download_token" in data


def test_post_analyze_batch_csv_and_export():
    """Test POST /api/analyze/batch with CSV file upload and export download"""
    csv_content = b"student_id,feedback_text\n1,Very good teacher and great course\n2,Unclear assignments and long slides\n"
    files = {"file": ("test.csv", io.BytesIO(csv_content), "text/csv")}
    response = client.post("/api/analyze/batch", files=files, data={"text_column": "feedback_text"})
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert "download_token" in data

    # Test downloading analyzed CSV export
    token = data["download_token"]
    export_res = client.get(f"/api/analyze/export/{token}")
    assert export_res.status_code == 200
    assert "text/csv" in export_res.headers["content-type"]
    assert b"predicted_sentiment" in export_res.content


def test_post_analyze_batch_invalid_file():
    """Test POST /api/analyze/bulk with non-CSV file"""
    files = {"file": ("test.txt", io.BytesIO(b"hello world"), "text/plain")}
    response = client.post("/api/analyze/bulk", files=files)
    assert response.status_code == 400


def test_post_analyze_batch_csv_autodetect_and_validation():
    """Test CSV auto-detection on student evaluation dataset and rejection of invalid columns"""
    with open("data/course_evals_with_topics_and_sentiments.csv", "rb") as f:
        content = f.read()

    # 1. Test auto-detection (no text_column passed)
    files = {"file": ("evals.csv", io.BytesIO(content), "text/csv")}
    response = client.post("/api/analyze/batch", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 129
    assert data["sentiment_breakdown"]["positive"] > 0
    assert data["sentiment_breakdown"]["negative"] > 0

    # 2. Test rejection when user manually selects timestamp column
    files_bad = {"file": ("evals.csv", io.BytesIO(content), "text/csv")}
    response_bad = client.post("/api/analyze/batch", files=files_bad, data={"text_column": "timestamp"})
    assert response_bad.status_code == 400
    assert "Please select a feedback/text column containing natural-language responses." in response_bad.json()["detail"]


def test_get_dashboard():
    """Test GET /api/dashboard and real stats"""
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert data["total_feedbacks"] == 129
    assert data["sentiment_distribution"]["positive"] > 0
    assert data["sentiment_distribution"]["neutral"] > 0
    assert data["sentiment_distribution"]["negative"] > 0
    assert len(data["topic_sentiment_matrix"]) > 0
    assert "top_positive_insight" in data["key_metrics"]
    assert len(data["key_metrics"]["top_positive_insight"]) > 0
    assert "top_improvement_insight" in data["key_metrics"]
    assert len(data["key_metrics"]["top_improvement_insight"]) > 0


def test_get_topics():
    """Test GET /api/topics"""
    response = client.get("/api/topics")
    assert response.status_code == 200
    data = response.json()
    assert len(data["topics"]) == 5
    for t in data["topics"]:
        assert len(t["top_words"]) == 10
        assert len(t["word_weights"]) == 10


def test_get_sentiment():
    """Test GET /api/sentiment"""
    response = client.get("/api/sentiment")
    assert response.status_code == 200
    data = response.json()
    assert data["total_analyzed"] == 129
    assert len(data["top_features"]["positive"]) > 0
    assert len(data["top_features"]["negative"]) > 0


def test_get_models():
    """Test GET /api/models"""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models_benchmark" in data
    assert "Logistic Regression" in data["models_benchmark"]
    assert "Multinomial Naive Bayes" in data["models_benchmark"]
    assert "Linear SVM" in data["models_benchmark"]
    assert "Random Forest" in data["models_benchmark"]
    assert data["best_model"] == "Logistic Regression"
    assert len(data["confusion_matrix"]["matrix"]) == 3


def test_cors_configuration():
    """Test CORS preflight and headers for production Vercel frontend and localhost"""
    # 1. Production Vercel Origin
    res_vercel = client.options(
        "/api/health",
        headers={
            "Origin": "https://student-feedback-analyzer-chi.vercel.app",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert res_vercel.status_code == 200
    assert (
        res_vercel.headers.get("access-control-allow-origin")
        == "https://student-feedback-analyzer-chi.vercel.app"
    )
    assert res_vercel.headers.get("access-control-allow-credentials") == "true"

    # 2. Localhost Origin
    res_local = client.get(
        "/api/health",
        headers={"Origin": "http://localhost:5173"},
    )
    assert res_local.status_code == 200
    assert res_local.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert res_local.headers.get("access-control-allow-credentials") == "true"
