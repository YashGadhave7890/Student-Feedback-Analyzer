# API Documentation - Student Feedback Intelligence System

The backend is built with **FastAPI** and uses **in-memory data processing, CSV files, and serialized scikit-learn models** (No database required).

## Base URL
- Local: `http://127.0.0.1:8000/api`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`

---

## Endpoints Summary

### 1. Health Check
- **Endpoint**: `GET /api/health`
- **Description**: Returns server and NLP engine status.
- **Response**:
```json
{
  "status": "healthy",
  "service": "student-feedback-intelligence-backend",
  "version": "1.0.0"
}
```

### 2. Dashboard Summary
- **Endpoint**: `GET /api/dashboard/summary`
- **Description**: Aggregated feedback analytics, sentiment percentages, topic counts, and topic-sentiment cross-tabulation.

### 3. Live Single Feedback Inference
- **Endpoint**: `POST /api/analyze/single`
- **Request Body**:
```json
{
  "text": "The practical lab sessions with Docker and Power BI were very helpful and well explained."
}
```
- **Response**: Returns predicted sentiment, confidence, sentiment class probabilities, dominant LDA topic, topic confidence, top 10 keywords, and token attribution weights.

### 4. Bulk CSV Feedback Analysis
- **Endpoint**: `POST /api/analyze/bulk`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: CSV file containing feedback text
  - `text_column` (optional): Name of the column containing feedback
- **Response**: Returns batch predictions summary, breakdown counts, sample items, and a unique `download_token`.

### 5. Export Analyzed CSV
- **Endpoint**: `GET /api/analyze/export/{token}`
- **Description**: Downloads the complete CSV with appended prediction columns (`predicted_sentiment`, `sentiment_confidence`, `dominant_topic`, `topic_label`, `topic_confidence`).

### 6. Discovered Topics & Word Weights
- **Endpoint**: `GET /api/topics`
- **Description**: Lists all 5 LDA topics with normalized term weights, sentiment ratios, and representative student quotes.

### 7. Model Performance & Benchmarks
- **Endpoint**: `GET /api/performance`
- **Description**: Returns 5-fold cross-validation benchmarks for Logistic Regression, Naive Bayes, Linear SVM, and Random Forest, plus the test-set Confusion Matrix and LDA Perplexity score.
