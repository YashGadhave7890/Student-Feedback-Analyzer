# System Architecture & Setup Guide

## Architecture Overview

```
[React + TypeScript + Vite Frontend]
         │ (Tailwind CSS, Recharts, React Router)
         │
         │ REST API (JSON / FormData)
         ▼
[FastAPI Python Backend]
   ├── App Config & Routing
   ├── NLP Services:
   │    ├── Preprocessor (WordNet Lemmatization + Tokenizer + Domain Stopwords)
   │    ├── SentimentEngine (TF-IDF N-Grams + Calibrated Logistic Regression)
   │    ├── TopicEngine (Unsupervised LDA $K=5$ + CountVectorizer)
   │    ├── ExplainabilityEngine (Token Attributions & Feature Weights)
   │    └── DatasetValidator (CSV Validation & Anomaly Checks)
   └── Storage Layer (NO DATABASE):
        ├── Models (`backend/trained_models/*.pkl`, `*.json`)
        ├── Data (`data/*.csv`)
        └── In-Memory File Buffers
```

## Running the Application Locally

### 1. Start the FastAPI Backend
```bash
# In the project root:
pip install -r backend/requirements.txt
python backend/run.py
```
The API server will run on `http://127.0.0.1:8000` (Swagger docs available at `http://127.0.0.1:8000/docs`).

### 2. Start the React Frontend
```bash
cd frontend
npm install
npm run dev
```
The React frontend will run on `http://localhost:5173`.

---

## Retraining Models

To retrain the models and regenerate benchmark metrics, run:
```bash
python scripts/train_and_export_models.py
```
