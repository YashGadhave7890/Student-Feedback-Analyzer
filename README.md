# Student Feedback Intelligence System

An NLP-powered full-stack web application designed for analyzing qualitative student course evaluations through unsupervised Latent Dirichlet Allocation (LDA) topic modeling and supervised multi-class sentiment classification.

---

## 1. System Architecture

```
[React + TypeScript + Vite + Tailwind CSS Frontend]
                        │
                        │ REST API (JSON / Multipart CSV)
                        ▼
         [FastAPI Python 3.11 Backend]
    ├── Preprocessor (WordNet Lemmatization + Stopword Cleaner)
    ├── TopicEngine (Unsupervised LDA $K=5$ + CountVectorizer)
    ├── SentimentEngine (TF-IDF N-Grams + Calibrated Classifier)
    └── ExplainabilityEngine (Token Attributions & Feature Weights)
                        │
                        ▼
            [In-Memory / Model Store]
    ├── Serialized Models (`backend/trained_models/*.pkl`)
    ├── Metrics & Configuration (`model_metrics.json`, `topic_labels.json`)
    └── In-Memory Streaming CSV Processing (NO DATABASE)
```

---

## 2. Core Features

1. **Executive Dashboard**: Key performance statistics, sentiment distribution pie chart, topic proportions, and a stacked Topic-Sentiment cross-tabulation matrix.
2. **Analyze Feedback**: Real-time live inference returning sentiment confidence, dominant LDA theme, top topic keywords, and token-level sentiment weight badges.
3. **Bulk CSV Analysis**: In-memory batch processing of course evaluation CSV files, preview table with filter/search, and streaming download of the analyzed dataset.
4. **Topic Explorer**: Interactive inspection of discovered course themes, normalized term weights, and representative student quotes.
5. **Sentiment Explorer**: Class distribution analytics and top positive vs. negative NLP feature predictor charts.
6. **Model Performance**: Cross-validation benchmark comparison table (**Logistic Regression**, **Multinomial Naive Bayes**, **Linear SVM**, **Random Forest**), test confusion matrix heatmap, and LDA perplexity metrics.

---

## 3. Local Development Setup

### Prerequisites
- **Python 3.10+** (Python 3.11 recommended)
- **Node.js 18+** & **npm**

### Step 1: Start the FastAPI Backend
```bash
# From the project root:
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
*Backend runs on `http://127.0.0.1:8000` (Interactive Swagger docs at `http://127.0.0.1:8000/docs`).*

### Step 2: Start the React Frontend
```bash
# In a separate terminal:
cd frontend
npm install
npm run dev
```
*Frontend opens at `http://localhost:5173`.*

### Step 3: Run Backend Test Suite
```bash
# From the project root:
python -m pytest backend/tests/
```

---

## 4. Production Deployment Guide

### A. Deploy Backend on Render (Web Service)

1. Connect your GitHub repository to [Render](https://render.com).
2. Create a new **Web Service** with the following configuration:
   - **Root Directory**: `backend`
   - **Environment**: `Python 3`
   - **Build Command**:
     ```bash
     pip install -r requirements.txt && python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
     ```
   - **Start Command**:
     ```bash
     uvicorn app.main:app --host 0.0.0.0 --port $PORT
     ```
3. Copy the assigned Render service URL (e.g., `https://student-feedback-backend.onrender.com`).

---

### B. Deploy Frontend on Vercel or Render

#### Option 1: Vercel (Recommended)
1. Import the repository into [Vercel](https://vercel.com).
2. Set the **Root Directory** to `frontend`.
3. Set the **Framework Preset** to `Vite`.
4. Add the Environment Variable:
   - **Key**: `VITE_API_BASE_URL`
   - **Value**: `https://your-render-backend-url.onrender.com/api`
5. Click **Deploy**.

#### Option 2: Render (Static Site)
1. Create a new **Static Site** on Render.
2. Set **Root Directory** to `frontend`.
3. Set **Build Command** to `npm install && npm run build`.
4. Set **Publish Directory** to `dist`.
5. Add the Environment Variable:
   - `VITE_API_BASE_URL` = `https://your-render-backend-url.onrender.com/api`
6. Add a rewrite rule in Render redirects: `/*` -> `/index.html` (Rewrite).

---

## 5. REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health status & model loading verification |
| `POST` | `/api/analyze` | Single feedback sentiment & LDA topic inference |
| `POST` | `/api/analyze/batch` | Batch feedback inference via JSON array or CSV upload |
| `GET` | `/api/analyze/export/{token}` | Download enriched CSV with predicted columns |
| `GET` | `/api/dashboard` | Aggregated dataset summary and cross-tabulation matrix |
| `GET` | `/api/topics` | Discovered topics, term weights, and student quotes |
| `GET` | `/api/sentiment` | Sentiment class distribution and top lexical predictors |
| `GET` | `/api/models` | Classifier benchmarks (LogReg, NB, SVM, RF) and confusion matrix |

---

## 6. Technology Stack

- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, React Router v6, Recharts, Lucide Icons, Axios
- **Backend**: Python 3.11, FastAPI, Pydantic v2, Uvicorn
- **NLP / ML**: scikit-learn, NLTK, Latent Dirichlet Allocation (LDA), CountVectorizer, TF-IDF Vectorizer
- **Storage / State**: In-memory file processing, serialized pickle models (NO DATABASE)
