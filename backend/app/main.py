import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure app package is in path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.api.routes import router as api_router

app = FastAPI(
    title="Student Feedback Intelligence System API",
    description="Full-stack NLP Backend for Student Feedback Sentiment Analysis & LDA Topic Modeling",
    version="1.0.0"
)

# Enable CORS for frontend clients (development + production domains)
origins = [
    "https://student-feedback-analyzer-chi.vercel.app",
    "https://student-feedback-analyzer-api.onrender.com",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:4173",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:4173",
]

# Allow custom origins via environment variable if specified
env_origins = os.getenv("ALLOWED_ORIGINS", "")
if env_origins:
    for orig in env_origins.split(","):
        clean_orig = orig.strip()
        if clean_orig and clean_orig not in origins:
            origins.append(clean_orig)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all API endpoints under /api
app.include_router(api_router, prefix="/api")


@app.get("/")
def root():
    return {
        "message": "Student Feedback Intelligence System API is online",
        "docs_url": "/docs",
        "endpoints": {
            "health": "/api/health",
            "analyze": "/api/analyze",
            "batch_analyze": "/api/analyze/batch",
            "dashboard": "/api/dashboard",
            "topics": "/api/topics",
            "sentiment": "/api/sentiment",
            "models": "/api/models"
        }
    }
