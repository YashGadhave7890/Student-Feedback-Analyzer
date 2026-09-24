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
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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
