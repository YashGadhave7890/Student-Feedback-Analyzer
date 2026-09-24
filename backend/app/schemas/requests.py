from typing import Optional, List
from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Student feedback text to analyze")


class BatchFeedbackRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of student feedback texts to analyze")
