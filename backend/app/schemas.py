from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# Describes the API's health-check response.
class HealthResponse(BaseModel):
    status:str
    app_name:str
    version:str

# Describes the AI model's prediction and associated information.
class AnalysisResponse (BaseModel):
    study_id:str
    model_name:str
    prediction:str
    confidence: float = Field(ge=0.0, le=1.0)
    heatmap_url: str | None = None
    disclaimer: str = (
        "For educational purposes only — not for clinical use."
    )

# Describes the AI model's prediction and associated information.
class ReviewCreate(BaseModel):
    study_id: str = Field(min_length=1, max_length=100)
    decision: Literal["approve", "reject"]
    notes: str | None = Field(default=None, max_length=1000)

# Describes the saved review returned by the backend.
class ReviewResponse(BaseModel):
    id: int
    study_id: str
    decision: Literal["approve", "reject"]
    notes: str | None
    created_at: datetime