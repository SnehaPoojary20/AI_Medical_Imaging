from fastapi import APIRouter, HTTPException

from app.schemas import ReviewCreate, ReviewResponse
from app.services.review_service import (
    fetch_review,
    fetch_study_reviews,
    submit_review,
)


router = APIRouter(prefix="/api/reviews", tags=["Reviews"])


@router.post("/", response_model=ReviewResponse, status_code=201)
def create_review_endpoint(payload: ReviewCreate):
    """Save a review for an analysis."""

    try:
        result = submit_review(
            study_id=payload.study_id,
            decision=payload.decision,
            notes=payload.notes,
        )
        return ReviewResponse(**result)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{review_id}", response_model=ReviewResponse)
def get_review_endpoint(review_id: int):
    """Retrieve one review by its ID."""

    try:
        result = fetch_review(review_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if result is None:
        raise HTTPException(status_code=404, detail="Review not found.")

    return ReviewResponse(**result)


@router.get("/study/{study_id}", response_model=list[ReviewResponse])
def get_study_reviews_endpoint(study_id: str):
    """Retrieve reviews associated with a study."""

    try:
        results = fetch_study_reviews(study_id)
        return [ReviewResponse(**item) for item in results]

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
