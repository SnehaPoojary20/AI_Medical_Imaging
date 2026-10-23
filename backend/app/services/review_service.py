from app.database import (
    create_review,
    get_review,
    get_reviews_for_study,
)


def submit_review(
    study_id: str,
    decision: str,
    notes: str | None = None,
) -> dict:
    """Save an approve/reject review for an analysis."""

    study_id = study_id.strip()

    if not study_id:
        raise ValueError("Study ID cannot be empty.")

    if decision not in {"approve", "reject"}:
        raise ValueError("Decision must be 'approve' or 'reject'.")

    if notes is not None:
        notes = notes.strip() or None

    return create_review(
        study_id=study_id,
        decision=decision,
        notes=notes,
    )


def fetch_review(review_id: int) -> dict | None:
    """Get one review using its database ID."""
    if review_id <= 0:
        raise ValueError("Review ID must be positive.")

    return get_review(review_id)


def fetch_study_reviews(study_id: str) -> list[dict]:
    """Get all reviews associated with a study."""
    study_id = study_id.strip()

    if not study_id:
        raise ValueError("Study ID cannot be empty.")

    return get_reviews_for_study(study_id)
