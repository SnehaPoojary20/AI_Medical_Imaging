from fastapi import APIRouter

from app.ml.model_loader import MODEL_NAME, MODEL_PATH


router = APIRouter(prefix="/api/health", tags=["Health"])


@router.get("/model-status")
def model_status():
    """Report whether the configured model checkpoint file exists."""

    available = MODEL_PATH.is_file()

    return {
        "model_name": MODEL_NAME,
        "checkpoint_available": available,
        "status": "available" if available else "not_configured",
    }
