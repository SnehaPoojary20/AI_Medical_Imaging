from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import BACKEND_DIR, MAX_UPLOAD_SIZE_MB
from app.schemas import AnalysisResponse
from app.services.analysis_service import analyze_image


router = APIRouter(prefix="/api/analysis", tags=["Analysis"])

OUTPUT_DIR = Path(BACKEND_DIR) / "data" / "outputs"
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024


@router.post("/", response_model=AnalysisResponse)
async def create_analysis(file: UploadFile = File(...)):
    """Upload an image and request an AI analysis."""

    try:
        # Read at most one byte beyond the configured limit.
        content = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)

        if len(content) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds {MAX_UPLOAD_SIZE_MB} MB.",
            )

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="A filename is required.",
            )

        result = analyze_image(file.filename, content)
        return AnalysisResponse(**result)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="The trained model is not available. Configure its weights first.",
        ) from exc
    except Exception as exc:
        # Keep internal exception details out of the client response.
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Check the backend logs.",
        ) from exc
    finally:
        await file.close()


@router.get("/heatmaps/{filename}", response_class=FileResponse)
def get_heatmap(filename: str):
    """Return a previously generated heatmap image."""

    # Only accept a plain PNG filename, not a path.
    if (
        not filename.endswith(".png")
        or Path(filename).name != filename
    ):
        raise HTTPException(status_code=400, detail="Invalid heatmap filename.")

    heatmap_path = OUTPUT_DIR / filename

    if not heatmap_path.is_file():
        raise HTTPException(status_code=404, detail="Heatmap not found.")

    return FileResponse(
        path=heatmap_path,
        media_type="image/png",
        filename=filename,
    )


