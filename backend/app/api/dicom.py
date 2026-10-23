from io import BytesIO

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.config import MAX_UPLOAD_SIZE_MB
from app.services.dicom_service import convert_dicom_to_png


router = APIRouter(prefix="/api/dicom", tags=["DICOM"])

MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024


@router.post("/convert")
async def convert_dicom(file: UploadFile = File(...)):
    """Convert an uploaded DICOM file into PNG format."""

    try:
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

        png_bytes = convert_dicom_to_png(file.filename, content)

        return StreamingResponse(
            BytesIO(png_bytes),
            media_type="image/png",
            headers={
                "Content-Disposition": 'inline; filename="converted.png"'
            },
        )

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="DICOM conversion failed. Check the backend logs.",
        ) from exc
    finally:
        await file.close()
