from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    MAX_UPLOAD_SIZE_MB,
)

MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024



def validate_image_extension(filename:str)->None:
    """Check whether the image extension is supported."""
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
        raise ValueError(
            f"Unsupported image extension. Allowed: {allowed}"
        )



def validate_image_size(content:bytes)->None:
    """Check that the uploaded file is non-empty and within the size limit."""
    if(not content):
        raise ValueError("The uploaded file is empty.")

    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise ValueError(
            f"Image exceeds the maximum size of "
            f"{MAX_UPLOAD_SIZE_MB} MB."
        )


    
def validate_image_content(content: bytes) -> None:
    """Verify that the file contains readable image data."""
    if not content:
        raise ValueError("The uploaded file is empty.")

    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError(
            "The uploaded file is not a valid or supported image."
        ) from exc



def validate_image_upload(
    filename: str,
    content: bytes,
) -> None:
    """Run all basic image-upload validation checks."""
    validate_image_extension(filename)
    validate_image_size(content)
    validate_image_content(content)



#File validation