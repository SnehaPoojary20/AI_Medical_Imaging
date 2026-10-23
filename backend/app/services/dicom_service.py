from io import BytesIO
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

from app.config import ALLOWED_DICOM_EXTENSIONS


def convert_dicom_to_png(filename: str, content: bytes) -> bytes:
    """Convert DICOM pixel data to PNG bytes."""

    # 1. Validate the filename extension.
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_DICOM_EXTENSIONS:
        raise ValueError(
            "Unsupported DICOM extension. Use .dcm or .dicom."
        )

    # 2. Reject empty input.
    if not content:
        raise ValueError("The uploaded DICOM file is empty.")

    # 3. Read the DICOM dataset from memory.
    try:
        dataset = pydicom.dcmread(
            BytesIO(content),
            force=False,
        )

        if not hasattr(dataset, "PixelData"):
            raise ValueError("The DICOM file contains no pixel data.")

        pixels = dataset.pixel_array.astype(np.float32)

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "Unable to read this DICOM file or decode its pixel data."
        ) from exc

    # 4. Apply rescale information when present.
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    pixels = pixels * slope + intercept

    # 5. Normalize pixel values to the 8-bit range.
    minimum = float(np.min(pixels))
    maximum = float(np.max(pixels))

    if maximum > minimum:
        pixels = (pixels - minimum) / (maximum - minimum) * 255.0
    else:
        pixels = np.zeros_like(pixels)

    pixels = np.clip(pixels, 0, 255).astype(np.uint8)

    # 6. Convert grayscale pixels to a PNG file in memory.
    image = Image.fromarray(pixels)

    output = BytesIO()
    image.save(output, format="PNG")

    return output.getvalue()
