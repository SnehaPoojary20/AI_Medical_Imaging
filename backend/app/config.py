import os
from pathlib import Path

APP_DIR =  Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent

APP_NAME = "AI-Assisted Medical Imaging and Diagnosis System"
APP_VERSION = "1.0.0"

# Local development defaults
FRONTEND_ORIGIN = os.getenv(
    "FRONTEND_ORIGIN",
    "http://localhost:5173",
)

DATABASE_PATH = Path(
    os.getenv(
        "DATABASE_PATH",
        str(BACKEND_DIR / "medical_imaging.db"),
    )
)

MAX_UPLOAD_SIZE_MB = 10
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALLOWED_DICOM_EXTENSIONS = {".dcm", ".dicom"}