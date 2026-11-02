from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    APP_NAME,
    APP_VERSION,
    FRONTEND_ORIGIN,
)
from app.database import init_db

from app.routes.analysis import router as analysis_router
from app.routes.reviews import router as reviews_router
from app.routes.dicom import router as dicom_router
from app.routes.health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize application resources at startup."""
    init_db()
    yield


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "An educational prototype for medical image analysis "
        "and AI-assisted diagnosis. Not intended for clinical use."
    ),
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_ORIGIN,
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


app.include_router(analysis_router)
app.include_router(reviews_router)
app.include_router(dicom_router)
app.include_router(health_router)


@app.get("/", tags=["General"])
def root():
    """Return basic application information."""
    return {
        "message": APP_NAME,
        "version": APP_VERSION,
        "docs": "/docs",
    }


@app.get("/api/health", tags=["Health"])
def health_check():
    """Check whether the API is responding."""
    return {
        "status": "ok",
        "app_name": APP_NAME,
        "version": APP_VERSION,
    }
