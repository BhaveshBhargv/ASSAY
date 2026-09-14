"""POST /upload"""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile

from ..core.config import settings
from ..core.errors import ParsingError, PayloadTooLargeError
from ..dependencies import get_upload_service
from ..schemas.common import ErrorResponse
from ..schemas.upload import UploadResponse
from ..services.upload_service import UploadService

router = APIRouter(tags=["upload"])

_ALLOWED_EXT = (".csv", ".json", ".pdf")


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Parse a blood report file",
    description=(
        "Upload a blood report as **CSV**, **JSON**, or **PDF**. CSV/JSON are parsed "
        "reliably; PDF uses best-effort text extraction. Returns the recognised "
        "biomarker values plus parse notes for you to **review and correct** before "
        "calling /predict or /recommend. No pipeline is run here."
    ),
    responses={
        413: {"model": ErrorResponse, "description": "File too large"},
        422: {"model": ErrorResponse, "description": "Unsupported or unreadable file"},
    },
)
async def upload(
    file: UploadFile = File(..., description="Blood report: .csv, .json, or .pdf"),
    service: UploadService = Depends(get_upload_service),
) -> UploadResponse:
    filename = file.filename or "upload"
    if not filename.lower().endswith(_ALLOWED_EXT):
        raise ParsingError(f"Unsupported file type: {filename}. Use CSV, JSON, or PDF.",
                           detail={"allowed": list(_ALLOWED_EXT)})
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise PayloadTooLargeError(
            f"File exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB limit.",
            detail={"size": len(data), "limit": settings.max_upload_bytes})
    return service.parse(filename, data)
