"""FastAPI HTTP surface for image-analyser."""

from __future__ import annotations

import os
from importlib.metadata import version as _pkg_version
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from .exceptions import UnsupportedFormatError
from .image_analyser import ImageAnalyser
from .schemas import AnalysisResult


def _origins() -> list[str]:
    raw = os.getenv("IMAGE_ANALYSER_ALLOWED_ORIGINS", "*")
    return [o.strip() for o in raw.split(",") if o.strip()]


def _max_upload_bytes() -> int:
    return int(os.getenv("IMAGE_ANALYSER_MAX_UPLOAD_MB", "50")) * 1024 * 1024


class AnalyseRequest(BaseModel):
    path: str | None = None
    skip: list[str] | None = None
    only: list[str] | None = None
    caption_backend: str | None = None


def _parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _build_analyser(
    skip: list[str] | None,
    only: list[str] | None,
    caption_backend: str | None,
) -> ImageAnalyser:
    try:
        return ImageAnalyser(skip=skip, only=only, caption_backend=caption_backend)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def create_app() -> FastAPI:
    app = FastAPI(
        title="image-analyser",
        version=_pkg_version("image-analyser"),
        description="Static image analysis (CLI + FastAPI) for the analyser family",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins(),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": _pkg_version("image-analyser")}

    @app.get("/")
    def root() -> dict[str, object]:
        return {
            "service": "image-analyser",
            "version": _pkg_version("image-analyser"),
            "endpoints": ["/analyse", "/health"],
        }

    @app.post("/analyse", response_model=AnalysisResult)
    async def analyse(request: Request) -> AnalysisResult:
        ctype = request.headers.get("content-type", "").lower()
        if ctype.startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("file")
            if upload is None or not hasattr(upload, "read"):
                raise HTTPException(status_code=400, detail="multipart requires a `file` field")
            data = await upload.read()
            if len(data) > _max_upload_bytes():
                raise HTTPException(status_code=413, detail="upload too large")
            analyser = _build_analyser(
                skip=_parse_csv(form.get("skip")),  # type: ignore[arg-type]
                only=_parse_csv(form.get("only")),  # type: ignore[arg-type]
                caption_backend=form.get("caption_backend") or None,  # type: ignore[arg-type]
            )
            try:
                return analyser.analyse(data)
            except UnsupportedFormatError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
        if ctype.startswith("application/json"):
            try:
                body = await request.json()
                req = AnalyseRequest.model_validate(body)
            except (ValueError, ValidationError) as e:
                raise HTTPException(status_code=400, detail=f"invalid JSON body: {e}") from e
            if not req.path:
                raise HTTPException(status_code=400, detail="`path` is required for JSON requests")
            p = Path(req.path)
            if not p.exists():
                raise HTTPException(status_code=404, detail=f"path not found: {p}")
            analyser = _build_analyser(req.skip, req.only, req.caption_backend)
            try:
                return analyser.analyse(p)
            except UnsupportedFormatError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
        raise HTTPException(
            status_code=400,
            detail="use multipart/form-data with a `file` field, or application/json with a `path`",
        )

    return app


app = create_app()
