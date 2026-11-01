from __future__ import annotations

from io import BytesIO
from typing import Any, Callable
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.meta_planner import MetaModelConfigurationError, MetaModelError, MetaModelPlanner
from agent.pipeline import PipelineError
from fastapi.responses import StreamingResponse
from backend.analysis import _brief_payload, run_analysis
from backend.streaming import analysis_stream
from backend.config import AppSettings, get_settings
from backend.data_preparation import DatasetPreparationError, prepare_dataset


class AnalysisRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    question: str = Field(min_length=1)


def _safe_dataset_profile(frame: pd.DataFrame) -> dict[str, Any]:
    columns: list[dict[str, Any]] = []
    row_count = int(len(frame))

    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series):
            role = "measure"
        elif pd.api.types.is_datetime64_any_dtype(series):
            role = "time"
        else:
            role = "dimension"

        missing_count = int(series.isna().sum())
        columns.append(
            {
                "name": str(name),
                "dtype": str(series.dtype),
                "role": role,
                "missing_count": missing_count,
                "missing_rate": (missing_count / row_count) if row_count else 0.0,
                "unique_count": int(series.nunique(dropna=True)),
            }
        )
        if role == "time" and series.notna().any():
            columns[-1]["min_timestamp"] = series.min().isoformat()
            columns[-1]["max_timestamp"] = series.max().isoformat()

    return {
        "row_count": row_count,
        "column_count": int(len(frame.columns)),
        "columns": columns,
    }


def create_app(
    planner_factory: Callable[[], Any] | None = None,
    settings: AppSettings | None = None,
) -> FastAPI:
    config = settings or get_settings()
    app = FastAPI(
        title=config.app_name,
        version=config.app_version,
        description="Verifiable Analytics Agent API",
    )

    if config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.cors_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.state.settings = config
    app.state.datasets = {}
    app.state.analyses = {}
    app.state.planner_factory = planner_factory or MetaModelPlanner

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "vera-api"}

    @app.post("/datasets", status_code=status.HTTP_201_CREATED)
    async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
        filename = file.filename or "dataset.csv"
        if not filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only CSV datasets are supported in VERA API v1.",
            )

        raw = await file.read(config.max_upload_bytes + 1)
        if len(raw) > config.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Uploaded dataset exceeds the configured maximum size of "
                    f"{config.max_upload_bytes} bytes."
                ),
            )
        if not raw:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded dataset is empty.",
            )

        try:
            frame = pd.read_csv(BytesIO(raw))
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not parse CSV dataset. Check its encoding and CSV structure.",
            ) from exc

        if frame.empty or len(frame.columns) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dataset must contain at least one row and one column.",
            )

        try:
            prepared = prepare_dataset(frame)
        except DatasetPreparationError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        frame = prepared.frame
        dataset_id = uuid4().hex
        profile = _safe_dataset_profile(frame)
        app.state.datasets[dataset_id] = {
            "filename": filename,
            "frame": frame,
            "profile": profile,
        }

        return {
            "dataset_id": dataset_id,
            "filename": filename,
            "profile": profile,
        }

    def analysis_input(request: AnalysisRequest) -> tuple[dict[str, Any], str]:
        dataset = app.state.datasets.get(request.dataset_id)
        if dataset is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found.",
            )

        question = request.question.strip()
        if not question:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Question must not be blank.",
            )

        return dataset, question

    @app.post("/analyses", status_code=status.HTTP_201_CREATED)
    def create_analysis(request: AnalysisRequest) -> dict[str, Any]:
        dataset, question = analysis_input(request)
        try:
            return run_analysis(
                analysis_id=uuid4().hex, dataset_id=request.dataset_id, question=question,
                dataset=dataset, planner_factory=app.state.planner_factory, analyses=app.state.analyses,
            )
        except MetaModelConfigurationError as exc:
            raise HTTPException(status_code=503, detail="Planning provider is not configured.") from exc
        except (MetaModelError, PipelineError) as exc:
            raise HTTPException(status_code=502, detail="Planning provider could not produce a valid analysis plan. Please retry.") from exc

    @app.post("/analyses/stream")
    async def stream_analysis(request: AnalysisRequest) -> StreamingResponse:
        dataset, question = analysis_input(request)
        return StreamingResponse(
            analysis_stream(analysis_id=uuid4().hex, dataset_id=request.dataset_id, question=question,
                            dataset=dataset, planner_factory=app.state.planner_factory,
                            analyses=app.state.analyses),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store, no-transform", "X-Accel-Buffering": "no"},
        )

    @app.get("/analyses/{analysis_id}")
    def get_analysis(analysis_id: str) -> dict[str, Any]:
        analysis = app.state.analyses.get(analysis_id)
        if analysis is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found.",
            )
        return analysis

    return app


app = create_app()
