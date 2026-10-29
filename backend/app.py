from __future__ import annotations

from io import BytesIO
from typing import Any, Callable
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.groq_planner import GroqModelPlanner
from agent.pipeline import VerifiedAnalyticsPipeline
from backend.config import AppSettings, get_settings


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

    return {
        "row_count": row_count,
        "column_count": int(len(frame.columns)),
        "columns": columns,
    }


def _brief_payload(brief: Any) -> Any:
    return jsonable_encoder(brief)


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
    app.state.planner_factory = planner_factory or GroqModelPlanner

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

        raw = await file.read()
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
                detail=f"Could not parse CSV dataset: {exc}",
            ) from exc

        if frame.empty or len(frame.columns) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dataset must contain at least one row and one column.",
            )

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

    @app.post("/analyses", status_code=status.HTTP_201_CREATED)
    def create_analysis(request: AnalysisRequest) -> dict[str, Any]:
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

        planner = app.state.planner_factory()
        pipeline = VerifiedAnalyticsPipeline(planner)
        result = pipeline.run(
            question,
            dataset=dataset["frame"],
            dataset_context=dataset["profile"],
        )

        analysis_id = uuid4().hex
        payload: dict[str, Any] = {
            "analysis_id": analysis_id,
            "dataset_id": request.dataset_id,
            "question": question,
            "succeeded": result.succeeded,
            "exhausted": result.exhausted,
            "steps": [step.as_dict() for step in result.steps],
            "final_status": (
                result.final_trace.verification_status.value
                if result.final_trace is not None
                else None
            ),
            "decision_brief": None,
            "visualizations": [],
        }

        if result.succeeded:
            brief = result.build_decision_brief()
            visuals = result.build_visualizations()
            payload["decision_brief"] = _brief_payload(brief)
            payload["visualizations"] = [spec.as_dict() for spec in visuals]

        encoded = jsonable_encoder(payload)
        app.state.analyses[analysis_id] = encoded
        return encoded

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
