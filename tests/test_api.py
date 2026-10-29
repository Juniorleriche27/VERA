from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from agent.tool_call_trace import ToolCallTrace
from backend.config import AppSettings


class FakePlanner:
    def plan(self, question, *, dataset_context, tool_catalog):
        trace = ToolCallTrace.planned(
            call_id="api-anova-1",
            tool="one_way_anova",
            objective="Compare average sales across regions.",
            reason="Compare three independent group means.",
            parameters={"alpha": 0.05},
            bindings={"region": "region", "sales": "sales"},
        )
        return SimpleNamespace(traces=(trace,))


def _client(settings=None):
    from backend.app import create_app

    return TestClient(
        create_app(
            planner_factory=lambda: FakePlanner(),
            settings=settings,
        )
    )


def _upload(client):
    response = client.post(
        "/datasets",
        files={
            "file": (
                "sales.csv",
                "region,sales\nNorth,10\nNorth,12\nSouth,20\nSouth,22\nWest,30\nWest,32\n",
                "text/csv",
            )
        },
    )
    assert response.status_code == 201
    return response.json()


def test_health():
    client = _client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "vera-api"}


def test_upload_dataset_returns_safe_profile_without_raw_rows():
    client = _client()
    payload = _upload(client)

    assert payload["filename"] == "sales.csv"
    assert payload["profile"]["row_count"] == 6
    assert payload["profile"]["column_count"] == 2
    assert [item["name"] for item in payload["profile"]["columns"]] == [
        "region",
        "sales",
    ]

    body = client.post(
        "/datasets",
        files={"file": ("tiny.csv", "region,sales\nSecretRegion,999\n", "text/csv")},
    ).text
    assert "SecretRegion" not in body
    assert "999" not in body


def test_upload_rejects_non_csv():
    client = _client()
    response = client.post(
        "/datasets",
        files={"file": ("sales.txt", b"region,sales\nA,1\n", "text/plain")},
    )
    assert response.status_code == 415


def test_analysis_requires_existing_dataset():
    client = _client()
    response = client.post(
        "/analyses",
        json={"dataset_id": "missing", "question": "Compare sales"},
    )
    assert response.status_code == 404


def test_analysis_runs_verified_pipeline_and_returns_brief_visual_and_steps():
    client = _client()
    dataset = _upload(client)

    response = client.post(
        "/analyses",
        json={
            "dataset_id": dataset["dataset_id"],
            "question": "Are average sales different across regions?",
        },
    )
    assert response.status_code == 201

    payload = response.json()
    assert payload["succeeded"] is True
    assert payload["final_status"] == "verified"
    assert (
        payload["steps"][0]["report"]["trace_hash"]
        == payload["steps"][0]["trace"]["trace_hash"]
    )
    assert payload["steps"][0]["trace"]["tool"] == "one_way_anova"
    assert payload["decision_brief"]["headline"]
    assert len(payload["visualizations"]) == 1

    chart = payload["visualizations"][0]
    assert chart["chartType"] == "bar"
    assert chart["data"] == [
        {"group": "North", "mean": 11.0},
        {"group": "South", "mean": 21.0},
        {"group": "West", "mean": 31.0},
    ]
    assert chart["evidence"]["call_id"] == "api-anova-1"
    assert len(chart["evidence"]["trace_hash"]) == 64


def test_analysis_is_persisted_for_get():
    client = _client()
    dataset = _upload(client)

    created = client.post(
        "/analyses",
        json={
            "dataset_id": dataset["dataset_id"],
            "question": "Are average sales different across regions?",
        },
    ).json()

    response = client.get(f"/analyses/{created['analysis_id']}")
    assert response.status_code == 200
    assert response.json() == created


def test_blank_question_is_rejected():
    client = _client()
    dataset = _upload(client)

    response = client.post(
        "/analyses",
        json={"dataset_id": dataset["dataset_id"], "question": "   "},
    )
    assert response.status_code == 422


def test_cors_allows_configured_frontend_origin():
    client = _client(
        AppSettings(cors_origins=("https://vera.example",))
    )
    response = client.options(
        "/datasets",
        headers={
            "Origin": "https://vera.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://vera.example"


def test_upload_rejects_dataset_larger_than_configured_limit():
    client = _client(
        AppSettings(max_upload_bytes=5)
    )
    response = client.post(
        "/datasets",
        files={"file": ("tiny.csv", b"a,b\n1,2\n", "text/csv")},
    )
    assert response.status_code == 413
    assert "maximum size" in response.json()["detail"]


def test_settings_load_environment_values(monkeypatch):
    monkeypatch.setenv("VERA_APP_NAME", "VERA Production")
    monkeypatch.setenv("VERA_APP_VERSION", "1.2.3")
    monkeypatch.setenv("VERA_ENV", "production")
    monkeypatch.setenv(
        "VERA_CORS_ORIGINS",
        "https://app.vera.example,https://admin.vera.example",
    )
    monkeypatch.setenv("VERA_MAX_UPLOAD_BYTES", "12345")

    settings = AppSettings.from_env()

    assert settings.app_name == "VERA Production"
    assert settings.app_version == "1.2.3"
    assert settings.environment == "production"
    assert settings.cors_origins == (
        "https://app.vera.example",
        "https://admin.vera.example",
    )
    assert settings.max_upload_bytes == 12345
