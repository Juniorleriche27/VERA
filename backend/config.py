from __future__ import annotations

from dataclasses import dataclass
import os


def _csv_env(name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return value


@dataclass(frozen=True, slots=True)
class AppSettings:
    app_name: str = "VERA API"
    app_version: str = "0.1.0"
    environment: str = "development"
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)
    max_upload_bytes: int = 25 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            app_name=os.getenv("VERA_APP_NAME", "VERA API").strip() or "VERA API",
            app_version=os.getenv("VERA_APP_VERSION", "0.1.0").strip() or "0.1.0",
            environment=os.getenv("VERA_ENV", "development").strip() or "development",
            cors_origins=_csv_env("VERA_CORS_ORIGINS", "http://localhost:3000"),
            max_upload_bytes=_positive_int_env(
                "VERA_MAX_UPLOAD_BYTES",
                25 * 1024 * 1024,
            ),
        )


def get_settings() -> AppSettings:
    return AppSettings.from_env()
