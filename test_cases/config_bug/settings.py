"""Configuration loading for the remote job service."""

from dataclasses import dataclass

from defaults import DEFAULT_REGION, DEFAULT_RETRY_ATTEMPTS, DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class ServiceSettings:
    region: str = DEFAULT_REGION
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS


def load_settings(overrides: dict[str, object] | None = None) -> ServiceSettings:
    """Apply user overrides while retaining defaults for omitted settings."""
    settings = ServiceSettings()
    if overrides is None:
        return settings

    return ServiceSettings(
        region=str(overrides.get("region", settings.region)),
        timeout_seconds=int(
            overrides.get("timeout_seconds", settings.timeout_seconds)
        ),
        retry_attempts=int(overrides.get("retry_attempts", 0)),
    )
