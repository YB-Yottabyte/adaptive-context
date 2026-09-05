from settings import ServiceSettings, load_settings


def test_uses_all_defaults_without_overrides() -> None:
    assert load_settings() == ServiceSettings(
        region="us-west",
        timeout_seconds=15,
        retry_attempts=3,
    )


def test_partial_override_preserves_unrelated_defaults() -> None:
    settings = load_settings({"region": "eu-central"})

    assert settings.region == "eu-central"
    assert settings.timeout_seconds == 15
    assert settings.retry_attempts == 3
