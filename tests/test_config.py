from duplicate_issue_finder.config import Settings


def test_settings_defaults_openai_model() -> None:
    settings = Settings(
        github_token="token",
        github_repository="owner/repo",
        openai_api_key="key",
    )

    assert settings.openai_model == "gpt-5-mini"
