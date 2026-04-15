import json

from duplicate_issue_finder.agent import _parse_json_response
from duplicate_issue_finder.cli import format_decision
from duplicate_issue_finder.config import Settings
from duplicate_issue_finder.models import DuplicateDecision


def test_settings_defaults_openai_model() -> None:
    settings = Settings(
        github_token="token",
        github_repository="owner/repo",
        openai_api_key="key",
    )

    assert settings.openai_model == "gpt-5-mini"


def test_parse_json_response_accepts_fenced_json() -> None:
    payload = {"action": "final", "is_duplicate": False}

    parsed = _parse_json_response(f"```json\n{json.dumps(payload)}\n```")

    assert parsed == payload


def test_format_decision_renders_duplicate_result() -> None:
    decision = DuplicateDecision(
        is_duplicate=True,
        confidence="high",
        summary="The target issue matches the existing crash report.",
        duplicate_issue_number=42,
        evidence_for=["Same stack trace"],
        evidence_against=["Different browser version"],
        considered_issue_numbers=[42, 77],
    )

    rendered = format_decision(101, decision)

    assert "Issue #101 is likely a duplicate of #42" in rendered
    assert "- Same stack trace" in rendered
    assert "- #77" in rendered
