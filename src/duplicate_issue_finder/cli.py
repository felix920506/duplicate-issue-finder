from __future__ import annotations

import typer

from duplicate_issue_finder.agent import DuplicateIssueAgent
from duplicate_issue_finder.config import load_settings
from duplicate_issue_finder.github import GitHubClient
from duplicate_issue_finder.models import DuplicateDecision


def main(issue_number: int) -> None:
    """Identify whether a GitHub issue is a likely duplicate."""
    try:
        settings = load_settings()
    except KeyError as exc:
        missing_name = exc.args[0]
        raise typer.BadParameter(
            f"Missing required environment variable: {missing_name}"
        ) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    agent = DuplicateIssueAgent(
        github_client=GitHubClient(settings.github_token, settings.github_repository),
        openai_api_key=settings.openai_api_key,
        model=settings.openai_model,
    )
    try:
        decision = agent.run(issue_number)
    except Exception as exc:  # pragma: no cover - exercised via integration use
        raise typer.Exit(code=_print_error(exc)) from exc

    typer.echo(format_decision(issue_number, decision))


def format_decision(issue_number: int, decision: DuplicateDecision) -> str:
    lines: list[str] = []
    if decision.is_duplicate and decision.duplicate_issue_number is not None:
        lines.append(
            f"Issue #{issue_number} is likely a duplicate of #{decision.duplicate_issue_number}"
        )
    else:
        lines.append(f"Issue #{issue_number} does not appear to be a duplicate")

    lines.append(f"Confidence: {decision.confidence}")
    lines.append("")
    lines.append("Summary:")
    lines.append(decision.summary)

    if decision.evidence_for:
        lines.append("")
        lines.append("Why:")
        for item in decision.evidence_for:
            lines.append(f"- {item}")

    if decision.evidence_against:
        lines.append("")
        lines.append("Why not:")
        for item in decision.evidence_against:
            lines.append(f"- {item}")

    if decision.considered_issue_numbers:
        lines.append("")
        lines.append("Considered issues:")
        for number in decision.considered_issue_numbers:
            lines.append(f"- #{number}")

    return "\n".join(lines)


def _print_error(exc: Exception) -> int:
    typer.echo(f"Error: {exc}", err=True)
    return 1


def run() -> None:
    typer.run(main)


if __name__ == "__main__":
    run()
