from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from github import Github
from github.Issue import Issue as PyGithubIssue
from openai import OpenAI

SYSTEM_PROMPT = """
You are a read-only agent that determines whether a target GitHub issue is a duplicate of an existing issue in the same repository.

You may only take one action at a time and must always return valid JSON with one of these shapes:

Search action:
{"action":"search_issues","reason":"short reason","query":"search text","limit":5}

Fetch action:
{"action":"get_issue","reason":"short reason","issue_number":123}

Final answer:
{"action":"final","is_duplicate":true,"duplicate_issue_number":123,"confidence":"high","summary":"short summary","evidence_for":["..."],"evidence_against":["..."],"considered_issue_numbers":[123,456]}

Rules:
- The target issue body and comments are important evidence.
- Distinguish duplicates from related issues.
- Search both open and closed issues.
- Prefer the older canonical issue when multiple issues describe the same underlying problem.
- Only conclude duplicate when the overlap is concrete.
- If evidence is weak, return a final non-duplicate decision instead of forcing a match.
- Never reference tools that do not exist.
""".strip()


@dataclass(frozen=True)
class Settings:
    github_token: str
    github_repository: str
    openai_api_key: str
    openai_model: str


@dataclass(frozen=True)
class IssueComment:
    author: str
    body: str


@dataclass(frozen=True)
class IssueDetails:
    number: int
    title: str
    body: str
    state: str
    labels: list[str]
    author: str
    created_at: str | None
    updated_at: str | None
    comments: list[IssueComment]


@dataclass(frozen=True)
class IssueSearchResult:
    number: int
    title: str
    state: str
    labels: list[str]
    created_at: str | None
    updated_at: str | None


@dataclass(frozen=True)
class DuplicateDecision:
    is_duplicate: bool
    confidence: str
    summary: str
    duplicate_issue_number: int | None
    evidence_for: list[str]
    evidence_against: list[str]
    considered_issue_numbers: list[int]


class GitHubClient:
    def __init__(self, token: str, repository: str) -> None:
        self.repository_name = repository
        self._client = Github(token)
        self._repository = self._client.get_repo(repository)

    def get_issue(self, issue_number: int) -> IssueDetails:
        issue = self._repository.get_issue(number=issue_number)
        self._ensure_issue(issue)
        comments = [
            IssueComment(author=comment.user.login, body=comment.body or "")
            for comment in issue.get_comments()
        ]
        return IssueDetails(
            number=issue.number,
            title=issue.title,
            body=issue.body or "",
            state=issue.state,
            labels=[label.name for label in issue.labels],
            author=issue.user.login,
            created_at=_isoformat(issue.created_at),
            updated_at=_isoformat(issue.updated_at),
            comments=comments,
        )

    def search_issues(self, query: str, limit: int) -> list[IssueSearchResult]:
        qualified_query = f"repo:{self.repository_name} is:issue {query}".strip()
        results: list[IssueSearchResult] = []
        for issue in self._client.search_issues(query=qualified_query):
            self._ensure_issue(issue)
            results.append(
                IssueSearchResult(
                    number=issue.number,
                    title=issue.title,
                    state=issue.state,
                    labels=[label.name for label in issue.labels],
                    created_at=_isoformat(issue.created_at),
                    updated_at=_isoformat(issue.updated_at),
                )
            )
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _ensure_issue(issue: PyGithubIssue) -> None:
        if issue.pull_request is not None:
            raise ValueError(f"#{issue.number} is a pull request, not an issue")


class DuplicateIssueAgent:
    def __init__(
        self,
        github_client: GitHubClient,
        openai_api_key: str,
        model: str,
        max_steps: int = 6,
        max_search_results: int = 5,
        max_fetched_candidates: int = 8,
    ) -> None:
        self.github_client = github_client
        self.model = model
        self.max_steps = max_steps
        self.max_search_results = max_search_results
        self.max_fetched_candidates = max_fetched_candidates
        self._client = OpenAI(api_key=openai_api_key)

    def run(self, issue_number: int) -> DuplicateDecision:
        target_issue = self.github_client.get_issue(issue_number)
        fetched_issues: dict[int, IssueDetails] = {issue_number: target_issue}
        search_history: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []

        for step in range(1, self.max_steps + 1):
            action = self._next_action(
                target_issue,
                fetched_issues,
                search_history,
                observations,
                step == self.max_steps,
            )
            name = action.get("action")

            if name == "search_issues":
                query = str(action["query"]).strip()
                limit = min(
                    max(int(action.get("limit", self.max_search_results)), 1),
                    self.max_search_results,
                )
                results = [
                    asdict(result)
                    for result in self.github_client.search_issues(query, limit + 1)
                    if result.number != issue_number
                ][:limit]
                search_history.append({"query": query, "results": results})
                observations.append(
                    {
                        "step": step,
                        "action": "search_issues",
                        "reason": action.get("reason", ""),
                        "query": query,
                        "results": results,
                    }
                )
                continue

            if name == "get_issue":
                candidate_number = int(action["issue_number"])
                if candidate_number == issue_number:
                    observations.append(
                        {
                            "step": step,
                            "action": "get_issue",
                            "issue_number": candidate_number,
                            "error": "The target issue cannot be fetched as a candidate.",
                        }
                    )
                    continue
                if (
                    candidate_number not in fetched_issues
                    and len(fetched_issues) - 1 >= self.max_fetched_candidates
                ):
                    observations.append(
                        {
                            "step": step,
                            "action": "get_issue",
                            "issue_number": candidate_number,
                            "error": "Candidate fetch budget exhausted.",
                        }
                    )
                    continue
                if candidate_number not in fetched_issues:
                    fetched_issues[candidate_number] = self.github_client.get_issue(
                        candidate_number
                    )
                observations.append(
                    {
                        "step": step,
                        "action": "get_issue",
                        "reason": action.get("reason", ""),
                        "issue": asdict(fetched_issues[candidate_number]),
                    }
                )
                continue

            if name == "final":
                return build_decision(action)

            raise ValueError(f"Unknown action returned by model: {name}")

        raise RuntimeError("Agent loop ended without a final answer")

    def _next_action(
        self,
        target_issue: IssueDetails,
        fetched_issues: dict[int, IssueDetails],
        search_history: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        force_final: bool,
    ) -> dict[str, Any]:
        prompt = {
            "repository": self.github_client.repository_name,
            "target_issue": asdict(target_issue),
            "already_fetched_candidates": [
                asdict(issue)
                for number, issue in fetched_issues.items()
                if number != target_issue.number
            ],
            "search_history": search_history,
            "observations": observations,
            "limits": {
                "max_search_results_per_query": self.max_search_results,
                "max_fetched_candidates": self.max_fetched_candidates,
                "must_return_final": force_final,
            },
        }
        response = self._client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt, indent=2)},
            ],
        )
        return parse_json_response(response.output_text)


def load_settings() -> Settings:
    repository = os.environ["GITHUB_REPOSITORY"]
    if "/" not in repository:
        raise ValueError("GITHUB_REPOSITORY must be in owner/name format")
    return Settings(
        github_token=os.environ["GITHUB_TOKEN"],
        github_repository=repository,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
    )


def build_decision(payload: dict[str, Any]) -> DuplicateDecision:
    duplicate_issue_number = payload.get("duplicate_issue_number")
    if payload.get("is_duplicate") and duplicate_issue_number is None:
        raise ValueError(
            "Model marked issue as duplicate without a matching issue number"
        )
    return DuplicateDecision(
        is_duplicate=bool(payload["is_duplicate"]),
        confidence=str(payload["confidence"]),
        summary=str(payload["summary"]),
        duplicate_issue_number=int(duplicate_issue_number)
        if duplicate_issue_number is not None
        else None,
        evidence_for=[str(item) for item in payload.get("evidence_for", [])],
        evidence_against=[str(item) for item in payload.get("evidence_against", [])],
        considered_issue_numbers=[
            int(item) for item in payload.get("considered_issue_numbers", [])
        ],
    )


def parse_json_response(text: str) -> dict[str, Any]:
    normalized = text.strip()
    if normalized.startswith("```"):
        normalized = normalized.split("\n", 1)[1]
        normalized = normalized.rsplit("```", 1)[0].strip()
    return json.loads(normalized)


def format_decision(issue_number: int, decision: DuplicateDecision) -> str:
    lines = []
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify whether a GitHub issue is a likely duplicate."
    )
    parser.add_argument("issue_number", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        settings = load_settings()
        agent = DuplicateIssueAgent(
            GitHubClient(settings.github_token, settings.github_repository),
            settings.openai_api_key,
            settings.openai_model,
        )
        decision = agent.run(args.issue_number)
    except KeyError as exc:
        print(f"Error: missing required environment variable {exc.args[0]}")
        return 1
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print(format_decision(args.issue_number, decision))
    return 0


def _isoformat(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


if __name__ == "__main__":
    raise SystemExit(main())
