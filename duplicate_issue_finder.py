from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from github import Auth, Github
from github.GithubException import UnknownObjectException
from github.Issue import Issue as PyGithubIssue
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
SYSTEM_PROMPT = (
    (Path(__file__).with_name("system_prompt.txt")).read_text(encoding="utf-8").strip()
)
VERIFIER_PROMPT = (
    (Path(__file__).with_name("verifier_prompt.txt"))
    .read_text(encoding="utf-8")
    .strip()
)


@dataclass(frozen=True)
class Settings:
    github_token: str
    openai_api_key: str
    openai_model: str
    verifier_model: str | None
    openai_base_url: str | None
    agent_max_steps: int
    search_max_results: int


@dataclass(frozen=True)
class ParsedIssueUrl:
    repository: str
    issue_number: int


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
        logger.info("Initializing GitHub client for %s", repository)
        self._client = Github(auth=Auth.Token(token))
        self._repository = self._client.get_repo(repository)

    def get_issue(self, issue_number: int) -> IssueDetails:
        logger.info("Fetching issue #%s", issue_number)
        issue = self._repository.get_issue(number=issue_number)
        self._ensure_issue(issue)
        comments = [
            IssueComment(author=comment.user.login, body=comment.body or "")
            for comment in issue.get_comments()
        ]
        logger.info(
            "Fetched issue #%s with %s comments and %s labels",
            issue.number,
            len(comments),
            len(issue.labels),
        )
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
        logger.info("Searching issues with query: %s", qualified_query)
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
        logger.info("Search returned %s issues", len(results))
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
        verifier_model: str | None = None,
        base_url: str | None = None,
        max_steps: int = 6,
        max_search_results: int = 25,
        max_fetched_candidates: int = 8,
    ) -> None:
        self.github_client = github_client
        self.model = model
        self.verifier_model = verifier_model
        self.max_steps = max_steps
        self.max_search_results = max_search_results
        self.max_fetched_candidates = max_fetched_candidates
        logger.info(
            "Initializing OpenAI client with model=%s base_url=%s",
            model,
            base_url or "default",
        )
        self._client = OpenAI(api_key=openai_api_key, base_url=base_url)

    def run(self, issue_number: int) -> DuplicateDecision:
        logger.info("Starting duplicate detection for issue #%s", issue_number)
        target_issue = self.github_client.get_issue(issue_number)
        fetched_issues: dict[int, IssueDetails] = {issue_number: target_issue}
        conversation = self._build_initial_input(target_issue)
        response = self._client.responses.create(
            model=self.model,
            input=conversation,
            tools=self._build_tools(),
            parallel_tool_calls=True,
        )
        logger.info(
            "Received model response (%s characters)", len(response.output_text)
        )

        for step in range(1, self.max_steps + 1):
            logger.info("Agent step %s/%s", step, self.max_steps)
            tool_calls = self._extract_tool_calls(response)
            if not tool_calls:
                logger.info("Model returned final decision")
                decision = build_decision(parse_json_response(response.output_text))
                return self._verify_decision_if_configured(
                    target_issue,
                    fetched_issues,
                    decision,
                )

            logger.info("Model requested %s tool calls", len(tool_calls))
            conversation.extend(self._response_items_to_input(response))
            tool_outputs = self._execute_tool_calls(
                issue_number, fetched_issues, tool_calls
            )
            conversation.extend(tool_outputs)
            response = self._client.responses.create(
                model=self.model,
                input=conversation,
                tools=self._build_tools(),
                parallel_tool_calls=True,
            )
            logger.info(
                "Received follow-up model response (%s characters)",
                len(response.output_text),
            )

        raise RuntimeError(
            "Agent reached the maximum number of steps without a final answer"
        )

    def _verify_decision_if_configured(
        self,
        target_issue: IssueDetails,
        fetched_issues: dict[int, IssueDetails],
        decision: DuplicateDecision,
    ) -> DuplicateDecision:
        if not self.verifier_model:
            logger.info("No verifier model configured; skipping verification")
            return decision

        logger.info("Verifying decision with model=%s", self.verifier_model)
        verifier_input = {
            "repository": self.github_client.repository_name,
            "target_issue": asdict(target_issue),
            "primary_decision": asdict(decision),
            "fetched_candidate_issues": [
                asdict(issue)
                for number, issue in fetched_issues.items()
                if number != target_issue.number
            ],
        }
        response = self._client.responses.create(
            model=self.verifier_model,
            input=[
                {"role": "system", "content": VERIFIER_PROMPT},
                {"role": "user", "content": json.dumps(verifier_input, indent=2)},
            ],
        )
        logger.info(
            "Received verifier response (%s characters)", len(response.output_text)
        )
        return build_decision(parse_json_response(response.output_text))

    def _build_initial_input(self, target_issue: IssueDetails) -> list[dict[str, Any]]:
        logger.info("Building initial model input")
        prompt = {
            "repository": self.github_client.repository_name,
            "target_issue": asdict(target_issue),
            "limits": {
                "max_search_results_per_query": self.max_search_results,
                "max_fetched_candidates": self.max_fetched_candidates,
                "max_steps": self.max_steps,
            },
        }
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(prompt, indent=2)},
        ]

    def _build_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "search_issues",
                "description": "Search GitHub issues in the configured repository",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": self.max_search_results,
                        },
                    },
                    "required": ["query", "limit"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "get_issue",
                "description": "Fetch a GitHub issue by number from the configured repository",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "issue_number": {"type": "integer", "minimum": 1},
                    },
                    "required": ["issue_number"],
                    "additionalProperties": False,
                },
            },
        ]

    def _extract_tool_calls(self, response: Any) -> list[Any]:
        tool_calls = [item for item in response.output if item.type == "function_call"]
        for item in tool_calls:
            logger.info("Model requested tool=%s call_id=%s", item.name, item.call_id)
        return tool_calls

    def _response_items_to_input(self, response: Any) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for item in response.output:
            if hasattr(item, "model_dump"):
                items.append(item.model_dump(exclude_none=True))
            else:
                items.append(dict(item))
        logger.info("Converted %s response items into follow-up input", len(items))
        return items

    def _execute_tool_calls(
        self,
        target_issue_number: int,
        fetched_issues: dict[int, IssueDetails],
        tool_calls: list[Any],
    ) -> list[dict[str, str]]:
        outputs: list[dict[str, str]] = []
        for tool_call in tool_calls:
            arguments = json.loads(tool_call.arguments)
            if tool_call.name == "search_issues":
                query = str(arguments["query"]).strip()
                limit = min(
                    max(int(arguments["limit"]), 1),
                    self.max_search_results,
                )
                logger.info(
                    "Executing tool search_issues query=%r limit=%s", query, limit
                )
                results = [
                    asdict(result)
                    for result in self.github_client.search_issues(query, limit + 1)
                    if result.number != target_issue_number
                ][:limit]
                logger.info("Tool search_issues returned %s issues", len(results))
                output = results
            elif tool_call.name == "get_issue":
                candidate_number = int(arguments["issue_number"])
                logger.info(
                    "Executing tool get_issue issue_number=%s", candidate_number
                )
                if candidate_number == target_issue_number:
                    output = {
                        "issue_number": candidate_number,
                        "error": "The target issue cannot be fetched as a candidate.",
                    }
                elif (
                    candidate_number not in fetched_issues
                    and len(fetched_issues) - 1 >= self.max_fetched_candidates
                ):
                    output = {
                        "issue_number": candidate_number,
                        "error": "Candidate fetch budget exhausted.",
                    }
                else:
                    if candidate_number not in fetched_issues:
                        fetched_issues[candidate_number] = self.github_client.get_issue(
                            candidate_number
                        )
                    else:
                        logger.info(
                            "Candidate issue #%s already fetched, reusing cached copy",
                            candidate_number,
                        )
                    output = asdict(fetched_issues[candidate_number])
            else:
                raise ValueError(f"Unknown tool requested by model: {tool_call.name}")

            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": json.dumps(output),
                }
            )
        return outputs


def load_settings() -> Settings:
    logger.info("Loading configuration")
    load_dotenv()
    agent_max_steps = int(os.environ.get("AGENT_MAX_STEPS", "6"))
    if agent_max_steps < 1:
        raise ValueError("AGENT_MAX_STEPS must be greater than 0")
    search_max_results = int(os.environ.get("SEARCH_MAX_RESULTS", "25"))
    if search_max_results < 1:
        raise ValueError("SEARCH_MAX_RESULTS must be greater than 0")
    logger.info("Loaded configuration")
    return Settings(
        github_token=os.environ["GITHUB_TOKEN"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
        verifier_model=(os.environ.get("VERIFIER_MODEL") or "").strip() or None,
        openai_base_url=os.environ.get("OPENAI_BASE_URL") or None,
        agent_max_steps=agent_max_steps,
        search_max_results=search_max_results,
    )


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        logger.info("No %s file found, using existing environment", path)
        return

    logger.info("Loading environment variables from %s", path)

    with open(path, encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or key in os.environ:
                if key in os.environ:
                    logger.info("Keeping existing environment value for %s", key)
                continue

            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            os.environ[key] = value
            logger.info("Loaded %s from %s", key, path)


def build_decision(payload: dict[str, Any]) -> DuplicateDecision:
    duplicate_issue_number = payload.get("duplicate_issue_number")
    if payload.get("is_duplicate") and duplicate_issue_number is None:
        raise ValueError(
            "Model marked issue as duplicate without a matching issue number"
        )
    decision = DuplicateDecision(
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
    logger.info(
        "Built decision duplicate=%s match=%s confidence=%s",
        decision.is_duplicate,
        decision.duplicate_issue_number,
        decision.confidence,
    )
    return decision


def parse_json_response(text: str) -> dict[str, Any]:
    logger.info("Parsing model response as JSON")
    normalized = text.strip()
    if normalized.startswith("```"):
        logger.info("Stripping fenced code block from model response")
        normalized = normalized.split("\n", 1)[1]
        normalized = normalized.rsplit("```", 1)[0].strip()

    object_start = normalized.find("{")
    if object_start == -1:
        raise ValueError(f"Model did not return JSON: {text}")

    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(normalized[object_start:])
    if not isinstance(payload, dict):
        raise ValueError(f"Model returned non-object JSON: {payload}")
    logger.info("Parsed JSON action=%s", payload.get("action"))
    return payload


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
    parser.add_argument("issue_url")
    return parser.parse_args()


def parse_issue_url(issue_url: str) -> ParsedIssueUrl:
    parsed = urlparse(issue_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Issue URL must start with http:// or https://")
    if parsed.netloc != "github.com":
        raise ValueError("Issue URL must be a github.com issue URL")

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 4 or parts[2] != "issues":
        raise ValueError(
            "Issue URL must look like https://github.com/owner/repo/issues/123"
        )

    repository = f"{parts[0]}/{parts[1]}"
    try:
        issue_number = int(parts[3])
    except ValueError as exc:
        raise ValueError("Issue URL must end with a numeric issue number") from exc

    return ParsedIssueUrl(repository=repository, issue_number=issue_number)


def main() -> int:
    args = parse_args()
    try:
        parsed_issue = parse_issue_url(args.issue_url)
        logger.info(
            "Starting CLI for %s issue #%s",
            parsed_issue.repository,
            parsed_issue.issue_number,
        )
        settings = load_settings()
        agent = DuplicateIssueAgent(
            GitHubClient(settings.github_token, parsed_issue.repository),
            settings.openai_api_key,
            settings.openai_model,
            verifier_model=settings.verifier_model,
            base_url=settings.openai_base_url,
            max_steps=settings.agent_max_steps,
            max_search_results=settings.search_max_results,
        )
        decision = agent.run(parsed_issue.issue_number)
    except KeyError as exc:
        logger.error("Missing required configuration: %s", exc.args[0])
        print(f"Error: missing required environment variable {exc.args[0]}")
        return 1
    except UnknownObjectException:
        logger.warning("Issue #%s was not found", parsed_issue.issue_number)
        print(f"Error: issue #{parsed_issue.issue_number} was not found")
        return 1
    except ValueError as exc:
        logger.warning("Input error: %s", exc)
        print(f"Error: {exc}")
        return 1
    except Exception as exc:
        logger.exception("Duplicate detection failed")
        print(f"Error: {exc}")
        return 1

    logger.info("Duplicate detection completed successfully")
    print(format_decision(parsed_issue.issue_number, decision))
    return 0


def _isoformat(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


if __name__ == "__main__":
    raise SystemExit(main())
