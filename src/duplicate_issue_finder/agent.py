from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from duplicate_issue_finder.github import (
    GitHubClient,
    issue_to_prompt_dict,
    search_result_to_prompt_dict,
)
from duplicate_issue_finder.models import DuplicateDecision, IssueDetails

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


class DuplicateIssueAgent:
    def __init__(
        self,
        github_client: GitHubClient,
        openai_api_key: str,
        model: str,
        *,
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
                target_issue=target_issue,
                fetched_issues=fetched_issues,
                search_history=search_history,
                observations=observations,
                force_final=step == self.max_steps,
            )

            action_name = action.get("action")
            if action_name == "search_issues":
                limit = min(
                    max(int(action.get("limit", self.max_search_results)), 1),
                    self.max_search_results,
                )
                query = str(action["query"]).strip()
                results = [
                    result
                    for result in self.github_client.search_issues(
                        query=query, limit=limit + 1
                    )
                    if result.number != issue_number
                ][:limit]
                serialized_results = [
                    search_result_to_prompt_dict(result) for result in results
                ]
                search_history.append({"query": query, "results": serialized_results})
                observations.append(
                    {
                        "step": step,
                        "action": "search_issues",
                        "reason": action.get("reason", ""),
                        "query": query,
                        "results": serialized_results,
                    }
                )
                continue

            if action_name == "get_issue":
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

                issue = fetched_issues.get(candidate_number)
                if issue is None:
                    issue = self.github_client.get_issue(candidate_number)
                    fetched_issues[candidate_number] = issue
                observations.append(
                    {
                        "step": step,
                        "action": "get_issue",
                        "reason": action.get("reason", ""),
                        "issue": issue_to_prompt_dict(issue),
                    }
                )
                continue

            if action_name == "final":
                return self._build_decision(action)

            raise ValueError(f"Unknown action returned by model: {action_name}")

        raise RuntimeError("Agent loop ended without a final answer")

    def _next_action(
        self,
        *,
        target_issue: IssueDetails,
        fetched_issues: dict[int, IssueDetails],
        search_history: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        force_final: bool,
    ) -> dict[str, Any]:
        prompt = {
            "repository": self.github_client.repository_name,
            "target_issue": issue_to_prompt_dict(target_issue),
            "already_fetched_candidates": [
                issue_to_prompt_dict(issue)
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
        return _parse_json_response(response.output_text)

    @staticmethod
    def _build_decision(payload: dict[str, Any]) -> DuplicateDecision:
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
            evidence_against=[
                str(item) for item in payload.get("evidence_against", [])
            ],
            considered_issue_numbers=[
                int(item) for item in payload.get("considered_issue_numbers", [])
            ],
        )


def _parse_json_response(text: str) -> dict[str, Any]:
    normalized = text.strip()
    if normalized.startswith("```"):
        normalized = normalized.split("\n", 1)[1]
        normalized = normalized.rsplit("```", 1)[0].strip()
    return json.loads(normalized)
