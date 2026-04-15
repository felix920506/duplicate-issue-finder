from __future__ import annotations

from dataclasses import dataclass

from github import Github


@dataclass(frozen=True)
class IssueSummary:
    number: int
    title: str
    state: str


class GitHubClient:
    def __init__(self, token: str, repository: str) -> None:
        self.token = token
        self.repository = repository
        self._client = Github(token)

    def get_issue(self, issue_number: int) -> dict:
        raise NotImplementedError

    def search_issues(self, query: str, limit: int) -> list[IssueSummary]:
        raise NotImplementedError
