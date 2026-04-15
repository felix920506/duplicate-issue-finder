from __future__ import annotations

from dataclasses import asdict
from datetime import datetime

from github import Github
from github.Issue import Issue as PyGithubIssue

from duplicate_issue_finder.models import IssueComment, IssueDetails, IssueSearchResult


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


def issue_to_prompt_dict(issue: IssueDetails) -> dict:
    return asdict(issue)


def search_result_to_prompt_dict(result: IssueSearchResult) -> dict:
    return asdict(result)


def _isoformat(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None
