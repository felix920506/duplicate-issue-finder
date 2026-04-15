from __future__ import annotations

from dataclasses import dataclass


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
