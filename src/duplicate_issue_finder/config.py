from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    github_token: str
    github_repository: str
    openai_api_key: str
    openai_model: str = "gpt-5-mini"


def load_settings() -> Settings:
    return Settings(
        github_token=os.environ["GITHUB_TOKEN"],
        github_repository=os.environ["GITHUB_REPOSITORY"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
    )
