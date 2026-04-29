# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run CLI:**
```bash
python duplicate_issue_finder.py https://github.com/owner/repo/issues/1234
```

**Run web UI** (http://127.0.0.1:7860):
```bash
python web_ui.py
```

**Docker:**
```bash
docker-compose up
```

There is no test suite.

## Configuration

Create a `.env` file with:

| Variable | Required | Default | Description |
|---|---|---|---|
| `GITHUB_TOKEN` | Yes | — | GitHub token with read access |
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Primary model |
| `VERIFIER_MODEL` | No | — | Optional second model for skeptical verification |
| `AGENT_MAX_STEPS` | No | `6` | Max agent loop iterations |
| `SEARCH_MAX_RESULTS` | No | `25` | Max results per search |
| `CACHE_PROVIDER` | No | `sqlite` | `sqlite` or `memory` |
| `CACHE_PATH` | No | — | Path for SQLite cache file |

## Architecture

Three main modules:

### `duplicate_issue_finder.py` — Core logic & CLI entry point
- `Settings` — reads env vars into a frozen config dataclass
- `GitHubClient` — wraps PyGithub for issue fetching and searching; supports lexical, hybrid, and semantic search strategies
- `DuplicateIssueAgent` — orchestrates the agent loop using OpenAI native tool calling
- Domain models (`IssueDetails`, `IssueSearchResult`, `DuplicateDecision`, `DuplicateCheckResult`) — all frozen dataclasses

### `web_ui.py` — Gradio web interface
- Streams agent log messages in real-time via context variables
- Request queue for concurrent checks (default: 4 concurrent, 32 max queued)
- Displays and loads cached prior runs from a dropdown

### `cache_provider.py` — Run persistence
- `RunCache` Protocol defining the interface
- `InMemoryRunCache` — thread-safe ordered dict with LRU eviction
- `SQLiteRunCache` — thread-safe persistent disk cache

### Agent loop (`DuplicateIssueAgent.run`)
1. Fetch the target issue and its comments
2. LLM iteratively calls two tools: `search_issues(query, limit, search_type)` and `get_issue(issue_number)`
3. After `max_steps` iterations, produce a `DuplicateDecision` as JSON
4. Optionally pass decision + evidence to `verifier_model` for a skeptical second opinion

System prompts are in `system_prompt.txt` (primary agent) and `verifier_prompt.txt` (verifier).

### Key patterns
- **`contextvars`** — used for thread-safe propagation of `WEBUI_MODE`, `ACTIVE_LOG_MESSAGES`, and `ACTIVE_LOG_SINK` across the async/threaded Gradio context
- **Frozen dataclasses** — all domain models are immutable
- **Custom logging** — `log_runtime()` routes to either the Gradio UI element or stdout depending on context

## CI

`.github/workflows/publish-ghcr.yml` builds a multi-architecture (amd64 + arm64) Docker image and pushes it to GitHub Container Registry on pushes to `main`, tags, or manual dispatch.
