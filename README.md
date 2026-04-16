# duplicate-issue-finder

Read-only CLI that inspects a configured GitHub repository and determines whether a target issue is likely a duplicate of an existing issue.

The CLI runs a bounded agent loop. The agent receives the target issue body and comments, uses native tool calling for issue search and issue fetches, and prints a final duplicate decision to stdout.

## Configuration

Create a `.env` file in the repository root:

```dotenv
GITHUB_TOKEN=ghp_your_token
GITHUB_REPOSITORY=owner/repo
OPENAI_API_KEY=sk_your_key
OPENAI_MODEL=gpt-5-mini
OPENAI_BASE_URL=
AGENT_MAX_STEPS=6
```

Fields:

- `GITHUB_TOKEN`: token with read access to the repository
- `GITHUB_REPOSITORY`: repository in `owner/name` format
- `OPENAI_API_KEY`: API key used for the agent loop
- `OPENAI_MODEL`: optional; defaults to `gpt-5-mini`
- `OPENAI_BASE_URL`: optional; set this if you want to use a non-default OpenAI-compatible endpoint
- `AGENT_MAX_STEPS`: optional; defaults to `6`

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python duplicate_issue_finder.py 1234
```

The CLI prints whether the issue looks like a duplicate, the best matching issue if one was found, confidence, and a short explanation.

## How It Works

1. Fetch the target issue and its comments.
2. Let the model iteratively call the native `search_issues(query, limit)` and `get_issue(issue_number)` tools, including multiple calls in one step when useful.
3. Stop after a small fixed number of steps.
4. Print the final duplicate or non-duplicate decision.
