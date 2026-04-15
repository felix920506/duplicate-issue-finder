# duplicate-issue-finder

Read-only CLI that inspects a configured GitHub repository and determines whether a target issue is likely a duplicate of an existing issue.

The CLI runs a bounded agent loop. The agent receives the target issue body and comments, can search the issue tracker, can fetch candidate issues by number, and prints a final duplicate decision to stdout.

## Configuration

Set these environment variables before running the CLI:

- `GITHUB_TOKEN`: token with read access to the repository
- `GITHUB_REPOSITORY`: repository in `owner/name` format
- `OPENAI_API_KEY`: API key used for the agent loop
- `OPENAI_MODEL`: optional; defaults to `gpt-5-mini`

## Usage

```bash
duplicate-issue-finder 1234
```

The CLI prints whether the issue looks like a duplicate, the best matching issue if one was found, confidence, and a short explanation.

## How It Works

1. Fetch the target issue and its comments.
2. Let the model iteratively choose between two read-only tools:
   - `search_issues(query, limit)`
   - `get_issue(issue_number)`
3. Stop after a small fixed number of steps.
4. Print the final duplicate or non-duplicate decision.
