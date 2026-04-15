# duplicate-issue-finder

Read-only CLI that inspects a configured GitHub repository and determines whether a target issue is likely a duplicate of an existing issue.

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
