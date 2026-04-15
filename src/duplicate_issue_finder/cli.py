from __future__ import annotations

import typer

app = typer.Typer(help="Identify whether a GitHub issue is a likely duplicate.")


@app.command()
def main(issue_number: int) -> None:
    """CLI entrypoint placeholder."""
    typer.echo(f"Issue #{issue_number} support is not implemented yet.")


if __name__ == "__main__":
    app()
