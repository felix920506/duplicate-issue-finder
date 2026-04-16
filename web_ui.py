from __future__ import annotations

import argparse

import gradio as gr

from duplicate_issue_finder import (
    apply_settings_overrides,
    load_settings,
    run_duplicate_check_with_logs,
)

DEFAULT_CONCURRENCY_LIMIT = 4
DEFAULT_MAX_QUEUE_SIZE = 32


def format_error_markdown(message: object) -> str:
    return f"### Run failed\n\n```\n{message}\n```"


def run_from_ui(
    issue_url: str,
    openai_model: str,
    verifier_model: str,
    agent_max_steps: float,
    search_max_results: float,
) -> tuple[str, str, str]:
    try:
        settings = apply_settings_overrides(
            load_settings(),
            openai_model=openai_model.strip() or None,
            verifier_model=verifier_model.strip(),
            agent_max_steps=int(agent_max_steps),
            search_max_results=int(search_max_results),
        )
        result, logs, error = run_duplicate_check_with_logs(
            issue_url, settings=settings
        )
    except Exception as exc:
        return (
            format_error_markdown(exc),
            "",
            "",
        )

    if error is not None or result is None:
        message = error if error is not None else "Unknown error"
        return (
            format_error_markdown(message),
            "",
            logs,
        )

    summary = [
        "### Result",
        "",
        result.formatted_output,
    ]
    return "\n".join(summary), result.formatted_output, logs


def build_demo() -> gr.Blocks:
    settings = load_settings()

    with gr.Blocks(title="Duplicate Issue Finder") as demo:
        gr.Markdown(
            "# Duplicate Issue Finder\n"
            "Check whether a GitHub issue URL is likely a duplicate of another issue in the same repository."
        )

        issue_url = gr.Textbox(
            label="Issue URL",
            placeholder="https://github.com/owner/repo/issues/1234",
        )

        with gr.Accordion("Advanced Settings", open=False):
            openai_model = gr.Textbox(
                label="Main Model",
                value=settings.openai_model,
            )
            verifier_model = gr.Textbox(
                label="Verifier Model",
                value=settings.verifier_model or "",
                placeholder="Leave blank to disable verification",
            )
            agent_max_steps = gr.Number(
                label="Agent Max Steps",
                value=settings.agent_max_steps,
                precision=0,
                minimum=1,
            )
            search_max_results = gr.Number(
                label="Search Max Results",
                value=settings.search_max_results,
                precision=0,
                minimum=1,
            )

        run_button = gr.Button("Check for duplicates", variant="primary")
        result_markdown = gr.Markdown(label="Result")
        formatted_output = gr.Textbox(
            label="Formatted Output",
            lines=16,
            interactive=False,
        )
        logs = gr.Textbox(
            label="Run Logs",
            lines=20,
            interactive=False,
        )

        run_button.click(
            fn=run_from_ui,
            inputs=[
                issue_url,
                openai_model,
                verifier_model,
                agent_max_steps,
                search_max_results,
            ],
            outputs=[result_markdown, formatted_output, logs],
        )

    demo.queue(
        default_concurrency_limit=DEFAULT_CONCURRENCY_LIMIT,
        max_size=DEFAULT_MAX_QUEUE_SIZE,
    )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Duplicate Issue Finder web UI."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    demo = build_demo()
    demo.launch(server_name=args.host, server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
