from __future__ import annotations

import argparse
import queue
import re
import threading

import gradio as gr

from duplicate_issue_finder import load_settings, run_duplicate_check_with_logs

DEFAULT_CONCURRENCY_LIMIT = 4
DEFAULT_MAX_QUEUE_SIZE = 32
LOGS_ELEMENT_ID = "run-logs"
AUTO_SCROLL_SCRIPT = f"""
<script>
(() => {{
  const getTextarea = () => {{
    const container = document.getElementById('{LOGS_ELEMENT_ID}');
    return container?.querySelector('textarea');
  }};

  const scrollLogsToBottom = (textarea) => {{
    textarea.scrollTop = textarea.scrollHeight;
  }};

  const syncScrollOnNewContent = () => {{
    const textarea = getTextarea();
    if (!textarea) {{
      return;
    }}

    const previousLength = Number(textarea.dataset.previousLength || '0');
    const currentLength = textarea.value.length;
    if (currentLength > previousLength) {{
      scrollLogsToBottom(textarea);
    }}
    textarea.dataset.previousLength = String(currentLength);
  }};

  const observeLogs = () => {{
    const textarea = getTextarea();
    if (!textarea || textarea.dataset.autoscrollAttached === 'true') {{
      return;
    }}

    textarea.dataset.autoscrollAttached = 'true';
    textarea.dataset.previousLength = String(textarea.value.length);
    new MutationObserver(syncScrollOnNewContent).observe(textarea, {{
      attributes: true,
      attributeFilter: ['value'],
    }});
    textarea.addEventListener('input', syncScrollOnNewContent);
  }};

  const boot = () => {{
    observeLogs();
    syncScrollOnNewContent();
  }};

  new MutationObserver(boot).observe(document.body, {{
    childList: true,
    subtree: true,
  }});
  window.addEventListener('load', boot);
  setInterval(syncScrollOnNewContent, 300);
}})();
</script>
"""


def format_error_markdown(message: object) -> str:
    return f"### Run failed\n\n```\n{message}\n```"


def format_success_markdown(formatted_output: str) -> str:
    linked_output = re.sub(
        r"https://github\.com/[\w.-]+/[\w.-]+/issues/\d+",
        lambda match: (
            f'<a href="{match.group(0)}" target="_blank" rel="noopener noreferrer">'
            f"{match.group(0)}</a>"
        ),
        formatted_output,
    )
    html_output = linked_output.replace("\n", "<br>\n")
    return "\n".join(["### Result", "", html_output])


def run_from_ui(
    issue_url: str,
):
    try:
        settings = load_settings()
    except Exception as exc:
        return (
            format_error_markdown(exc),
            "",
        )

    log_queue: queue.Queue[str] = queue.Queue()
    state: dict[str, object] = {"result": None, "logs": "", "error": None}

    def worker() -> None:
        result, logs, error = run_duplicate_check_with_logs(
            issue_url,
            settings=settings,
            log_sink=log_queue.put,
        )
        state["result"] = result
        state["logs"] = logs
        state["error"] = error

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    collected_logs: list[str] = []
    yield "### Running...", ""

    while thread.is_alive() or not log_queue.empty():
        try:
            line = log_queue.get(timeout=0.2)
            collected_logs.append(line)
            yield "### Running...", "\n".join(collected_logs)
        except queue.Empty:
            continue

    logs = str(state["logs"])
    error = state["error"]
    result = state["result"]

    if error is not None or result is None:
        message = error if error is not None else "Unknown error"
        yield format_error_markdown(message), logs
        return

    yield format_success_markdown(result.formatted_output), logs


def build_demo() -> gr.Blocks:
    settings = load_settings()

    with gr.Blocks(title="Duplicate Issue Finder", head=AUTO_SCROLL_SCRIPT) as demo:
        gr.Markdown(
            "# Duplicate Issue Finder\n"
            "Check whether a GitHub issue URL is likely a duplicate of another issue in the same repository."
        )

        issue_url = gr.Textbox(
            label="Issue URL",
            placeholder="https://github.com/owner/repo/issues/1234",
        )

        run_button = gr.Button("Check for duplicates", variant="primary")
        result_markdown = gr.Markdown(label="Result")
        with gr.Accordion("Run Logs", open=False):
            logs = gr.Textbox(
                label="Run Logs",
                lines=20,
                interactive=False,
                elem_id=LOGS_ELEMENT_ID,
            )

        run_button.click(
            fn=run_from_ui,
            inputs=[issue_url],
            outputs=[result_markdown, logs],
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
