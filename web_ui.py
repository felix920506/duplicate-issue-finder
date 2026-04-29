from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import queue
import re
import tempfile
import threading

import gradio as gr

from cache_provider import RunCache, create_run_cache

from duplicate_issue_finder import (
    issue_url,
    load_settings,
    run_duplicate_check_with_logs,
)

logger = logging.getLogger(__name__)
DEFAULT_CONCURRENCY_LIMIT = 4
DEFAULT_MAX_QUEUE_SIZE = 32
LOGS_ELEMENT_ID = "run-logs"
RUN_CACHE: RunCache | None = None
ISSUE_URL_ELEM_ID = "issue-url-input"
AUTO_SCROLL_SCRIPT = f"""
<script>
(() => {{
  const ISSUE_URL_PATTERN = /^https:\/\/github\.com\/[\w.-]+\/[\w.-]+\/issues\/\d+$/;

  const setupUrlValidation = () => {{
    const container = document.getElementById('{ISSUE_URL_ELEM_ID}');
    if (!container || container.dataset.validationAttached === 'true') return;

    const input = container.querySelector('input');
    if (!input) return;

    container.dataset.validationAttached = 'true';

    let warning = document.getElementById('url-format-warning');
    if (!warning) {{
      warning = document.createElement('p');
      warning.id = 'url-format-warning';
      warning.style.cssText = 'color:#c0392b;font-size:0.85rem;margin:0.25rem 0 0;display:none;';
      warning.textContent = 'Enter a valid GitHub issue URL: https://github.com/owner/repo/issues/N';
      container.parentNode.insertBefore(warning, container.nextSibling);
    }}

    const validate = () => {{
      const val = input.value.trim();
      const invalid = val.length > 0 && !ISSUE_URL_PATTERN.test(val);
      input.style.outline = invalid ? '2px solid #e74c3c' : '';
      warning.style.display = invalid ? 'block' : 'none';
    }};

    input.addEventListener('input', validate);
    validate();
  }};

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
    setupUrlValidation();
  }};

  new MutationObserver(boot).observe(document.body, {{
    childList: true,
    subtree: true,
  }});
  window.addEventListener('load', boot);
  setInterval(syncScrollOnNewContent, 300);

  document.addEventListener('click', (event) => {{
    const button = event.target.closest('[data-open-urls]');
    if (!button) {{
      return;
    }}

    const raw = button.getAttribute('data-open-urls');
    if (!raw) {{
      return;
    }}

    try {{
      const urls = JSON.parse(raw);
      for (const url of urls) {{
        window.open(url, '_blank', 'noopener,noreferrer');
      }}
    }} catch (_error) {{
      console.error('Failed to open issue URLs');
    }}
  }});
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


def list_cached_run_choices() -> list[tuple[str, str]]:
    cache = get_run_cache()
    return [(cache.format_label(run), run.run_id) for run in cache.list_recent()]


def refresh_cached_run_choices():
    return gr.update(choices=list_cached_run_choices())


def load_cached_run(run_id: str | None):
    if not run_id:
        return format_error_markdown("No cached run selected"), "", "", None

    cached = get_run_cache().get(run_id)
    if cached is None:
        return format_error_markdown("Cached run not found"), "", "", None

    return (
        cached.result_markdown,
        cached.actions_html,
        cached.logs,
        write_logs_to_file(cached.issue_url, cached.logs),
    )


def get_run_cache() -> RunCache:
    global RUN_CACHE
    if RUN_CACHE is None:
        settings = load_settings()
        RUN_CACHE = create_run_cache(
            settings.cache_provider,
            settings.cache_path,
            max_runs=50,
        )
    return RUN_CACHE


def write_logs_to_file(issue_url: str, logs: str) -> str | None:
    if not logs:
        return None

    content = f"Issue URL: {issue_url}\n\n{logs}"

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".log",
        prefix="duplicate-issue-finder-",
        delete=False,
    ) as file:
        file.write(content)
        return file.name


def get_request_ip(request: gr.Request | None) -> str:
    if request is None:
        return "unknown"

    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()

    if request.client is not None and request.client.host:
        return request.client.host

    return "unknown"


def get_direct_client_ip(request: gr.Request | None) -> str | None:
    if request is None or request.client is None or not request.client.host:
        return None
    return request.client.host


def parse_trusted_proxies(
    entries: tuple[str, ...],
) -> tuple[ipaddress._BaseNetwork, ...]:
    networks: list[ipaddress._BaseNetwork] = []
    for entry in entries:
        try:
            networks.append(ipaddress.ip_network(entry, strict=False))
        except ValueError as exc:
            raise ValueError(f"Invalid TRUSTED_PROXIES entry: {entry}") from exc
    return tuple(networks)


def ensure_trusted_proxy(
    request: gr.Request | None, trusted_proxies: tuple[str, ...]
) -> None:
    if request is None:
        return

    forwarded_for = request.headers.get("x-forwarded-for")
    if not forwarded_for:
        return

    direct_client_ip = get_direct_client_ip(request)
    if direct_client_ip is None:
        raise PermissionError("Forwarded request missing direct client IP")

    networks = parse_trusted_proxies(trusted_proxies)
    direct_ip = ipaddress.ip_address(direct_client_ip)
    if not any(direct_ip in network for network in networks):
        raise PermissionError(
            "Request included X-Forwarded-For from an untrusted proxy IP"
        )


def build_action_buttons(result) -> str:
    original_url = issue_url(result.repository, result.issue_number)
    best_match_url = None
    if result.decision.duplicate_issue_number is not None:
        best_match_url = issue_url(
            result.repository, result.decision.duplicate_issue_number
        )

    all_urls: list[str] = [original_url]
    if best_match_url is not None:
        all_urls.append(best_match_url)
    for number in result.decision.considered_issue_numbers:
        candidate_url = issue_url(result.repository, number)
        if candidate_url not in all_urls:
            all_urls.append(candidate_url)

    original_and_best = [original_url]
    if best_match_url is not None:
        original_and_best.append(best_match_url)

    return "\n".join(
        [
            '<div style="display:flex;gap:0.75rem;flex-wrap:wrap;margin:0.5rem 0 1rem;">',
            (
                '<button type="button" style="padding:0.5rem 0.9rem;cursor:pointer;" '
                f"data-open-urls='{json.dumps(original_and_best)}'>"
                f"Open original{'' if best_match_url is None else ' + best match'}"
                "</button>"
            ),
            (
                '<button type="button" style="padding:0.5rem 0.9rem;cursor:pointer;" '
                f"data-open-urls='{json.dumps(all_urls)}'>Open all related issues</button>"
            ),
            "</div>",
            '<div style="font-size:0.9rem;color:#666;margin:-0.25rem 0 1rem;">'
            "Your browser may block opening multiple tabs at once and ask for confirmation."
            "</div>",
        ]
    )


def run_from_ui(
    issue_url: str,
    request: gr.Request,
):
    try:
        settings = load_settings()
        ensure_trusted_proxy(request, settings.trusted_proxies)
    except Exception as exc:
        logger.warning(
            "Rejected or failed web UI request from %s for %s: %s",
            get_request_ip(request),
            issue_url,
            exc,
        )
        return (
            format_error_markdown(exc),
            "",
            "",
            None,
            gr.update(),
            gr.update(),
        )

    logger.info(
        "Received web UI request from %s for %s",
        get_request_ip(request),
        issue_url,
    )

    log_queue: queue.Queue[str] = queue.Queue()
    state: dict[str, object] = {
        "result": None,
        "logs": "",
        "error": None,
        "result_markdown": "",
        "actions_html": "",
        "download_path": None,
    }

    def worker() -> None:
        result, logs, error = run_duplicate_check_with_logs(
            issue_url,
            settings=settings,
            log_sink=log_queue.put,
            webui_mode=True,
        )
        state["result"] = result
        state["logs"] = logs
        state["error"] = error
        if error is not None or result is None:
            message = error if error is not None else "Unknown error"
            state["result_markdown"] = format_error_markdown(message)
            state["actions_html"] = ""
        else:
            state["result_markdown"] = format_success_markdown(result.formatted_output)
            state["actions_html"] = build_action_buttons(result)

        state["download_path"] = write_logs_to_file(issue_url, logs)
        get_run_cache().store(
            issue_url,
            str(state["result_markdown"]),
            str(state["actions_html"]),
            logs,
        )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    collected_logs: list[str] = []
    yield "### Running...", "", "", None, gr.update(), gr.update(interactive=False)

    while thread.is_alive() or not log_queue.empty():
        try:
            line = log_queue.get(timeout=0.2)
            collected_logs.append(line)
            yield "### Running...", "", "\n".join(collected_logs), None, gr.update(), gr.update(interactive=False)
        except queue.Empty:
            continue

    logs = str(state["logs"])
    yield (
        str(state["result_markdown"]),
        str(state["actions_html"]),
        logs,
        state["download_path"] if isinstance(state["download_path"], str) else None,
        gr.update(choices=list_cached_run_choices()),
        gr.update(interactive=True),
    )


def build_demo() -> gr.Blocks:
    settings = load_settings()

    with gr.Blocks(title="Duplicate Issue Finder") as demo:
        gr.Markdown(
            "# Duplicate Issue Finder\n"
            "Check whether a GitHub issue URL is likely a duplicate of another issue in the same repository."
        )
        with gr.Accordion("Model configuration", open=False):
            gr.Markdown(
                f"Main model: `{settings.openai_model}`, Verifier model: `{settings.verifier_model or 'N/A'}`"
            )

        issue_url = gr.Textbox(
            label="Issue URL",
            placeholder="https://github.com/owner/repo/issues/1234",
            elem_id=ISSUE_URL_ELEM_ID,
        )

        run_button = gr.Button("Check for duplicates", variant="primary")
        result_markdown = gr.Markdown(
            label="Result",
            value="*Run a check or load a cached result to see output here.*",
        )
        actions_html = gr.HTML()
        with gr.Accordion("Cached Runs", open=False):
            recent_runs = gr.Dropdown(
                label="Cached Runs",
                choices=list_cached_run_choices(),
                value=None,
                allow_custom_value=False,
            )
        with gr.Accordion("Run Logs", open=False):
            download_logs = gr.DownloadButton("Download logs")
            logs = gr.Textbox(
                label="Run Logs",
                lines=20,
                interactive=False,
                elem_id=LOGS_ELEMENT_ID,
            )

        run_button.click(
            fn=run_from_ui,
            inputs=[issue_url],
            outputs=[result_markdown, actions_html, logs, download_logs, recent_runs, run_button],
        )
        issue_url.submit(
            fn=run_from_ui,
            inputs=[issue_url],
            outputs=[result_markdown, actions_html, logs, download_logs, recent_runs, run_button],
        )
        recent_runs.change(
            fn=load_cached_run,
            inputs=[recent_runs],
            outputs=[result_markdown, actions_html, logs, download_logs],
        )
        demo.load(
            fn=refresh_cached_run_choices,
            inputs=None,
            outputs=[recent_runs],
        )

        gr.HTML(
            '<a href="https://github.com/felix920506/duplicate-issue-finder" '
            'target="_blank" rel="noopener noreferrer">View on Github</a>'
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
    demo.launch(server_name=args.host, server_port=args.port, head=AUTO_SCROLL_SCRIPT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
