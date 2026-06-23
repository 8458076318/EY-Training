"""Converted from `Autogen_groq (1).ipynb` into a runnable Python script.

The notebook mixed three separate concerns:
- installing dependencies in Colab,
- launching AutoGen Studio through ngrok,
- and running a Groq-backed AutoGen multi-agent demo.

This script keeps those workflows, but makes them usable from a local
Windows shell. It loads environment variables from the repo root `.env`
file first, so `GROQ_API_KEY` and `NGROK_AUTHTOKEN` can be supplied without
prompting.
"""

from __future__ import annotations

import argparse
import asyncio
import http.client
import os
import subprocess
import sys
import time
from pathlib import Path


APP_PORT = 8081
APP_HOST = "0.0.0.0"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
RESEARCH_MODEL = "llama-3.1-8b-instant"
EDITOR_MODEL = "llama-3.3-70b-versatile"


def load_repo_env() -> Path | None:
    """Load KEY=VALUE pairs from the repo root `.env` into ``os.environ``."""

    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        if line.startswith("export "):
            line = line[len("export ") :].lstrip()

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ[key] = value

    return env_path


def ensure_required_env(var_name: str) -> str:
    value = os.environ.get(var_name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {var_name}. "
            "Check the repo root .env file."
        )
    return value


def start_autogen_studio(port: int = APP_PORT, host: str = APP_HOST) -> subprocess.Popen[str]:
    """Start AutoGen Studio in the background."""

    appdir = Path(__file__).resolve().parent / "my-app"
    cmd = ["autogenstudio", "ui", "--port", str(port), "--host", host]
    if appdir.exists():
        cmd.extend(["--appdir", str(appdir)])

    print("Starting:", " ".join(cmd))
    if appdir.exists():
        print(f"Using appdir: {appdir}")
    else:
        print("No appdir folder found; starting Studio without --appdir.")

    return subprocess.Popen(cmd)


def wait_for_http_server(port: int, host: str = "127.0.0.1", timeout: int = 60) -> None:
    """Wait until the local web server answers an HTTP request."""

    deadline = time.time() + timeout
    last_error: Exception | None = None

    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=2)
            conn.request("GET", "/")
            response = conn.getresponse()
            response.read()
            conn.close()
            if response.status < 500:
                return
        except Exception as exc:  # pragma: no cover - depends on runtime startup
            last_error = exc
        time.sleep(1)

    raise RuntimeError(
        f"AutoGen Studio did not become ready on http://{host}:{port} within {timeout} seconds."
    ) from last_error


def launch_studio_with_ngrok() -> None:
    """Launch AutoGen Studio and expose it through ngrok."""

    ngrok_authtoken = ensure_required_env("NGROK_AUTHTOKEN")

    try:
        from pyngrok import ngrok
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "pyngrok is not installed. Install it with `pip install pyngrok`."
        ) from exc

    ngrok.set_auth_token(ngrok_authtoken)
    studio_process = start_autogen_studio()

    # Give the server a moment to boot, then fail fast if it crashed.
    time.sleep(5)
    exit_code = studio_process.poll()
    if exit_code is not None:
        raise RuntimeError(
            f"AutoGen Studio exited during startup with code {exit_code}. "
            "Fix the startup traceback above before retrying."
        )

    wait_for_http_server(APP_PORT)
    public_url = ngrok.connect(APP_PORT, bind_tls=True)

    print()
    print("=" * 60)
    print("[SUCCESS] AutoGen Studio is running!")
    print(f"Local URL: http://127.0.0.1:{APP_PORT}")
    print(f"Public URL: {public_url}")
    print("=" * 60)
    print("Press Ctrl+C to stop the tunnel and server.")

    try:
        studio_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if studio_process.poll() is None:
            studio_process.terminate()
            try:
                studio_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                studio_process.kill()
        ngrok.disconnect(str(public_url))
        ngrok.kill()


async def run_team() -> None:
    """Run the notebook's Groq-backed two-agent round robin demo."""

    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.conditions import MaxMessageTermination
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_ext.models.openai import OpenAIChatCompletionClient
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Missing AutoGen packages. Install `autogen-agentchat` and "
            "`autogen-ext[openai]` to run the demo."
        ) from exc

    groq_key = ensure_required_env("GROQ_API_KEY")

    # AutoGen's OpenAI-compatible client works with Groq by overriding base_url.
    researcher_model = OpenAIChatCompletionClient(
        model=RESEARCH_MODEL,
        base_url=GROQ_BASE_URL,
        api_key=groq_key,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "structured_output": True,
            "family": "unknown",
        },
    )
    editor_model = OpenAIChatCompletionClient(
        model=EDITOR_MODEL,
        base_url=GROQ_BASE_URL,
        api_key=groq_key,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "structured_output": True,
            "family": "unknown",
        },
    )

    researcher = AssistantAgent(
        name="Researcher",
        model_client=researcher_model,
        system_message=(
            "You are an expert researcher. Provide a highly detailed summary "
            "using clear Markdown formatting."
        ),
    )
    editor = AssistantAgent(
        name="Editor",
        model_client=editor_model,
        system_message=(
            "You are a strict editor. Critique the researcher's work and "
            "optimize it for professional delivery."
        ),
    )

    team = RoundRobinGroupChat(
        participants=[researcher, editor],
        termination_condition=MaxMessageTermination(max_messages=4),
    )

    print("--- Starting Multi-Agent Session ---")
    async for message in team.run_stream(
        task="Explain why Groq LPUs provide higher throughput for LLMs than standard GPUs."
    ):
        print(f"\n[{message.source}]: {message.content}")
        print("-" * 40)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the AutoGen Groq demo or launch AutoGen Studio through ngrok."
    )
    parser.add_argument(
        "--studio",
        action="store_true",
        help="Launch AutoGen Studio and expose it through ngrok.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the Groq-backed multi-agent demo.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run both the Studio launch and the Groq demo.",
    )
    return parser


def main() -> int:
    repo_env = load_repo_env()
    if repo_env:
        print(f"Loaded environment variables from {repo_env}")
    else:
        print("Repo .env file not found; using existing environment only.")

    parser = build_parser()
    args = parser.parse_args()

    run_studio = args.studio or args.all
    run_demo = args.demo or args.all or not run_studio

    if run_studio:
        launch_studio_with_ngrok()

    if run_demo:
        asyncio.run(run_team())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
