"""Launch AutoGen Studio locally and expose it through ngrok.

This script replaces the notebook-style paste that was in this file with a
real, runnable Python entrypoint. It loads secrets from the repo root .env
file, prefers existing environment variables, and then starts AutoGen Studio
on port 8081 before opening an ngrok tunnel.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


APP_PORT = 8081
APP_HOST = "0.0.0.0"


def load_env_file(env_path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Existing environment variables win. This keeps the script aligned with the
    repo's .env file without requiring python-dotenv.
    """

    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key or key in os.environ:
            continue

        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {'"', "'"}
        ):
            value = value[1:-1]

        os.environ[key] = value


def ensure_required_env(var_name: str) -> str:
    value = os.environ.get(var_name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {var_name}. "
            "Check the repo root .env file."
        )
    return value


def start_autogen_studio() -> subprocess.Popen[str]:
    primary_cmd = [
        "autogenstudio",
        "ui",
        "--port",
        str(APP_PORT),
        "--host",
        APP_HOST,
    ]

    try:
        return subprocess.Popen(primary_cmd)
    except FileNotFoundError:
        fallback_cmd = [
            "python",
            "-m",
            "autogenstudio",
            "ui",
            "--port",
            str(APP_PORT),
            "--host",
            APP_HOST,
        ]
        return subprocess.Popen(fallback_cmd)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    load_env_file(repo_root / ".env")

    groq_api_key = ensure_required_env("GROQ_API_KEY")
    ngrok_authtoken = ensure_required_env("NGROK_AUTHTOKEN")

    # Keep the selected keys visible without printing secrets.
    print("Using environment variables from repo root .env")
    print(f"GROQ_API_KEY: {'set' if groq_api_key else 'missing'}")
    print(f"NGROK_AUTHTOKEN: {'set' if ngrok_authtoken else 'missing'}")

    try:
        from pyngrok import ngrok
    except ImportError as exc:
        raise RuntimeError(
            "pyngrok is not installed. Install it with `pip install pyngrok`."
        ) from exc

    ngrok.set_auth_token(ngrok_authtoken)

    try:
        studio_process = start_autogen_studio()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "autogenstudio is not installed or is not on PATH. "
            "Install it with `pip install autogenstudio`."
        ) from exc

    # Give the server a moment to boot before tunneling.
    time.sleep(5)

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
        if 'studio_process' in locals() and studio_process.poll() is None:
            studio_process.terminate()
            try:
                studio_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                studio_process.kill()
        if 'public_url' in locals():
            ngrok.disconnect(str(public_url))
        ngrok.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
