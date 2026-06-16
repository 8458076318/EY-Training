"""Small OpenAI API key checker for the repo-root .env file.

Usage:
    python check_openai_key.py
    python check_openai_key.py --model omni-moderation-latest

What it does:
    - Loads OPENAI_API_KEY from C:\\Training\\AI-ML-Training-Projects\\.env
    - Creates an OpenAI client
    - Makes one harmless moderation request to verify the key works end to end

It never prints the secret value.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"


def load_env_file(path: Path) -> bool:
    """Load simple KEY=VALUE pairs from a .env file into os.environ."""
    if not path.exists():
        return False

    loaded = False
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded = True
    return loaded


def summarize_key(value: str | None) -> str:
    """Return a non-sensitive summary of the API key."""
    if not value:
        return "<missing>"
    if len(value) <= 12:
        return f"<len={len(value)}>"
    return f"{value[:7]}...{value[-4:]} (len={len(value)})"


def run_check(model: str, probe_text: str) -> int:
    loaded_env = load_env_file(ENV_PATH)
    print(f"Loaded .env: {loaded_env} ({ENV_PATH})")

    api_key = os.getenv("OPENAI_API_KEY")
    print(f"OPENAI_API_KEY: {summarize_key(api_key)}")

    if not api_key:
        print("FAIL: OPENAI_API_KEY is not set.")
        return 2

    try:
        from openai import OpenAI
    except Exception as exc:
        print(f"FAIL: could not import openai client: {exc}")
        return 3

    try:
        client = OpenAI()
    except Exception as exc:
        print(f"FAIL: OpenAI client initialization failed: {exc}")
        return 4

    try:
        response = client.moderations.create(model=model, input=probe_text)
    except Exception as exc:
        print(f"FAIL: moderation request failed: {exc}")
        return 5

    result = response.results[0]
    category_scores = getattr(result, "category_scores", None)
    categories = getattr(result, "categories", None)

    if hasattr(category_scores, "model_dump"):
        category_scores = category_scores.model_dump()
    elif category_scores is not None and not isinstance(category_scores, dict):
        category_scores = dict(category_scores)

    if hasattr(categories, "model_dump"):
        categories = categories.model_dump()
    elif categories is not None and not isinstance(categories, dict):
        categories = dict(categories)

    print("PASS: moderation request succeeded.")
    print(f"Model: {response.model}")
    print(f"Flagged: {bool(result.flagged)}")

    if category_scores:
        top_category = max(category_scores, key=category_scores.get)
        print(f"Top score: {top_category} = {category_scores[top_category]:.3f}")

    if categories:
        flagged_categories = [name for name, is_flagged in categories.items() if is_flagged]
        print(f"Flagged categories: {', '.join(flagged_categories) if flagged_categories else '<none>'}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check OPENAI_API_KEY and make a live moderation call.")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODERATION_MODEL", "omni-moderation-latest"),
        help="Moderation model to call. Default: omni-moderation-latest",
    )
    parser.add_argument(
        "--text",
        default="This is a harmless API connectivity check.",
        help="Text to send to the moderation API.",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    return run_check(args.model, args.text)


if __name__ == "__main__":
    raise SystemExit(main())
