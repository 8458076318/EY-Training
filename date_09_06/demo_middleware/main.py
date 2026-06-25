from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    import uvicorn

    uvicorn.run("demo_middleware.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
