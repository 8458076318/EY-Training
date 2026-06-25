"""Make the project importable when Python is started inside `fastapiapp`.

If the current working directory is `fastapiapp/`, Python's import path only
sees that folder, not its parent. Adding the parent directory lets imports like
`fastapiapp.app.main` resolve correctly from both places.
"""

from __future__ import annotations

import sys
from pathlib import Path


current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

