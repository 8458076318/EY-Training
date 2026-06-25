"""Compatibility shim for running Python from inside the `fastapiapp` folder.

When the current working directory is `fastapiapp/`, importing
`fastapiapp.app.main` normally fails because Python looks for a nested
`fastapiapp/fastapiapp` package. This module marks the current directory as a
package search location so the existing package layout still resolves.
"""

from __future__ import annotations

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent)]

