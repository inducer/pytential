from __future__ import annotations

from warnings import warn

from pytential.linalg.gmres import *  # noqa: F403


warn(
    "pytential.solve is deprecated and will be removed in 2023. Use "
    "pytential.linalg.gmres instead.", DeprecationWarning,
    stacklevel=1)
