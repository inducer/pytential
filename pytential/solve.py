from pytential.linalg.gmres import *        # noqa: F401,F403

from warnings import warn
warn(
    "pytential.solve is deprecated and will be removed in 2023. Use "
    "pytential.linalg.gmres instead.", DeprecationWarning)
