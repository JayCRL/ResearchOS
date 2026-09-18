"""Read-only HTTP surface plus the zero-build dashboard (optional: ``pip install researchos[api]``).

Importing this package requires FastAPI. Nothing else in ResearchOS imports it, so the base install
stays dependency-light.
"""

from .actions import build_actions_router
from .app import app_for_project, create_app

__all__ = ["app_for_project", "build_actions_router", "create_app"]
