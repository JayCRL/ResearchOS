"""Read-only HTTP surface (optional dependency: ``pip install researchos[api]``).

Importing this package requires FastAPI. Nothing else in ResearchOS imports it, so the base install
stays dependency-light.
"""

from .app import app_for_project, create_app

__all__ = ["app_for_project", "create_app"]
