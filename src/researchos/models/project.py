"""Project identity — what ``.researchos/project.yaml`` holds."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from .common import RosModel, new_id, utcnow


class ResearchProject(RosModel):
    project_id: str = Field(default_factory=lambda: new_id("project"))
    name: str = Field(min_length=1)
    description: str = ""
    domain: str = Field(default="", description="e.g. 'AI/ML', 'mechanistic interpretability'.")
    schema_version: str = "1.0"

    root: str = Field(description="Project root (the directory that contains .researchos/).")
    created_at: datetime = Field(default_factory=utcnow)
    created_by: str = "human"

    git_remote: str | None = None
    git_commit: str | None = None
    source_roots: list[str] = Field(
        default_factory=list, description="Imported material, kept for provenance and re-import."
    )
    tags: list[str] = Field(default_factory=list)
    timezone: str = "UTC"

    #: Whether this project was created by importing existing research.
    imported: bool = False
    import_confidence: str | None = Field(
        default=None, description="HIGH | MEDIUM | LOW — how well the import recovered real state."
    )
