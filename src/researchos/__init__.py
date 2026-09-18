"""ResearchOS — a Research Operating System for AI/ML science.

Layering rule (enforced by ``tests/test_layering.py``):

    models  <-  kernel  <-  {evidence, claims, literature, importer, analysis, skills, paper}  <-  agents  <-  cli

``models`` and ``kernel`` must never import ``researchos.llm``: the truth path is deterministic.
"""

__version__ = "0.1.0"
SCHEMA_VERSION = "1.0"

__all__ = ["__version__", "SCHEMA_VERSION"]
