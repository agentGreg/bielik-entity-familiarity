"""Dataset v2 subpackage (paper 2, RQ2): popularity-graded entity dataset.

Modules:
  templates  -- PL/EN prompt templates + neutral stem pair.
  harvest    -- Wikidata SPARQL + pl.wiki signal harvest (REAL entities).
  fabricate  -- token-matched, non-existent fabricated anchors.
"""

from . import fabricate, harvest, templates  # noqa: F401

__all__ = ["templates", "harvest", "fabricate"]
