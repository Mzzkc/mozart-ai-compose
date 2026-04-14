"""Composition compiler — turns semantic agent definitions into Mozart scores.

The compiler takes high-level descriptions (agent identities, patterns,
techniques, instrument assignments) and produces complete Mozart score YAML.

Modules:
    identity    — Seed agent identity stores (L1-L4)
    sheets      — Compose sheet structures with parallel phases
    techniques  — Wire technique manifests into cadenza context
    instruments — Resolve per-agent per-sheet instrument assignments
    validations — Generate per-sheet validation rules
    patterns    — Expand named patterns into sheet sequences
    fleet       — Generate fleet configs (concert-of-concerts)
    pipeline    — Top-level compilation pipeline
"""

from __future__ import annotations

from marianne.compose.fleet import FleetGenerator
from marianne.compose.identity import IdentitySeeder
from marianne.compose.instruments import InstrumentResolver
from marianne.compose.patterns import PatternExpander, PatternStage
from marianne.compose.pipeline import CompilationPipeline
from marianne.compose.sheets import SheetComposer
from marianne.compose.techniques import TechniqueWirer
from marianne.compose.validations import ValidationGenerator

__all__ = [
    "CompilationPipeline",
    "IdentitySeeder",
    "SheetComposer",
    "TechniqueWirer",
    "InstrumentResolver",
    "ValidationGenerator",
    "PatternExpander",
    "PatternStage",
    "FleetGenerator",
]
