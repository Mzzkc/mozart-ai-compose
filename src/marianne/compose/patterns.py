"""Pattern expander — expands named patterns into sheet sequences.

Patterns from the Rosetta corpus (Cathedral Construction, Composting Cascade,
Fan-out + Synthesis, etc.) are available as named patterns the compiler can
compose into sheet sequences.

This is the extensibility point: a pattern library that the compiler draws
from to produce sheets with the right cognitive structure for the task.

Two expansion modes:
  * **Prompt extension** — ``expand()`` merges per-phase prompt additions
    into the agent cycle.  Lightweight; enriches existing sheet structure.
  * **Stage expansion** — ``expand_stages()`` returns a concrete stage
    sequence with purposes, instrument guidance, and validation shapes.
    Structural; defines what the sheet arrangement looks like.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatternStage:
    """A single stage within an expanded pattern.

    Attributes:
        name: Stage identifier (e.g. ``recon``, ``synthesize``).
        sheets: Number of sheets — an ``int`` for fixed stages,
            or a string like ``"fan_out(6)"`` for parameterised fan-outs.
        purpose: What this stage accomplishes.
        instrument_guidance: Advice on which instrument fits the stage.
        fallback_friendly: Whether cheaper fallback instruments are viable.
        artifacts: Tuple of expected output file paths/globs.
    """

    name: str
    sheets: int | str
    purpose: str
    instrument_guidance: str
    fallback_friendly: bool
    artifacts: tuple[str, ...]


# ---------------------------------------------------------------------------
# Built-in prompt-extension patterns (backward compatible)
# ---------------------------------------------------------------------------

BUILTIN_PATTERNS: dict[str, dict[str, Any]] = {
    "cathedral-construction": {
        "description": "Long-term structural building with foundation-first approach",
        "phases": ["foundation", "structure", "refinement", "decoration"],
        "sheet_modifiers": {
            "work": {
                "prompt_extension": "Build from the foundation up. Structural integrity first.",
            },
            "inspect": {"prompt_extension": "Check load-bearing walls. Test boundaries."},
        },
    },
    "composting-cascade": {
        "description": "Iterative refinement through cycles of decomposition and growth",
        "phases": ["decompose", "ferment", "integrate", "grow"],
        "sheet_modifiers": {
            "recon": {"prompt_extension": "What raw material exists? What can be composted?"},
            "consolidate": {
                "prompt_extension": "Turn findings into soil. What grew from last cycle?"
            },
        },
    },
    "soil-maturity-index": {
        "description": "Developmental measurement and growth tracking",
        "phases": ["sample", "measure", "assess", "project"],
        "sheet_modifiers": {
            "aar": {
                "prompt_extension": "Measure developmental soil. What matured? What's still raw?"
            },
            "reflect": {
                "prompt_extension": "Track growth trajectory. Where is the growth edge?"
            },
        },
    },
    "boundary-trace": {
        "description": "Systematic boundary identification and integrity check",
        "phases": ["survey", "trace", "test", "document"],
        "sheet_modifiers": {
            "recon": {"prompt_extension": "Survey all boundaries — package, module, API, data."},
            "inspect": {
                "prompt_extension": "Test boundaries under pressure. Where do they leak?"
            },
        },
    },
    "forge-cycle": {
        "description": "Implementation craftsmanship with heat-test-cool rhythm",
        "phases": ["heat", "shape", "quench", "temper"],
        "sheet_modifiers": {
            "work": {
                "prompt_extension": "Shape the metal while hot. Clean strikes. No excess."
            },
            "inspect": {"prompt_extension": "Test the temper. Does it hold under stress?"},
        },
    },
    # --- Agent-cycle patterns (also available as stage expansions) ----------
    "fan-out-synthesis": {
        "description": (
            "Split work into parallel independent streams, merge in a synthesis "
            "stage — MapReduce for AI orchestration"
        ),
        "phases": ["prepare", "analyze", "synthesize"],
        "sheet_modifiers": {
            "work": {
                "prompt_extension": (
                    "Decompose the problem into independent parallel streams. "
                    "Each stream analyses a separate facet."
                ),
            },
            "consolidate": {
                "prompt_extension": (
                    "Synthesise all parallel outputs into a unified result "
                    "addressing cross-cutting concerns."
                ),
            },
        },
    },
    "the-tool-chain": {
        "description": (
            "CLI instruments handle deterministic work; AI instruments appear "
            "only at planning, triage, and interpretation points"
        ),
        "phases": ["plan", "fetch", "clean", "analyze", "interpret"],
        "sheet_modifiers": {
            "plan": {
                "prompt_extension": (
                    "Design a processing plan. Identify which steps are "
                    "deterministic (CLI) and which need judgment (AI)."
                ),
            },
            "work": {
                "prompt_extension": (
                    "Route deterministic transforms to CLI instruments. "
                    "Reserve AI for interpretation and triage."
                ),
            },
        },
    },
    "reconnaissance-pull": {
        "description": (
            "Cheap exploration before committing to a plan — discover the "
            "landscape, then plan, then execute"
        ),
        "phases": ["recon", "plan", "execute"],
        "sheet_modifiers": {
            "recon": {
                "prompt_extension": (
                    "Survey the landscape: structure, complexity, risks, "
                    "recommended approach. Write recon-report.md."
                ),
            },
            "plan": {
                "prompt_extension": (
                    "Read recon findings. Synthesise a detailed execution plan "
                    "informed by discovered landscape."
                ),
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Built-in stage definitions (structural expansion data from Rosetta corpus)
# ---------------------------------------------------------------------------

_BUILTIN_STAGE_DEFS: dict[str, dict[str, Any]] = {
    "reconnaissance-pull": {
        "fan_out": {},
        "stages": [
            {
                "name": "recon",
                "sheets": 1,
                "instrument_guidance": (
                    "sonnet — balanced cost and capability; sufficient for "
                    "landscape discovery without deep reasoning"
                ),
                "fallback_friendly": True,
                "purpose": (
                    "Discover and document the landscape of the input: "
                    "structure, complexity, and risks."
                ),
                "artifacts": ["recon-report.md"],
            },
            {
                "name": "plan",
                "sheets": 1,
                "instrument_guidance": (
                    "score-author's choice — planning complexity depends on "
                    "task and landscape complexity; stronger instruments "
                    "benefit from comprehensive recon"
                ),
                "fallback_friendly": True,
                "purpose": (
                    "Analyze reconnaissance findings and synthesize a "
                    "detailed execution plan."
                ),
                "artifacts": ["execution-plan.md"],
            },
            {
                "name": "execute",
                "sheets": 1,
                "instrument_guidance": (
                    "score-author's choice — execution capability must match "
                    "task requirements; recon and plan inform instrument "
                    "selection"
                ),
                "fallback_friendly": False,
                "purpose": "Execute the work as specified in the execution plan.",
                "artifacts": [],
            },
        ],
    },
    "fan-out-synthesis": {
        "fan_out": {"analyze": 6},
        "stages": [
            {
                "name": "prepare",
                "sheets": 1,
                "instrument_guidance": (
                    "score-author's choice — sonnet or opus for complex "
                    "problem decomposition requiring clear scope definition; "
                    "haiku may suffice for simple scoping tasks"
                ),
                "fallback_friendly": True,
                "purpose": (
                    "Define scope and shared context for parallel analysis."
                ),
                "artifacts": ["scope.md"],
            },
            {
                "name": "analyze",
                "sheets": "fan_out(6)",
                "instrument_guidance": (
                    "score-author's choice — instrument capability must match "
                    "the analysis complexity; sonnet recommended for code "
                    "review or detailed analysis; haiku suffices for simple "
                    "classification or data extraction"
                ),
                "fallback_friendly": True,
                "purpose": (
                    "Analyze independent facets in parallel, each producing "
                    "separate findings."
                ),
                "artifacts": ["analysis-{{ instance_id }}.md"],
            },
            {
                "name": "synthesize",
                "sheets": 1,
                "instrument_guidance": (
                    "sonnet or opus — synthesis requires finding cross-cutting "
                    "themes and integrating diverse perspectives, higher-order "
                    "reasoning beyond what produced individual analyses; "
                    "fallback to cheaper instruments risks mere concatenation"
                ),
                "fallback_friendly": False,
                "purpose": (
                    "Read all parallel outputs and produce unified result "
                    "addressing cross-cutting concerns."
                ),
                "artifacts": [],
            },
        ],
    },
    "the-tool-chain": {
        "fan_out": {},
        "stages": [
            {
                "name": "plan",
                "sheets": 1,
                "instrument_guidance": (
                    "claude — synthesizes processing plan from input "
                    "characteristics; judgment needed"
                ),
                "fallback_friendly": False,
                "purpose": "Read input and design a processing plan.",
                "artifacts": ["processing-plan.yaml"],
            },
            {
                "name": "fetch",
                "sheets": 1,
                "instrument_guidance": (
                    "cli — deterministic API call; no AI needed"
                ),
                "fallback_friendly": True,
                "purpose": "Fetch raw data from the source API.",
                "artifacts": ["raw.csv"],
            },
            {
                "name": "clean",
                "sheets": 1,
                "instrument_guidance": (
                    "cli — user-supplied Python script; data transformation "
                    "is deterministic"
                ),
                "fallback_friendly": True,
                "purpose": "Clean and normalize the raw data.",
                "artifacts": ["clean.csv"],
            },
            {
                "name": "analyze",
                "sheets": 1,
                "instrument_guidance": (
                    "cli — user-supplied Python script; analysis logic is "
                    "deterministic"
                ),
                "fallback_friendly": True,
                "purpose": (
                    "Analyze cleaned data and produce structured report."
                ),
                "artifacts": ["report.md"],
            },
            {
                "name": "interpret",
                "sheets": 1,
                "instrument_guidance": (
                    "claude — interprets numerical results and synthesizes "
                    "recommendations; judgment needed"
                ),
                "fallback_friendly": False,
                "purpose": (
                    "Interpret analysis results and write executive summary "
                    "with recommendations."
                ),
                "artifacts": [],
            },
        ],
    },
    "cathedral-construction": {
        "fan_out": {},
        "stages": [
            {
                "name": "plan-iteration",
                "sheets": 1,
                "instrument_guidance": (
                    "score-author's choice — planning benefits from strong "
                    "reasoning (sonnet/opus recommended), but the iteration "
                    "cycle provides correction opportunities so mid-tier "
                    "instruments are viable"
                ),
                "fallback_friendly": True,
                "purpose": (
                    "Read current state and plan what to add this iteration."
                ),
                "artifacts": [],
            },
            {
                "name": "build",
                "sheets": 1,
                "instrument_guidance": (
                    "score-author's choice — depends entirely on what is "
                    "being built (code, documentation, analysis); match "
                    "instrument capability to the construction task complexity"
                ),
                "fallback_friendly": True,
                "purpose": "Execute the plan and add to the cathedral.",
                "artifacts": ["cathedral/**"],
            },
            {
                "name": "inspect",
                "sheets": 1,
                "instrument_guidance": (
                    "score-author's choice — review and critique work; "
                    "sonnet-level reasoning typically sufficient since this "
                    "is evaluation rather than primary construction"
                ),
                "fallback_friendly": True,
                "purpose": "Review what was built and write inspection report.",
                "artifacts": ["inspection-report.md"],
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Rosetta frontmatter parser
# ---------------------------------------------------------------------------


def _parse_frontmatter(text: str) -> dict[str, Any]:
    """Parse YAML frontmatter from a ``---``-delimited markdown file.

    Returns an empty dict if the frontmatter is missing or unparseable.
    """
    text = text.lstrip()
    if not text.startswith("---"):
        return {}
    # Find the closing ---
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    fm_text = text[3:end].strip()
    try:
        parsed = yaml.safe_load(fm_text)
        return parsed if isinstance(parsed, dict) else {}
    except yaml.YAMLError:
        return {}


def _slugify(name: str) -> str:
    """Turn a human-readable pattern name into a slug.

    ``"Fan-out + Synthesis"`` → ``"fan-out-synthesis"``
    ``"The Tool Chain"`` → ``"the-tool-chain"``
    """
    slug = name.lower().strip()
    # Replace non-alphanumeric (except hyphen) with hyphen
    result: list[str] = []
    for ch in slug:
        if ch.isalnum() or ch == "-":
            result.append(ch)
        elif ch in (" ", "+", "_"):
            if result and result[-1] != "-":
                result.append("-")
        # else: drop
    return "".join(result).strip("-")


# ---------------------------------------------------------------------------
# PatternExpander
# ---------------------------------------------------------------------------


class PatternExpander:
    """Expands named patterns into sheet sequence modifiers.

    Patterns inject additional prompt context and structural guidance
    into specific phases of the agent cycle.  Two expansion modes:

    * ``expand()`` — prompt extensions (per-phase text additions).
    * ``expand_stages()`` — concrete stage sequence with purposes,
      instrument guidance, and validation shapes.
    """

    def __init__(self, custom_patterns: dict[str, dict[str, Any]] | None = None) -> None:
        """Initialize with optional custom patterns.

        Args:
            custom_patterns: Additional patterns to register beyond builtins.
        """
        self.patterns: dict[str, dict[str, Any]] = dict(BUILTIN_PATTERNS)
        self._stage_defs: dict[str, dict[str, Any]] = dict(_BUILTIN_STAGE_DEFS)
        if custom_patterns:
            self.patterns.update(custom_patterns)

    # ------------------------------------------------------------------
    # Prompt extension mode (backward-compatible)
    # ------------------------------------------------------------------

    def expand(
        self,
        pattern_names: list[str],
        agent_def: dict[str, Any],
    ) -> dict[str, Any]:
        """Expand named patterns into per-phase prompt extensions.

        Args:
            pattern_names: List of pattern names to apply.
            agent_def: Agent definition for context.

        Returns:
            Dict with keys:
                ``prompt_extensions``: dict[str, str] — per-phase prompt additions
                ``applied_patterns``: list[str] — names of patterns actually applied

        Raises:
            ValueError: If a pattern name is not recognised.
        """
        prompt_extensions: dict[str, str] = {}
        applied: list[str] = []

        for name in pattern_names:
            pattern = self.patterns.get(name)
            if pattern is None:
                raise ValueError(
                    f"Unknown pattern '{name}'. "
                    f"Available: {sorted(self.patterns.keys())}"
                )

            modifiers = pattern.get("sheet_modifiers", {})
            for phase, mods in modifiers.items():
                if isinstance(mods, dict):
                    extension = mods.get("prompt_extension", "")
                    if extension:
                        if phase in prompt_extensions:
                            prompt_extensions[phase] += f"\n{extension}"
                        else:
                            prompt_extensions[phase] = extension

            applied.append(name)
            _logger.debug(
                "Applied pattern '%s' to agent '%s'",
                name, agent_def.get("name"),
            )

        return {
            "prompt_extensions": prompt_extensions,
            "applied_patterns": applied,
        }

    # ------------------------------------------------------------------
    # Stage expansion mode
    # ------------------------------------------------------------------

    def expand_stages(
        self,
        pattern_name: str,
        params: dict[str, Any] | None = None,
    ) -> list[PatternStage]:
        """Expand a pattern into a concrete stage sequence.

        Args:
            pattern_name: Name of the pattern to expand.
            params: Optional parameters to customise expansion.
                Supported keys:
                    ``fan_out`` — dict mapping stage name to fan-out width
                        (overrides the pattern default).

        Returns:
            Ordered list of :class:`PatternStage` instances.

        Raises:
            ValueError: If the pattern is unknown or has no stage definition.
        """
        stage_def = self._stage_defs.get(pattern_name)
        if stage_def is None:
            available = sorted(self._stage_defs.keys())
            raise ValueError(
                f"Unknown pattern '{pattern_name}' (or pattern has no stage "
                f"definition). Available: {available}"
            )

        # Merge fan_out overrides
        default_fan_out: dict[str, int] = dict(stage_def.get("fan_out", {}))
        if params and "fan_out" in params:
            default_fan_out.update(params["fan_out"])

        stages: list[PatternStage] = []
        for raw in stage_def["stages"]:
            sheets_val = raw["sheets"]

            # Apply fan_out parameter override if applicable
            if isinstance(sheets_val, str) and sheets_val.startswith("fan_out("):
                stage_name = raw["name"]
                if stage_name in default_fan_out:
                    sheets_val = f"fan_out({default_fan_out[stage_name]})"
            elif isinstance(sheets_val, int) and raw["name"] in default_fan_out:
                sheets_val = f"fan_out({default_fan_out[raw['name']]})"

            artifacts = raw.get("artifacts", [])
            stages.append(
                PatternStage(
                    name=raw["name"],
                    sheets=sheets_val,
                    purpose=raw.get("purpose", ""),
                    instrument_guidance=raw.get("instrument_guidance", ""),
                    fallback_friendly=raw.get("fallback_friendly", True),
                    artifacts=tuple(artifacts),
                )
            )

        _logger.debug(
            "Expanded pattern '%s' into %d stages", pattern_name, len(stages),
        )
        return stages

    def expand_stages_with_validations(
        self,
        pattern_name: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Expand a pattern into stages and generate validation shapes.

        Validation shapes are derived from each stage's declared artifacts.
        Each artifact produces a ``file_exists`` validation template.

        Args:
            pattern_name: Name of the pattern to expand.
            params: Optional expansion parameters (see :meth:`expand_stages`).

        Returns:
            Dict with keys:
                ``stages``: list[:class:`PatternStage`]
                ``validations``: list[dict] — validation rule shapes
        """
        stages = self.expand_stages(pattern_name, params=params)
        validations: list[dict[str, str]] = []

        for stage in stages:
            for artifact in stage.artifacts:
                validations.append({
                    "type": "file_exists",
                    "path": f"{{{{workspace}}}}/{artifact}",
                    "stage": stage.name,
                })

        return {"stages": stages, "validations": validations}

    # ------------------------------------------------------------------
    # Pattern listing / lookup
    # ------------------------------------------------------------------

    def list_patterns(self) -> list[dict[str, str]]:
        """List all available patterns with descriptions.

        Returns:
            List of dicts with ``name`` and ``description`` keys.
        """
        return [
            {"name": name, "description": pattern.get("description", "")}
            for name, pattern in sorted(self.patterns.items())
        ]

    def get_pattern(self, name: str) -> dict[str, Any] | None:
        """Get a pattern definition by name.

        Returns:
            Pattern dict or ``None`` if not found.
        """
        return self.patterns.get(name)

    # ------------------------------------------------------------------
    # Rosetta corpus loading
    # ------------------------------------------------------------------

    def load_rosetta_corpus(
        self,
        corpus_dir: str | Path,
    ) -> dict[str, dict[str, Any]]:
        """Load patterns from a Rosetta corpus directory.

        Reads ``*.md`` files in *corpus_dir*, parses YAML frontmatter,
        and registers patterns that contain a ``stages`` field.

        Args:
            corpus_dir: Directory containing pattern markdown files.

        Returns:
            Dict of loaded pattern slug → pattern data.  Only patterns
            with parseable frontmatter and a ``name`` field are returned.
        """
        corpus_dir = Path(corpus_dir)
        if not corpus_dir.is_dir():
            _logger.warning("Rosetta corpus directory not found: %s", corpus_dir)
            return {}

        loaded: dict[str, dict[str, Any]] = {}

        for md_file in sorted(corpus_dir.glob("*.md")):
            try:
                text = md_file.read_text(encoding="utf-8")
            except OSError:
                _logger.warning("Could not read corpus file: %s", md_file)
                continue

            fm = _parse_frontmatter(text)
            if not fm or "name" not in fm:
                continue

            slug = _slugify(fm["name"])

            # Build a pattern entry
            pattern_entry: dict[str, Any] = {
                "description": fm.get("problem", fm.get("name", "")),
                "scale": fm.get("scale", ""),
                "status": fm.get("status", ""),
                "signals": fm.get("signals", []),
                "composes_with": fm.get("composes_with", []),
            }

            # If the pattern has stages, register them
            raw_stages = fm.get("stages", [])
            if raw_stages:
                pattern_entry["stages"] = raw_stages
                # Also register as a stage definition
                self._stage_defs.setdefault(slug, {
                    "fan_out": fm.get("fan_out", {}),
                    "stages": raw_stages,
                })

            # Generate sheet_modifiers from stages for prompt-extension compat
            if raw_stages and "sheet_modifiers" not in pattern_entry:
                modifiers: dict[str, dict[str, str]] = {}
                for stage in raw_stages:
                    stage_name = stage.get("name", "")
                    purpose = stage.get("purpose", "")
                    if stage_name and purpose:
                        modifiers[stage_name] = {"prompt_extension": purpose}
                if modifiers:
                    pattern_entry["sheet_modifiers"] = modifiers

            self.patterns.setdefault(slug, pattern_entry)
            loaded[slug] = pattern_entry

        _logger.info(
            "Loaded %d patterns from Rosetta corpus: %s",
            len(loaded), corpus_dir,
        )
        return loaded
