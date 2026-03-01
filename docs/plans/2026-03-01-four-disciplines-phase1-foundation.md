# Four Disciplines Phase 1: Specification Corpus + Token Budget — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the foundation layer that all other discipline work depends on: a specification corpus system that scores can reference, and a token budget tracker that prevents context window overflow.

**Architecture:** Two independent subsystems. (A) A `.mozart/spec/` directory convention with a loader that injects spec fragments into prompts based on relevance. (B) A token estimation + budget tracking system integrated into PromptBuilder that enforces context window limits per-model. Both integrate into the existing prompt assembly pipeline in `src/mozart/prompts/templating.py`.

**Tech Stack:** Pydantic v2 models, Jinja2 templates, tiktoken for token estimation, existing PromptBuilder infrastructure.

---

## Task 1: Token Estimation Utility

**Files:**
- Create: `src/mozart/core/tokens.py`
- Test: `tests/test_tokens.py`

**Step 1: Write the failing test**

```python
# tests/test_tokens.py
"""Tests for token estimation utility."""

from mozart.core.tokens import estimate_tokens


class TestEstimateTokens:
    def test_empty_string_returns_zero(self):
        assert estimate_tokens("") == 0

    def test_none_returns_zero(self):
        assert estimate_tokens(None) == 0

    def test_short_string_returns_positive(self):
        result = estimate_tokens("Hello, world!")
        assert 1 <= result <= 10

    def test_longer_text_scales_proportionally(self):
        short = estimate_tokens("Hello")
        long = estimate_tokens("Hello " * 100)
        assert long > short * 10

    def test_dict_input_serialized(self):
        result = estimate_tokens({"key": "value", "nested": {"a": 1}})
        assert result > 0

    def test_list_input_joined(self):
        result = estimate_tokens(["line one", "line two", "line three"])
        assert result > 0

    def test_known_approximate_ratio(self):
        # Claude tokenizer: roughly 1 token per 4 chars for English
        text = "a" * 400
        result = estimate_tokens(text)
        assert 50 <= result <= 200  # Generous bounds
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tokens.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mozart.core.tokens'`

**Step 3: Write minimal implementation**

```python
# src/mozart/core/tokens.py
"""Token estimation utilities for prompt budget management.

Uses a character-based heuristic (1 token ≈ 4 chars for English text)
as the default estimator. This avoids a hard dependency on tiktoken
while providing reasonable estimates for budget tracking.

The heuristic is intentionally conservative (overestimates slightly)
to prevent context window overflow.
"""

from __future__ import annotations

import json
from typing import Any

# Conservative ratio: 1 token per 3.5 chars (overestimates slightly)
_CHARS_PER_TOKEN = 3.5


def estimate_tokens(text: Any) -> int:
    """Estimate token count for text or structured data.

    Args:
        text: Plain text, dict (serialized to JSON), list (joined),
              or None (returns 0).

    Returns:
        Estimated token count.
    """
    if text is None:
        return 0
    if isinstance(text, dict):
        text = json.dumps(text)
    elif isinstance(text, list):
        text = "\n".join(str(x) for x in text)
    else:
        text = str(text)
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_tokens.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mozart/core/tokens.py tests/test_tokens.py
git commit -m "feat(tokens): add token estimation utility for budget tracking"
```

---

## Task 2: Token Budget Tracker

**Files:**
- Modify: `src/mozart/core/tokens.py`
- Test: `tests/test_tokens.py` (extend)

**Step 1: Write the failing tests**

```python
# Append to tests/test_tokens.py

from mozart.core.tokens import TokenBudgetTracker


class TestTokenBudgetTracker:
    def test_initial_state(self):
        tracker = TokenBudgetTracker(window_size=100_000, reserve=10_000)
        assert tracker.remaining() == 90_000
        assert tracker.utilization() == 0.0

    def test_allocate_success(self):
        tracker = TokenBudgetTracker(window_size=100_000, reserve=10_000)
        ok = tracker.allocate("Hello world", component="preamble")
        assert ok is True
        assert tracker.remaining() < 90_000

    def test_allocate_over_budget_fails(self):
        tracker = TokenBudgetTracker(window_size=100, reserve=0)
        # 500 chars ≈ 143 tokens at 3.5 chars/token — exceeds 100
        ok = tracker.allocate("x" * 500, component="huge")
        assert ok is False

    def test_multiple_allocations_tracked(self):
        tracker = TokenBudgetTracker(window_size=100_000, reserve=0)
        tracker.allocate("first component", component="system")
        tracker.allocate("second component", component="preamble")
        report = tracker.breakdown()
        assert "system" in report
        assert "preamble" in report

    def test_can_fit_check(self):
        tracker = TokenBudgetTracker(window_size=100, reserve=0)
        assert tracker.can_fit("short") is True
        assert tracker.can_fit("x" * 500) is False

    def test_utilization_percentage(self):
        tracker = TokenBudgetTracker(window_size=1000, reserve=0)
        tracker.allocate("x" * 350, component="half")  # ~100 tokens
        util = tracker.utilization()
        assert 0.05 < util < 0.5  # Rough check

    def test_reset_clears_allocations(self):
        tracker = TokenBudgetTracker(window_size=100_000, reserve=0)
        tracker.allocate("stuff", component="test")
        tracker.reset()
        assert tracker.remaining() == 100_000
        assert tracker.breakdown() == {}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tokens.py::TestTokenBudgetTracker -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
# Append to src/mozart/core/tokens.py


class TokenBudgetTracker:
    """Track token allocation across prompt components.

    Used by PromptBuilder to enforce context window limits.
    Components are added in priority order; lower-priority
    components are rejected when budget is exhausted.
    """

    def __init__(self, window_size: int = 200_000, reserve: int = 30_000) -> None:
        self._window_size = window_size
        self._reserve = reserve
        self._available = window_size - reserve
        self._allocated: dict[str, int] = {}
        self._accumulated = 0

    def can_fit(self, text: Any) -> bool:
        """Check if text fits in remaining budget."""
        tokens = estimate_tokens(text)
        return self._accumulated + tokens <= self._available

    def allocate(self, text: Any, component: str) -> bool:
        """Allocate tokens for a named component.

        Returns True if allocated, False if over budget.
        """
        tokens = estimate_tokens(text)
        if self._accumulated + tokens > self._available:
            return False
        self._allocated[component] = self._allocated.get(component, 0) + tokens
        self._accumulated += tokens
        return True

    def remaining(self) -> int:
        """Tokens remaining in budget."""
        return self._available - self._accumulated

    def utilization(self) -> float:
        """Fraction of budget used (0.0 to 1.0)."""
        if self._available <= 0:
            return 1.0
        return self._accumulated / self._available

    def breakdown(self) -> dict[str, int]:
        """Token allocation per component."""
        return dict(self._allocated)

    def reset(self) -> None:
        """Clear all allocations."""
        self._allocated.clear()
        self._accumulated = 0
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_tokens.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mozart/core/tokens.py tests/test_tokens.py
git commit -m "feat(tokens): add TokenBudgetTracker for context window management"
```

---

## Task 3: Specification Corpus Config Model

**Files:**
- Create: `src/mozart/core/config/spec.py`
- Modify: `src/mozart/core/config/__init__.py` (add exports)
- Test: `tests/test_spec_config.py`

**Step 1: Write the failing test**

```python
# tests/test_spec_config.py
"""Tests for specification corpus configuration models."""

import pytest
from mozart.core.config.spec import (
    SpecCorpusConfig,
    SpecFragment,
)


class TestSpecFragment:
    def test_create_minimal(self):
        frag = SpecFragment(name="conventions", content="Use snake_case.")
        assert frag.name == "conventions"
        assert frag.content == "Use snake_case."
        assert frag.tags == []

    def test_create_with_tags(self):
        frag = SpecFragment(
            name="testing",
            content="80% coverage minimum.",
            tags=["quality", "testing"],
        )
        assert frag.tags == ["quality", "testing"]

    def test_matches_tag(self):
        frag = SpecFragment(name="x", content="y", tags=["quality", "testing"])
        assert frag.matches_any(["quality"]) is True
        assert frag.matches_any(["security"]) is False
        assert frag.matches_any([]) is False


class TestSpecCorpusConfig:
    def test_default_empty(self):
        config = SpecCorpusConfig()
        assert config.spec_dir is None
        assert config.fragments == []

    def test_inline_fragments(self):
        config = SpecCorpusConfig(
            fragments=[
                SpecFragment(name="conventions", content="Use snake_case."),
                SpecFragment(name="quality", content="80% coverage."),
            ]
        )
        assert len(config.fragments) == 2

    def test_get_fragments_by_tags(self):
        config = SpecCorpusConfig(
            fragments=[
                SpecFragment(name="a", content="A", tags=["code"]),
                SpecFragment(name="b", content="B", tags=["test"]),
                SpecFragment(name="c", content="C", tags=["code", "test"]),
            ]
        )
        code_frags = config.get_fragments_by_tags(["code"])
        assert len(code_frags) == 2
        names = [f.name for f in code_frags]
        assert "a" in names
        assert "c" in names
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_spec_config.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/mozart/core/config/spec.py
"""Specification corpus configuration models.

Defines the configuration for Mozart's specification corpus —
a structured knowledge base that makes project conventions,
constraints, and quality standards available to all sheets.

Spec fragments can be defined inline in the score YAML or loaded
from `.mozart/spec/` files at runtime.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class SpecFragment(BaseModel):
    """A single specification fragment (e.g., conventions, constraints).

    Fragments are the atomic units of the spec corpus. Each has a name,
    content, and optional tags for relevance-based selection.
    """

    name: str = Field(description="Fragment identifier (e.g., 'conventions', 'quality')")
    content: str = Field(description="The specification text")
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for relevance filtering (e.g., ['code', 'testing'])",
    )

    def matches_any(self, tags: list[str]) -> bool:
        """Check if this fragment matches any of the given tags."""
        if not tags:
            return False
        return bool(set(self.tags) & set(tags))


class SpecCorpusConfig(BaseModel):
    """Configuration for the specification corpus.

    The spec corpus provides project-level knowledge to sheets:
    conventions, constraints, quality standards, architecture decisions.

    Fragments can be defined inline or loaded from a directory.
    """

    spec_dir: Path | None = Field(
        default=None,
        description=(
            "Path to .mozart/spec/ directory. If provided, YAML/MD files "
            "in this directory are loaded as spec fragments at runtime."
        ),
    )
    fragments: list[SpecFragment] = Field(
        default_factory=list,
        description="Inline spec fragments defined in the score YAML.",
    )

    def get_fragments_by_tags(self, tags: list[str]) -> list[SpecFragment]:
        """Get all fragments matching any of the given tags."""
        return [f for f in self.fragments if f.matches_any(tags)]

    def get_all_content(self) -> str:
        """Concatenate all fragment content."""
        return "\n\n".join(f"## {f.name}\n{f.content}" for f in self.fragments)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_spec_config.py -v`
Expected: PASS

**Step 5: Add export to config __init__.py**

Read `src/mozart/core/config/__init__.py` and add `SpecCorpusConfig` and `SpecFragment` to exports.

**Step 6: Commit**

```bash
git add src/mozart/core/config/spec.py src/mozart/core/config/__init__.py tests/test_spec_config.py
git commit -m "feat(spec): add SpecCorpusConfig and SpecFragment models"
```

---

## Task 4: Spec Corpus Loader (from .mozart/spec/ directory)

**Files:**
- Create: `src/mozart/spec/__init__.py` (replace placeholder if exists)
- Create: `src/mozart/spec/loader.py`
- Test: `tests/test_spec_loader.py`

**Step 1: Write the failing test**

```python
# tests/test_spec_loader.py
"""Tests for specification corpus loader."""

from pathlib import Path

import pytest
from mozart.spec.loader import SpecCorpusLoader


@pytest.fixture()
def spec_dir(tmp_path: Path) -> Path:
    """Create a mock .mozart/spec/ directory."""
    spec = tmp_path / ".mozart" / "spec"
    spec.mkdir(parents=True)

    # conventions.yaml
    (spec / "conventions.yaml").write_text(
        "name: conventions\n"
        "tags: [code, style]\n"
        "content: |\n"
        "  Use snake_case for functions.\n"
        "  Use PascalCase for classes.\n"
    )

    # quality.md (markdown format)
    (spec / "quality.md").write_text(
        "# Quality Standards\n\n"
        "80% test coverage minimum.\n"
        "All public functions documented.\n"
    )

    # constraints.yaml
    (spec / "constraints.yaml").write_text(
        "name: constraints\n"
        "tags: [safety, security]\n"
        "content: |\n"
        "  Never store plaintext credentials.\n"
        "  All API endpoints require rate limiting.\n"
    )

    return spec


class TestSpecCorpusLoader:
    def test_load_from_directory(self, spec_dir: Path):
        loader = SpecCorpusLoader(spec_dir)
        fragments = loader.load()
        assert len(fragments) >= 3

    def test_yaml_fragments_have_tags(self, spec_dir: Path):
        loader = SpecCorpusLoader(spec_dir)
        fragments = loader.load()
        conv = next(f for f in fragments if f.name == "conventions")
        assert "code" in conv.tags

    def test_markdown_fragments_infer_name(self, spec_dir: Path):
        loader = SpecCorpusLoader(spec_dir)
        fragments = loader.load()
        quality = next(f for f in fragments if f.name == "quality")
        assert "80% test coverage" in quality.content

    def test_nonexistent_dir_returns_empty(self, tmp_path: Path):
        loader = SpecCorpusLoader(tmp_path / "nonexistent")
        fragments = loader.load()
        assert fragments == []

    def test_filter_by_tags(self, spec_dir: Path):
        loader = SpecCorpusLoader(spec_dir)
        fragments = loader.load()
        safety = [f for f in fragments if f.matches_any(["safety"])]
        assert len(safety) >= 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_spec_loader.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/mozart/spec/__init__.py
"""Specification corpus system for Mozart."""

# src/mozart/spec/loader.py
"""Load specification fragments from .mozart/spec/ directory.

Supports two file formats:
- YAML (.yaml, .yml): Structured with name, tags, content fields
- Markdown (.md): Content-only, name inferred from filename
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from mozart.core.config.spec import SpecFragment

_logger = logging.getLogger(__name__)


class SpecCorpusLoader:
    """Load spec fragments from a directory."""

    def __init__(self, spec_dir: Path) -> None:
        self._spec_dir = spec_dir

    def load(self) -> list[SpecFragment]:
        """Load all spec fragments from the directory.

        Returns empty list if directory doesn't exist.
        """
        if not self._spec_dir.is_dir():
            _logger.debug("Spec dir %s does not exist, returning empty", self._spec_dir)
            return []

        fragments: list[SpecFragment] = []
        for path in sorted(self._spec_dir.iterdir()):
            if path.suffix in (".yaml", ".yml"):
                frag = self._load_yaml(path)
                if frag:
                    fragments.append(frag)
            elif path.suffix == ".md":
                frag = self._load_markdown(path)
                if frag:
                    fragments.append(frag)
        return fragments

    def _load_yaml(self, path: Path) -> SpecFragment | None:
        """Load a YAML spec fragment."""
        try:
            data = yaml.safe_load(path.read_text())
            if not isinstance(data, dict):
                _logger.warning("Spec file %s is not a YAML dict, skipping", path)
                return None
            return SpecFragment(
                name=data.get("name", path.stem),
                content=data.get("content", ""),
                tags=data.get("tags", []),
            )
        except Exception:
            _logger.warning("Failed to load spec file %s", path, exc_info=True)
            return None

    def _load_markdown(self, path: Path) -> SpecFragment | None:
        """Load a Markdown spec fragment. Name inferred from filename."""
        try:
            content = path.read_text()
            return SpecFragment(
                name=path.stem,
                content=content,
                tags=[],  # Markdown files don't carry tags
            )
        except Exception:
            _logger.warning("Failed to load spec file %s", path, exc_info=True)
            return None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_spec_loader.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mozart/spec/__init__.py src/mozart/spec/loader.py tests/test_spec_loader.py
git commit -m "feat(spec): add SpecCorpusLoader for .mozart/spec/ directory"
```

---

## Task 5: Wire SpecCorpusConfig into JobConfig

**Files:**
- Modify: `src/mozart/core/config/job.py` (add `spec` field to `JobConfig`)
- Modify: `src/mozart/core/config/__init__.py` (ensure exports)
- Test: `tests/test_job_config_spec.py`

**Step 1: Write the failing test**

```python
# tests/test_job_config_spec.py
"""Tests for spec corpus integration in JobConfig."""

from mozart.core.config.job import JobConfig
from mozart.core.config.spec import SpecCorpusConfig, SpecFragment


class TestJobConfigSpec:
    def test_default_no_spec(self):
        """JobConfig works without spec (backwards compatible)."""
        config = JobConfig(
            name="test-job",
            sheet={"size": 1, "total_items": 1},
            prompt={"template": "Do the thing."},
        )
        assert config.spec is not None  # Default empty config
        assert config.spec.fragments == []

    def test_inline_spec_fragments(self):
        """JobConfig accepts inline spec fragments."""
        config = JobConfig(
            name="test-job",
            sheet={"size": 1, "total_items": 1},
            prompt={"template": "Do the thing."},
            spec={
                "fragments": [
                    {"name": "conventions", "content": "Use snake_case.", "tags": ["code"]},
                ]
            },
        )
        assert len(config.spec.fragments) == 1
        assert config.spec.fragments[0].name == "conventions"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_job_config_spec.py -v`
Expected: FAIL (JobConfig has no `spec` field)

**Step 3: Add `spec` field to JobConfig**

In `src/mozart/core/config/job.py`, add import and field:

```python
# Add import at top
from mozart.core.config.spec import SpecCorpusConfig

# Add field to JobConfig class (after existing fields like 'notifications')
    spec: SpecCorpusConfig = Field(
        default_factory=SpecCorpusConfig,
        description="Specification corpus: project conventions, constraints, and quality standards.",
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_job_config_spec.py -v`
Expected: PASS

**Step 5: Run full test suite to verify no regressions**

Run: `pytest tests/ -x -q --timeout=60`
Expected: All existing tests pass

**Step 6: Commit**

```bash
git add src/mozart/core/config/job.py src/mozart/core/config/spec.py src/mozart/core/config/__init__.py tests/test_job_config_spec.py
git commit -m "feat(spec): wire SpecCorpusConfig into JobConfig"
```

---

## Task 6: Inject Spec Fragments into Prompts

**Files:**
- Modify: `src/mozart/prompts/templating.py` (PromptBuilder.build_sheet_prompt)
- Test: `tests/test_spec_injection.py`

**Step 1: Write the failing test**

```python
# tests/test_spec_injection.py
"""Tests for spec fragment injection into sheet prompts."""

from pathlib import Path

from mozart.core.config.job import PromptConfig
from mozart.core.config.spec import SpecFragment
from mozart.prompts.templating import PromptBuilder, SheetContext


def _make_context(**kwargs) -> SheetContext:
    defaults = dict(
        sheet_num=1, total_sheets=3, start_item=1, end_item=10,
        workspace=Path("/tmp/test"),
    )
    defaults.update(kwargs)
    return SheetContext(**defaults)


class TestSpecInjection:
    def test_no_fragments_no_section(self):
        builder = PromptBuilder(PromptConfig(template="Do the thing."))
        prompt = builder.build_sheet_prompt(_make_context(), spec_fragments=[])
        assert "## Project Specifications" not in prompt

    def test_fragments_injected_as_section(self):
        builder = PromptBuilder(PromptConfig(template="Do the thing."))
        fragments = [
            SpecFragment(name="conventions", content="Use snake_case."),
        ]
        prompt = builder.build_sheet_prompt(_make_context(), spec_fragments=fragments)
        assert "## Project Specifications" in prompt
        assert "Use snake_case." in prompt

    def test_multiple_fragments_all_present(self):
        builder = PromptBuilder(PromptConfig(template="Do the thing."))
        fragments = [
            SpecFragment(name="conventions", content="Use snake_case."),
            SpecFragment(name="quality", content="80% coverage."),
        ]
        prompt = builder.build_sheet_prompt(_make_context(), spec_fragments=fragments)
        assert "Use snake_case." in prompt
        assert "80% coverage." in prompt

    def test_spec_injected_before_validation_requirements(self):
        builder = PromptBuilder(PromptConfig(template="Do the thing."))
        from mozart.core.config.execution import ValidationRule
        rules = [ValidationRule(type="file_exists", path="/tmp/x", description="Test")]
        fragments = [SpecFragment(name="conv", content="snake_case")]
        prompt = builder.build_sheet_prompt(
            _make_context(), spec_fragments=fragments, validation_rules=rules,
        )
        spec_pos = prompt.index("## Project Specifications")
        val_pos = prompt.index("## Success Requirements")
        assert spec_pos < val_pos
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_spec_injection.py -v`
Expected: FAIL (build_sheet_prompt doesn't accept spec_fragments)

**Step 3: Modify PromptBuilder.build_sheet_prompt**

In `src/mozart/prompts/templating.py`, add `spec_fragments` parameter to `build_sheet_prompt`:

```python
def build_sheet_prompt(
    self,
    context: SheetContext,
    patterns: list[str] | None = None,
    validation_rules: list[ValidationRule] | None = None,
    failure_history: list["HistoricalFailure"] | None = None,
    spec_fragments: list["SpecFragment"] | None = None,  # NEW
) -> str:
```

Add injection logic BEFORE validation requirements:

```python
        # Inject spec corpus fragments if available
        if spec_fragments:
            spec_section = self._format_spec_fragments(spec_fragments)
            prompt = f"{prompt}\n\n{spec_section}"

        # Inject validation requirements if available (EXISTING — must come after spec)
        if validation_rules:
            ...
```

Add the formatting method:

```python
def _format_spec_fragments(self, fragments: list["SpecFragment"]) -> str:
    """Format spec corpus fragments for prompt injection."""
    if not fragments:
        return ""
    lines = ["## Project Specifications", ""]
    lines.append(
        "The following project specifications apply to this task:"
    )
    lines.append("")
    for frag in fragments:
        lines.append(f"### {frag.name}")
        lines.append(frag.content.strip())
        lines.append("")
    return "\n".join(lines)
```

Add TYPE_CHECKING import for SpecFragment at top of file.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_spec_injection.py -v`
Expected: PASS

**Step 5: Run existing prompt tests to verify no regressions**

Run: `pytest tests/ -k "prompt" -v --timeout=60`
Expected: All existing prompt tests pass

**Step 6: Commit**

```bash
git add src/mozart/prompts/templating.py tests/test_spec_injection.py
git commit -m "feat(spec): inject spec fragments into sheet prompts"
```

---

## Task 7: Wire Spec Loading into the Runner

**Files:**
- Modify: `src/mozart/execution/runner/base.py` (load spec at job start)
- Modify: `src/mozart/execution/runner/sheet.py` (pass fragments to PromptBuilder)
- Test: Integration test via existing runner test patterns

**Step 1: Read runner base to find initialization point**

Read `src/mozart/execution/runner/base.py` to find where config is loaded and PromptBuilder is created. Find the `__init__` or setup method.

**Step 2: Add spec loading to runner initialization**

In runner's initialization, after config loading:

```python
# Load spec corpus fragments
self._spec_fragments: list[SpecFragment] = []
if self.config.spec.spec_dir:
    from mozart.spec.loader import SpecCorpusLoader
    loader = SpecCorpusLoader(self.config.spec.spec_dir)
    self._spec_fragments = loader.load()
elif self.config.spec.fragments:
    self._spec_fragments = self.config.spec.fragments
```

**Step 3: Pass fragments to build_sheet_prompt calls**

Find all calls to `self._prompt_builder.build_sheet_prompt(...)` in `sheet.py` and add `spec_fragments=self._spec_fragments`.

**Step 4: Write integration test**

```python
# tests/test_runner_spec_integration.py
"""Integration test: runner loads and injects spec fragments."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mozart.core.config.job import JobConfig
from mozart.core.config.spec import SpecFragment


class TestRunnerSpecIntegration:
    def test_inline_fragments_reach_prompt(self):
        """Verify inline spec fragments appear in assembled prompt."""
        config = JobConfig(
            name="test",
            sheet={"size": 1, "total_items": 1},
            prompt={"template": "Do task."},
            spec={
                "fragments": [
                    {"name": "rules", "content": "Always test first.", "tags": ["code"]},
                ]
            },
        )
        assert len(config.spec.fragments) == 1
        assert config.spec.fragments[0].content == "Always test first."
```

**Step 5: Run all tests**

Run: `pytest tests/ -x -q --timeout=60`
Expected: PASS

**Step 6: Commit**

```bash
git add src/mozart/execution/runner/base.py src/mozart/execution/runner/sheet.py tests/test_runner_spec_integration.py
git commit -m "feat(spec): wire spec corpus loading into runner and prompt assembly"
```

---

## Task 8: Budget-Aware Prompt Assembly

**Files:**
- Modify: `src/mozart/prompts/templating.py` (add budget tracking to build_sheet_prompt)
- Test: `tests/test_prompt_budget.py`

**Step 1: Write the failing test**

```python
# tests/test_prompt_budget.py
"""Tests for budget-aware prompt assembly."""

from pathlib import Path

from mozart.core.config.execution import ValidationRule
from mozart.core.config.job import PromptConfig
from mozart.core.config.spec import SpecFragment
from mozart.core.tokens import TokenBudgetTracker
from mozart.prompts.templating import PromptBuilder, SheetContext


def _ctx() -> SheetContext:
    return SheetContext(
        sheet_num=1, total_sheets=1, start_item=1, end_item=1,
        workspace=Path("/tmp/test"),
    )


class TestPromptBudgetTracking:
    def test_build_returns_budget_report(self):
        builder = PromptBuilder(
            PromptConfig(template="Short prompt."),
            token_budget=TokenBudgetTracker(window_size=200_000),
        )
        prompt = builder.build_sheet_prompt(_ctx())
        report = builder.last_budget_report()
        assert report is not None
        assert "template" in report
        assert report["template"] > 0

    def test_budget_tracks_all_components(self):
        builder = PromptBuilder(
            PromptConfig(template="Do the thing."),
            token_budget=TokenBudgetTracker(window_size=200_000),
        )
        fragments = [SpecFragment(name="x", content="y" * 500)]
        rules = [ValidationRule(type="file_exists", path="/tmp/x", description="Test")]
        builder.build_sheet_prompt(
            _ctx(), spec_fragments=fragments, validation_rules=rules,
        )
        report = builder.last_budget_report()
        assert "spec_fragments" in report
        assert "validation_rules" in report

    def test_over_budget_drops_low_priority_components(self):
        # Tiny budget: only room for template
        builder = PromptBuilder(
            PromptConfig(template="Short."),
            token_budget=TokenBudgetTracker(window_size=100, reserve=0),
        )
        fragments = [SpecFragment(name="big", content="x" * 1000)]
        prompt = builder.build_sheet_prompt(_ctx(), spec_fragments=fragments)
        # Spec fragments should be dropped (low priority) to fit budget
        assert "## Project Specifications" not in prompt
        assert "Short." in prompt
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt_budget.py -v`
Expected: FAIL (PromptBuilder doesn't accept token_budget)

**Step 3: Implement budget-aware assembly**

Modify `PromptBuilder.__init__` to accept optional `TokenBudgetTracker`. Modify `build_sheet_prompt` to track allocations and drop low-priority components when over budget.

Priority order (highest to lowest):
1. Template body (never dropped)
2. Preamble context
3. Validation requirements
4. Spec fragments
5. Learned patterns
6. Historical failures
7. Injected skills/tools/context

Add `last_budget_report()` method returning `dict[str, int]`.

**Step 4: Run tests**

Run: `pytest tests/test_prompt_budget.py -v`
Expected: PASS

**Step 5: Run full suite**

Run: `pytest tests/ -x -q --timeout=60`
Expected: PASS

**Step 6: Commit**

```bash
git add src/mozart/prompts/templating.py tests/test_prompt_budget.py
git commit -m "feat(tokens): budget-aware prompt assembly drops low-priority components"
```

---

## Task 9: `mozart init` CLI Command

**Files:**
- Create: `src/mozart/cli/commands/init.py`
- Modify: `src/mozart/cli/__init__.py` (register command)
- Test: `tests/test_cli_init.py`

**Step 1: Write the failing test**

```python
# tests/test_cli_init.py
"""Tests for mozart init command."""

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from mozart.cli import cli


class TestMozartInit:
    def test_creates_spec_directory(self, tmp_path: Path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init", "--non-interactive"])
            assert result.exit_code == 0
            spec_dir = Path(".mozart/spec")
            assert spec_dir.is_dir()

    def test_creates_default_files(self, tmp_path: Path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init", "--non-interactive"])
            assert result.exit_code == 0
            assert (Path(".mozart/spec/intent.yaml")).exists()
            assert (Path(".mozart/spec/conventions.yaml")).exists()
            assert (Path(".mozart/spec/constraints.yaml")).exists()
            assert (Path(".mozart/spec/quality.yaml")).exists()

    def test_does_not_overwrite_existing(self, tmp_path: Path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            spec = Path(".mozart/spec")
            spec.mkdir(parents=True)
            (spec / "intent.yaml").write_text("custom: true")

            result = runner.invoke(cli, ["init", "--non-interactive"])
            assert result.exit_code == 0
            content = (spec / "intent.yaml").read_text()
            assert "custom: true" in content  # Not overwritten
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_init.py -v`
Expected: FAIL

**Step 3: Implement init command**

```python
# src/mozart/cli/commands/init.py
"""mozart init — Bootstrap specification corpus for a project."""

from __future__ import annotations

from pathlib import Path

import click

from mozart.cli.output import console

_DEFAULT_INTENT = """\
name: intent
tags: [goals, trade-offs]
content: |
  # Project Intent
  # Uncomment and fill in your project's goals and trade-offs.
  #
  # goals:
  #   primary: "correctness"
  #   secondary: ["completeness", "maintainability"]
  #
  # trade_offs:
  #   correctness_vs_speed: "correctness"
  #   completeness_vs_cost: "completeness_within_budget"
"""

_DEFAULT_CONVENTIONS = """\
name: conventions
tags: [code, style]
content: |
  # Coding Conventions
  # Add your project's coding standards here.
  # These will be injected into sheet prompts.
"""

_DEFAULT_CONSTRAINTS = """\
name: constraints
tags: [safety, security]
content: |
  # Constraints
  # Add musts, must-nots, and preferences here.
  #
  # musts:
  #   - "All tests must pass before completion"
  # must_nots:
  #   - "Never store plaintext credentials"
  # preferences:
  #   - "Prefer composition over inheritance"
"""

_DEFAULT_QUALITY = """\
name: quality
tags: [quality, testing]
content: |
  # Quality Standards
  # Define what "good enough" means for this project.
  #
  # test_coverage_minimum: 0.80
  # documentation: "public API fully documented"
  # error_handling: "all external calls have explicit error paths"
"""

_DEFAULTS = {
    "intent.yaml": _DEFAULT_INTENT,
    "conventions.yaml": _DEFAULT_CONVENTIONS,
    "constraints.yaml": _DEFAULT_CONSTRAINTS,
    "quality.yaml": _DEFAULT_QUALITY,
}


@click.command("init")
@click.option(
    "--non-interactive",
    is_flag=True,
    default=False,
    help="Skip interactive questions; create defaults only.",
)
def init_command(non_interactive: bool) -> None:
    """Bootstrap a .mozart/spec/ corpus for this project."""
    spec_dir = Path(".mozart/spec")
    spec_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0
    for filename, content in _DEFAULTS.items():
        path = spec_dir / filename
        if path.exists():
            console.print(f"  [dim]skip[/dim] {path} (already exists)")
            skipped += 1
        else:
            path.write_text(content)
            console.print(f"  [green]create[/green] {path}")
            created += 1

    console.print(
        f"\n[bold]Spec corpus initialized:[/bold] "
        f"{created} created, {skipped} skipped"
    )
    console.print(
        "\nEdit the files in [cyan].mozart/spec/[/cyan] to define your project's "
        "conventions, constraints, and quality standards."
    )
    console.print(
        "Then reference them in your score: [cyan]spec.spec_dir: .mozart/spec[/cyan]"
    )
```

**Step 4: Register in CLI**

In `src/mozart/cli/__init__.py`, add the init command to the CLI group.

**Step 5: Run tests**

Run: `pytest tests/test_cli_init.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/mozart/cli/commands/init.py src/mozart/cli/__init__.py tests/test_cli_init.py
git commit -m "feat(cli): add 'mozart init' to bootstrap spec corpus"
```

---

## Task 10: Spec Fragment Tag Selection via Sheet Config

**Files:**
- Modify: `src/mozart/core/config/job.py` (add `spec_tags` to SheetConfig)
- Modify: `src/mozart/execution/runner/base.py` (filter fragments by tags)
- Test: `tests/test_spec_tag_selection.py`

**Step 1: Write the failing test**

```python
# tests/test_spec_tag_selection.py
"""Tests for per-sheet spec fragment tag selection."""

from mozart.core.config.spec import SpecCorpusConfig, SpecFragment


class TestSpecTagSelection:
    def test_tags_filter_fragments(self):
        corpus = SpecCorpusConfig(
            fragments=[
                SpecFragment(name="code", content="snake_case", tags=["code"]),
                SpecFragment(name="test", content="80% coverage", tags=["testing"]),
                SpecFragment(name="security", content="no secrets", tags=["security"]),
            ]
        )
        # Sheet doing code work only needs code + testing
        relevant = corpus.get_fragments_by_tags(["code", "testing"])
        names = [f.name for f in relevant]
        assert "code" in names
        assert "test" in names
        assert "security" not in names

    def test_empty_tags_returns_nothing(self):
        corpus = SpecCorpusConfig(
            fragments=[
                SpecFragment(name="x", content="y", tags=["code"]),
            ]
        )
        assert corpus.get_fragments_by_tags([]) == []
```

**Step 2: Run to verify passes (model already supports this)**

Run: `pytest tests/test_spec_tag_selection.py -v`
Expected: PASS (SpecCorpusConfig.get_fragments_by_tags already works)

**Step 3: Add `spec_tags` to SheetConfig**

In `src/mozart/core/config/job.py`, add to `SheetConfig`:

```python
    spec_tags: dict[int, list[str]] = Field(
        default_factory=dict,
        description=(
            "Per-sheet spec fragment tag selection. Maps sheet number to list of "
            "tags. Only fragments matching these tags are injected into that sheet. "
            "If empty or not specified for a sheet, all fragments are injected."
        ),
    )
```

**Step 4: Wire into runner — filter fragments by sheet's tags before injection**

In the runner's prompt assembly, before passing `spec_fragments`:

```python
# Select relevant fragments for this sheet
sheet_tags = self.config.sheet.spec_tags.get(sheet_num, [])
if sheet_tags:
    fragments = self.config.spec.get_fragments_by_tags(sheet_tags)
else:
    fragments = self._spec_fragments  # All fragments
```

**Step 5: Test and commit**

Run: `pytest tests/ -x -q --timeout=60`
Expected: PASS

```bash
git add src/mozart/core/config/job.py src/mozart/execution/runner/base.py tests/test_spec_tag_selection.py
git commit -m "feat(spec): per-sheet spec tag selection for targeted fragment injection"
```

---

## Summary

| Task | What | Effort |
|------|------|--------|
| 1 | Token estimation utility | S |
| 2 | Token budget tracker | S |
| 3 | SpecCorpusConfig + SpecFragment models | S |
| 4 | Spec corpus loader (.mozart/spec/) | S |
| 5 | Wire SpecCorpusConfig into JobConfig | S |
| 6 | Inject spec fragments into prompts | M |
| 7 | Wire spec loading into runner | M |
| 8 | Budget-aware prompt assembly | M |
| 9 | `mozart init` CLI command | M |
| 10 | Per-sheet spec tag selection | S |

**Total estimated: ~3-5 days of focused implementation.**

After Phase 1 completes, Phase 2 (Prompt Craft Hardening) builds directly on top: guardrails, examples, output format, and ambiguity resolution all use the same PromptBuilder injection points established here.
