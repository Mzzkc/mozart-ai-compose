"""Tests for the `raw_prompt` instrument-profile bypass.

When an `InstrumentProfile` declares `raw_prompt: true`, the prompt-assembly
pipeline must skip every wrapping layer (skills/tools, injected context,
spec fragments, failure history, learned patterns, validation requirements,
preamble, completion suffix) and pass the rendered Jinja template verbatim.

Validations still RUN after execution; they just never appear in the prompt.

The primary motivator is the `cli` (bash) instrument — its prompt must be
a literal shell command, and Marianne's wrapping layers would corrupt it.
"""

from __future__ import annotations

from pathlib import Path

from marianne.core.config import PromptConfig, ValidationRule
from marianne.core.config.spec import SpecFragment
from marianne.instruments.loader import InstrumentProfileLoader
from marianne.prompts.templating import PromptBuilder, SheetContext

# Use an absolute-ish path that doesn't need to exist on disk; PromptBuilder
# only stringifies the workspace, never reads from it.
_WS = Path("/tmp/raw-prompt-test-ws")


def _make_context(*, sheet_num: int = 1, instance: int = 1) -> SheetContext:
    return SheetContext(
        sheet_num=sheet_num,
        total_sheets=3,
        start_item=sheet_num,
        end_item=sheet_num,
        workspace=_WS,
        stage=sheet_num,
        instance=instance,
        fan_count=2,
        total_stages=3,
    )


def _make_spec_fragment(content: str) -> SpecFragment:
    return SpecFragment(
        name="intent",
        content=content,
        tags=["goals"],
    )


# ──────────────────────────────────────────────────────────────────────────
# PromptBuilder.build_sheet_prompt — direct bypass test
# ──────────────────────────────────────────────────────────────────────────


class TestPromptBuilderRawPrompt:
    """Direct tests of the bypass at the PromptBuilder layer."""

    def test_raw_prompt_returns_only_rendered_template(self) -> None:
        """raw_prompt=True with all wrapping inputs → output is template only."""
        cfg = PromptConfig(template="ls -la {{ workspace }}")
        pb = PromptBuilder(cfg)
        ctx = _make_context()
        # Populate ALL injection slots with content that would normally appear
        ctx.injected_skills.append("SKILL CONTENT THAT MUST NOT APPEAR")
        ctx.injected_tools.append("TOOL CONTENT THAT MUST NOT APPEAR")
        ctx.injected_context.append("CONTEXT THAT MUST NOT APPEAR")

        prompt = pb.build_sheet_prompt(
            ctx,
            patterns=["pattern that must not appear"],
            validation_rules=[
                ValidationRule(
                    type="file_exists",
                    path="{workspace}/out.txt",
                    description="Output file exists",
                )
            ],
            spec_fragments=[_make_spec_fragment("SPEC THAT MUST NOT APPEAR")],
            raw_prompt=True,
        )

        assert prompt == f"ls -la {_WS}"
        # No wrapping markers
        assert "## Injected Skills" not in prompt
        assert "## Injected Tools" not in prompt
        assert "## Injected Context" not in prompt
        assert "## Injected Specs" not in prompt
        assert "## Lessons From Previous Sheets" not in prompt
        assert "## Learned Patterns" not in prompt
        assert "## Success Requirements" not in prompt
        # And no leaked content from the inputs
        assert "MUST NOT APPEAR" not in prompt
        assert "pattern that must not appear" not in prompt

    def test_raw_prompt_still_substitutes_template_variables(self) -> None:
        """Jinja rendering against SheetContext vars must still happen."""
        cfg = PromptConfig(
            template="echo sheet={{ sheet_num }} instance={{ instance }} ws={{ workspace }}"
        )
        pb = PromptBuilder(cfg)
        ctx = _make_context(sheet_num=2, instance=3)

        prompt = pb.build_sheet_prompt(ctx, raw_prompt=True)

        assert prompt == f"echo sheet=2 instance=3 ws={_WS}"

    def test_raw_prompt_default_false_preserves_existing_behavior(self) -> None:
        """Default raw_prompt=False must produce the wrapped prompt unchanged."""
        cfg = PromptConfig(template="Do the task.")
        pb = PromptBuilder(cfg)
        ctx = _make_context()
        ctx.injected_skills.append("SKILL X")

        wrapped = pb.build_sheet_prompt(ctx)
        assert "## Injected Skills" in wrapped
        assert "SKILL X" in wrapped
        assert "Do the task." in wrapped

    def test_raw_prompt_with_no_template_falls_back_to_default_prompt(self) -> None:
        """Edge case: no template → default prompt, still no wrapping."""
        cfg = PromptConfig()
        pb = PromptBuilder(cfg)
        ctx = _make_context()
        ctx.injected_skills.append("SKILL X")

        prompt = pb.build_sheet_prompt(ctx, raw_prompt=True)

        # Default prompt is built but no skills section appears
        assert "## Injected Skills" not in prompt
        assert "SKILL X" not in prompt


# ──────────────────────────────────────────────────────────────────────────
# Builtin profile flag
# ──────────────────────────────────────────────────────────────────────────


class TestBuiltinProfileRawPromptFlag:
    """The `cli` builtin must declare raw_prompt=true; AI agent profiles must not."""

    def test_cli_profile_has_raw_prompt_true(self) -> None:
        """The cli (bash) builtin profile must opt into raw mode."""
        builtins = (
            Path(__file__).resolve().parent.parent
            / "src" / "marianne" / "instruments" / "builtins"
        )
        profiles = InstrumentProfileLoader.load_directory(builtins)
        assert "cli" in profiles, "cli profile not loaded from builtins"
        assert profiles["cli"].raw_prompt is True, (
            "cli profile must set raw_prompt: true — its prompt is a bash "
            "command and prompt-wrapping layers would corrupt it"
        )

    def test_agent_harness_profiles_do_not_set_raw_prompt(self) -> None:
        """AI-agent profiles must keep raw_prompt False (the default)."""
        builtins = (
            Path(__file__).resolve().parent.parent
            / "src" / "marianne" / "instruments" / "builtins"
        )
        profiles = InstrumentProfileLoader.load_directory(builtins)
        for name in ("claude-code", "gemini-cli", "codex-cli", "aider", "goose"):
            assert name in profiles, f"{name} profile not loaded"
            assert profiles[name].raw_prompt is False, (
                f"{name} should keep prompt-assembly wrapping; raw_prompt "
                "should remain False"
            )
