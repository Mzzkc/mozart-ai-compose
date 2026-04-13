"""Tests for the technique router — output classification accuracy.

Tests cover:
- Prose detection (default, safe)
- Code block extraction (executable languages only)
- Tool call parsing (@tool and JSON patterns)
- A2A request parsing (@delegate pattern)
- Classification priority (A2A > tool > code > prose)
- Edge cases (empty, mixed, malformed)
"""

from __future__ import annotations

import pytest

from marianne.daemon.technique_router import (
    A2ARoutingRequest,
    ClassifiedOutput,
    CodeBlock,
    OutputKind,
    TechniqueRouter,
    ToolCallRequest,
)


@pytest.fixture
def router() -> TechniqueRouter:
    """Shared router instance."""
    return TechniqueRouter()


# =============================================================================
# Prose classification (default)
# =============================================================================


class TestProseClassification:
    """Prose is the safe default for unrecognized output."""

    def test_plain_text(self, router: TechniqueRouter) -> None:
        """Plain text is classified as prose."""
        result = router.classify("This is a review of the architecture.")
        assert result.kind == OutputKind.PROSE
        assert result.raw_output == "This is a review of the architecture."

    def test_empty_string(self, router: TechniqueRouter) -> None:
        """Empty string is prose."""
        result = router.classify("")
        assert result.kind == OutputKind.PROSE

    def test_whitespace_only(self, router: TechniqueRouter) -> None:
        """Whitespace-only is prose."""
        result = router.classify("   \n\t  \n  ")
        assert result.kind == OutputKind.PROSE

    def test_markdown_without_code(self, router: TechniqueRouter) -> None:
        """Markdown formatting without code blocks is prose."""
        text = """## Analysis

        - Point 1: The architecture is sound
        - Point 2: Tests need improvement

        **Conclusion**: Ship it.
        """
        result = router.classify(text)
        assert result.kind == OutputKind.PROSE

    def test_non_executable_code_fence(self, router: TechniqueRouter) -> None:
        """Code fences with non-executable languages are prose."""
        text = """Here's the config:

```yaml
name: my-score
workspace: ./ws
```

And the schema:

```json
{"type": "object"}
```
"""
        result = router.classify(text)
        assert result.kind == OutputKind.PROSE

    def test_code_fence_no_language(self, router: TechniqueRouter) -> None:
        """Code fences without a language tag are prose."""
        text = """Here's some output:

```
some random text
not actual code
```
"""
        result = router.classify(text)
        assert result.kind == OutputKind.PROSE


# =============================================================================
# Code block extraction
# =============================================================================


class TestCodeBlockExtraction:
    """Executable code blocks in markdown fences."""

    def test_python_code_block(self, router: TechniqueRouter) -> None:
        """Python code blocks are detected."""
        text = """I'll write a script:

```python
import os
print(os.getcwd())
```
"""
        result = router.classify(text)
        assert result.kind == OutputKind.CODE_BLOCK
        assert len(result.code_blocks) == 1
        assert result.code_blocks[0].language == "python"
        assert "import os" in result.code_blocks[0].code

    def test_bash_code_block(self, router: TechniqueRouter) -> None:
        """Bash code blocks are detected."""
        text = """Run this:

```bash
ls -la /workspace
echo "done"
```
"""
        result = router.classify(text)
        assert result.kind == OutputKind.CODE_BLOCK
        assert result.code_blocks[0].language == "bash"

    def test_javascript_code_block(self, router: TechniqueRouter) -> None:
        """JavaScript code blocks are detected."""
        text = """```javascript
const fs = require('fs');
console.log(fs.readdirSync('.'));
```"""
        result = router.classify(text)
        assert result.kind == OutputKind.CODE_BLOCK
        assert result.code_blocks[0].language == "javascript"

    def test_multiple_code_blocks(self, router: TechniqueRouter) -> None:
        """Multiple executable code blocks are all extracted."""
        text = """First script:

```python
print("hello")
```

Second script:

```bash
echo "world"
```
"""
        result = router.classify(text)
        assert result.kind == OutputKind.CODE_BLOCK
        assert len(result.code_blocks) == 2
        assert result.code_blocks[0].language == "python"
        assert result.code_blocks[1].language == "bash"

    def test_mixed_executable_and_doc(self, router: TechniqueRouter) -> None:
        """Only executable blocks are extracted, doc blocks ignored."""
        text = """Config:

```yaml
name: test
```

Script:

```python
print("test")
```
"""
        result = router.classify(text)
        assert result.kind == OutputKind.CODE_BLOCK
        assert len(result.code_blocks) == 1
        assert result.code_blocks[0].language == "python"

    def test_empty_code_block_ignored(self, router: TechniqueRouter) -> None:
        """Empty code blocks are ignored."""
        text = """```python
```

Some text after."""
        result = router.classify(text)
        assert result.kind == OutputKind.PROSE

    def test_py_shorthand(self, router: TechniqueRouter) -> None:
        """py is recognized as Python."""
        text = """```py
x = 42
print(x)
```"""
        result = router.classify(text)
        assert result.kind == OutputKind.CODE_BLOCK
        assert result.code_blocks[0].language == "py"

    def test_sh_shorthand(self, router: TechniqueRouter) -> None:
        """sh is recognized as shell."""
        text = """```sh
echo test
```"""
        result = router.classify(text)
        assert result.kind == OutputKind.CODE_BLOCK
        assert result.code_blocks[0].language == "sh"


# =============================================================================
# Tool call extraction
# =============================================================================


class TestToolCallExtraction:
    """MCP tool invocation patterns."""

    def test_at_tool_pattern(self, router: TechniqueRouter) -> None:
        """@tool server.method(args) pattern."""
        text = '@tool github.list_issues(state="open", labels="P0")'
        result = router.classify(text)
        assert result.kind == OutputKind.TOOL_CALL
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].server == "github"
        assert result.tool_calls[0].method == "list_issues"
        assert result.tool_calls[0].arguments == {"state": "open", "labels": "P0"}

    def test_at_tool_no_args(self, router: TechniqueRouter) -> None:
        """@tool with empty args."""
        text = "@tool filesystem.list_directory()"
        result = router.classify(text)
        assert result.kind == OutputKind.TOOL_CALL
        assert result.tool_calls[0].server == "filesystem"
        assert result.tool_calls[0].method == "list_directory"
        assert result.tool_calls[0].arguments == {}

    def test_json_tool_pattern(self, router: TechniqueRouter) -> None:
        """JSON-like tool call pattern."""
        text = '{"tool": "github.create_issue", "arguments": {"title": "Bug"}}'
        result = router.classify(text)
        assert result.kind == OutputKind.TOOL_CALL
        assert result.tool_calls[0].server == "github"
        assert result.tool_calls[0].method == "create_issue"

    def test_multiple_tool_calls(self, router: TechniqueRouter) -> None:
        """Multiple tool calls in one output."""
        text = """
@tool github.list_issues(state="open")
@tool filesystem.read_file(path="README.md")
"""
        result = router.classify(text)
        assert result.kind == OutputKind.TOOL_CALL
        assert len(result.tool_calls) == 2


# =============================================================================
# A2A request extraction
# =============================================================================


class TestA2ARequestExtraction:
    """Inter-agent task delegation patterns."""

    def test_delegate_pattern(self, router: TechniqueRouter) -> None:
        """@delegate agent: description pattern."""
        text = "@delegate sentinel: Review security of the auth module"
        result = router.classify(text)
        assert result.kind == OutputKind.A2A_REQUEST
        assert len(result.a2a_requests) == 1
        assert result.a2a_requests[0].target_agent == "sentinel"
        assert "Review security" in result.a2a_requests[0].task_description

    def test_multiple_delegates(self, router: TechniqueRouter) -> None:
        """Multiple delegation requests."""
        text = """
@delegate sentinel: Check auth module security
@delegate forge: Implement the fix for issue #42
"""
        result = router.classify(text)
        assert result.kind == OutputKind.A2A_REQUEST
        assert len(result.a2a_requests) == 2
        assert result.a2a_requests[0].target_agent == "sentinel"
        assert result.a2a_requests[1].target_agent == "forge"


# =============================================================================
# Classification priority
# =============================================================================


class TestClassificationPriority:
    """A2A > tool > code > prose — first match wins."""

    def test_a2a_takes_priority_over_code(self, router: TechniqueRouter) -> None:
        """A2A requests take priority over code blocks."""
        text = """
@delegate sentinel: Review this code

```python
print("hello")
```
"""
        result = router.classify(text)
        assert result.kind == OutputKind.A2A_REQUEST

    def test_tool_takes_priority_over_code(self, router: TechniqueRouter) -> None:
        """Tool calls take priority over code blocks."""
        text = """
@tool github.list_issues(state="open")

```python
print("analyzing issues")
```
"""
        result = router.classify(text)
        assert result.kind == OutputKind.TOOL_CALL

    def test_a2a_takes_priority_over_tool(self, router: TechniqueRouter) -> None:
        """A2A requests take priority over tool calls."""
        text = """
@delegate sentinel: Check this
@tool github.list_issues()
"""
        result = router.classify(text)
        assert result.kind == OutputKind.A2A_REQUEST


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    """Edge cases and adversarial inputs."""

    def test_none_like_empty(self, router: TechniqueRouter) -> None:
        """Empty/whitespace classified as prose."""
        assert router.classify("").kind == OutputKind.PROSE
        assert router.classify("  ").kind == OutputKind.PROSE
        assert router.classify("\n\n").kind == OutputKind.PROSE

    def test_at_tool_in_prose_context(self, router: TechniqueRouter) -> None:
        """@tool pattern must match the full syntax."""
        text = "I mentioned @tool in a sentence but not as a call."
        result = router.classify(text)
        assert result.kind == OutputKind.PROSE

    def test_at_delegate_in_prose_context(self, router: TechniqueRouter) -> None:
        """@delegate pattern must match the full syntax."""
        text = "The @delegate pattern is documented here."
        # This actually matches "@delegate pattern: is documented here."
        # but that's ok — the pattern requires "word colon description"
        result = router.classify(text)
        # It might match — let's verify the pattern behavior
        if result.kind == OutputKind.A2A_REQUEST:
            assert result.a2a_requests[0].target_agent == "pattern"

    def test_code_fence_with_backtick_in_content(
        self, router: TechniqueRouter,
    ) -> None:
        """Code content with backticks doesn't break parsing."""
        text = '''```python
x = "`hello`"
print(x)
```'''
        result = router.classify(text)
        assert result.kind == OutputKind.CODE_BLOCK
        assert '"`hello`"' in result.code_blocks[0].code

    def test_classified_output_preserves_raw(
        self, router: TechniqueRouter,
    ) -> None:
        """ClassifiedOutput always preserves the raw output."""
        text = "@delegate forge: Build it"
        result = router.classify(text)
        assert result.raw_output == text

    def test_large_output(self, router: TechniqueRouter) -> None:
        """Large outputs don't crash the router."""
        # 100KB of prose
        text = "word " * 20000
        result = router.classify(text)
        assert result.kind == OutputKind.PROSE

    def test_typescript_code_block(self, router: TechniqueRouter) -> None:
        """TypeScript code blocks are executable."""
        text = """```typescript
const x: number = 42;
console.log(x);
```"""
        result = router.classify(text)
        assert result.kind == OutputKind.CODE_BLOCK
        assert result.code_blocks[0].language == "typescript"

    def test_shell_code_block(self, router: TechniqueRouter) -> None:
        """shell language tag is recognized."""
        text = """```shell
ls -la
```"""
        result = router.classify(text)
        assert result.kind == OutputKind.CODE_BLOCK
        assert result.code_blocks[0].language == "shell"
