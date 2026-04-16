"""Technique resolution for the baton dispatch pipeline.

Resolves which techniques are active for a given sheet phase and generates
technique manifests for prompt injection. This module bridges the technique
declarations in JobConfig (``TechniqueConfig``) to the prompt assembly
pipeline in the baton adapter.

Usage::

    from marianne.daemon.baton.techniques import resolve_techniques_for_sheet

    resolved = resolve_techniques_for_sheet(config.techniques, "work")
    # resolved.manifest -> markdown text for injection
    # resolved.mcp_servers -> dict of MCP server names for config file generation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from marianne.core.config.techniques import TechniqueConfig, TechniqueKind

if TYPE_CHECKING:
    from marianne.daemon.mcp_pool import McpPoolManager


@dataclass(frozen=True)
class ResolvedTechniques:
    """Resolved techniques for a specific phase.

    Contains the filtered techniques, generated manifest text,
    and MCP server names for config file generation.

    Attributes:
        skills: Names of active skill techniques.
        mcp_servers: Mapping of technique name to MCP server name.
        protocols: Names of active protocol techniques.
        manifest: Generated markdown text for prompt injection.
    """

    skills: list[str] = field(default_factory=list)
    mcp_servers: dict[str, str] = field(default_factory=dict)
    protocols: list[str] = field(default_factory=list)
    manifest: str = ""


def filter_techniques_for_phase(
    techniques: dict[str, TechniqueConfig],
    phase: str,
) -> dict[str, TechniqueConfig]:
    """Filter techniques to those active in a given phase.

    A technique matches if:
    - Its phases list contains the exact phase name, OR
    - Its phases list contains "all" (wildcard)

    Args:
        techniques: Full technique declarations from JobConfig.
        phase: Current sheet phase name (e.g., "work", "recon").

    Returns:
        Dict of technique name to TechniqueConfig for active techniques.
    """
    if not techniques:
        return {}
    return {name: tc for name, tc in techniques.items() if phase in tc.phases or "all" in tc.phases}


def generate_technique_manifest(
    techniques: dict[str, TechniqueConfig],
) -> str:
    """Generate a technique manifest for prompt injection.

    Produces human-readable markdown describing available techniques,
    organized by kind (MCP Tools, Protocols, Skills).

    Args:
        techniques: Filtered techniques for the current phase.

    Returns:
        Markdown text for injection as a SKILL-category item.
    """
    if not techniques:
        return ""

    sections: list[str] = ["## Techniques Available This Phase"]

    mcp = {n: t for n, t in techniques.items() if t.kind == TechniqueKind.MCP}
    protocols = {n: t for n, t in techniques.items() if t.kind == TechniqueKind.PROTOCOL}
    skills = {n: t for n, t in techniques.items() if t.kind == TechniqueKind.SKILL}

    if mcp:
        sections.append("\n### MCP Tools")
        for name, tc in mcp.items():
            server = tc.config.get("server", name)
            sections.append(f"- **{name}** (server: {server})")

    if protocols:
        sections.append("\n### Protocols")
        for name in protocols:
            sections.append(f"- **{name}**")

    if skills:
        sections.append("\n### Skills")
        for name in skills:
            sections.append(f"- **{name}**")

    return "\n".join(sections)


def resolve_techniques_for_sheet(
    techniques: dict[str, TechniqueConfig],
    phase: str,
) -> ResolvedTechniques:
    """Full resolution pipeline: filter + manifest generation.

    Args:
        techniques: All technique declarations from JobConfig.
        phase: Current sheet phase.

    Returns:
        ResolvedTechniques with filtered data and generated manifest.
    """
    filtered = filter_techniques_for_phase(techniques, phase)
    if not filtered:
        return ResolvedTechniques()

    manifest = generate_technique_manifest(filtered)

    skill_names = [n for n, t in filtered.items() if t.kind == TechniqueKind.SKILL]
    mcp_servers = {
        n: t.config.get("server", n) for n, t in filtered.items() if t.kind == TechniqueKind.MCP
    }
    protocol_names = [n for n, t in filtered.items() if t.kind == TechniqueKind.PROTOCOL]

    return ResolvedTechniques(
        skills=skill_names,
        mcp_servers=mcp_servers,
        protocols=protocol_names,
        manifest=manifest,
    )


def generate_mcp_config_file(
    mcp_servers: dict[str, dict[str, str]],
    pool: McpPoolManager,
    workspace: Path,
) -> Path | None:
    """Generate an MCP config JSON file from pool socket paths.

    Produces a JSON file mapping server names to their Unix socket paths.
    Only servers that are both declared in ``mcp_servers`` and currently
    running in the pool are included.

    The file is written atomically (write-to-temp + rename) to prevent
    partial reads by concurrent processes.

    Args:
        mcp_servers: Mapping of technique name to server declaration.
            Each value should have a ``"server"`` key with the pool
            server name.
        pool: The MCP pool manager to query for running state and
            socket paths.
        workspace: Directory where the config file is written.

    Returns:
        Path to the generated config file, or None if no servers are
        running.
    """
    if not mcp_servers:
        return None

    servers_config: dict[str, dict[str, str]] = {}
    for _tech_name, server_decl in mcp_servers.items():
        server_name = server_decl.get("server", "")
        if not pool.is_running(server_name):
            continue
        socket_path = pool.get_socket_path(server_name)
        if socket_path is None:
            continue
        servers_config[server_name] = {"socket": str(socket_path)}

    if not servers_config:
        return None

    config_data: dict[str, object] = {"mcpServers": servers_config}
    config_path = workspace / ".mcp-pool-config.json"
    tmp_path = config_path.with_suffix(".tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(json.dumps(config_data, indent=2))
    tmp_path.rename(config_path)
    return config_path
