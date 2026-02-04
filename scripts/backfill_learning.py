#!/usr/bin/env python3
"""Backfill global learning store from local outcome files.

Scans workspaces for .mozart-outcomes.json files and imports any
outcomes not already in the global learning database.

Also extracts and processes archived workspaces (tar.gz files).
"""

import json
import sys
import tarfile
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mozart.learning.global_store import GlobalLearningStore
from mozart.learning.outcomes import SheetOutcome, SheetStatus


def load_outcomes_from_file(filepath: Path) -> list[dict]:
    """Load outcomes from a JSON file."""
    try:
        with open(filepath) as f:
            data = json.load(f)

        # Handle both nested and flat structures
        if isinstance(data, dict) and "outcomes" in data:
            return data["outcomes"]
        elif isinstance(data, list):
            return data
        else:
            return []
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return []


def dict_to_sheet_outcome(d: dict) -> SheetOutcome | None:
    """Convert a dict to a SheetOutcome object."""
    try:
        # Map final_status string to enum
        status_str = d.get("final_status", "completed")
        try:
            status = SheetStatus(status_str)
        except ValueError:
            status = SheetStatus.COMPLETED

        return SheetOutcome(
            sheet_id=d.get("sheet_id", "unknown"),
            job_id=d.get("job_id", "unknown"),
            validation_results=d.get("validation_results", []),
            execution_duration=d.get("execution_duration", 0.0),
            retry_count=d.get("retry_count", 0),
            completion_mode_used=d.get("completion_mode_used", False),
            final_status=status,
            validation_pass_rate=d.get("validation_pass_rate", 0.0),
            first_attempt_success=d.get("first_attempt_success", True),
            patterns_detected=d.get("patterns_detected", []),
            timestamp=datetime.fromisoformat(d["timestamp"]) if "timestamp" in d else datetime.now(),
        )
    except Exception as e:
        print(f"  Error converting outcome: {e}")
        return None


def find_outcome_files(root: Path) -> list[tuple[Path, Path]]:
    """Find all outcome files and their workspace paths."""
    results = []
    for outcome_file in root.glob("**/.mozart-outcomes.json"):
        # Skip if in archives (we handle those separately)
        if "archives" in str(outcome_file):
            continue
        workspace = outcome_file.parent
        results.append((outcome_file, workspace))
    return results


def find_archived_workspaces(root: Path) -> list[Path]:
    """Find all archived workspace tar.gz files."""
    archives_dir = root / "memory-bank" / "archives" / "workspaces"
    if not archives_dir.exists():
        return []
    return list(archives_dir.glob("*.tar.gz"))


def process_archive(archive_path: Path, store: GlobalLearningStore, temp_dir: Path) -> tuple[int, int, int]:
    """Extract and process outcomes from an archived workspace.

    Returns: (imported, skipped, errors)
    """
    imported = 0
    skipped = 0
    errors = 0

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            # Look for outcome files in the archive
            outcome_members = [m for m in tar.getmembers()
                            if m.name.endswith(".mozart-outcomes.json")]

            if not outcome_members:
                return 0, 0, 0

            # Extract just the outcome files
            extract_dir = temp_dir / archive_path.stem
            extract_dir.mkdir(exist_ok=True)

            for member in outcome_members:
                tar.extract(member, extract_dir)
                outcome_file = extract_dir / member.name

                # Determine workspace path from archive name
                workspace_name = archive_path.stem  # e.g., "evolution-workspace-v10"
                fake_workspace = Path(f"/archived/{workspace_name}")

                outcomes = load_outcomes_from_file(outcome_file)

                for outcome_dict in outcomes:
                    outcome = dict_to_sheet_outcome(outcome_dict)
                    if outcome is None:
                        errors += 1
                        continue

                    # Check if already exists
                    job_hash = store.hash_job(outcome.job_id)
                    sheet_num = store._extract_sheet_num(outcome.sheet_id)

                    with store._get_connection() as conn:
                        existing = conn.execute(
                            "SELECT COUNT(*) FROM executions WHERE job_hash = ? AND sheet_num = ?",
                            (job_hash, sheet_num)
                        ).fetchone()[0]

                    if existing > 0:
                        skipped += 1
                        continue

                    try:
                        store.record_outcome(outcome, fake_workspace)
                        imported += 1
                    except Exception as e:
                        errors += 1

    except Exception as e:
        print(f"    Error processing archive: {e}")
        errors += 1

    return imported, skipped, errors


def main():
    # Use cwd or explicit path
    root = Path.cwd()
    if not (root / "src" / "mozart").exists():
        root = Path(__file__).parent.parent
    store = GlobalLearningStore()

    print("=" * 60)
    print("Global Learning Store Backfill")
    print("=" * 60)

    # Get current count
    with store._get_connection() as conn:
        before_count = conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0]
    print(f"\nExecutions before: {before_count}")

    total_imported = 0
    total_skipped = 0
    total_errors = 0

    # === PART 1: Active workspaces ===
    outcome_files = find_outcome_files(root)
    print(f"\n--- Active Workspaces ({len(outcome_files)} found) ---\n")

    for outcome_file, workspace in outcome_files:
        print(f"Processing: {outcome_file.relative_to(root)}")
        outcomes = load_outcomes_from_file(outcome_file)
        print(f"  Found {len(outcomes)} outcomes")

        imported = 0
        skipped = 0
        errors = 0

        for outcome_dict in outcomes:
            outcome = dict_to_sheet_outcome(outcome_dict)
            if outcome is None:
                errors += 1
                continue

            # Check if already exists (by job_hash + sheet_num)
            job_hash = store.hash_job(outcome.job_id)
            sheet_num = store._extract_sheet_num(outcome.sheet_id)

            with store._get_connection() as conn:
                existing = conn.execute(
                    "SELECT COUNT(*) FROM executions WHERE job_hash = ? AND sheet_num = ?",
                    (job_hash, sheet_num)
                ).fetchone()[0]

            if existing > 0:
                skipped += 1
                continue

            # Import it
            try:
                store.record_outcome(outcome, workspace)
                imported += 1
            except Exception as e:
                print(f"    Error importing {outcome.sheet_id}: {e}")
                errors += 1

        print(f"  Imported: {imported}, Skipped: {skipped}, Errors: {errors}")
        total_imported += imported
        total_skipped += skipped
        total_errors += errors

    # === PART 2: Archived workspaces ===
    archives = find_archived_workspaces(root)
    print(f"\n--- Archived Workspaces ({len(archives)} found) ---\n")

    if archives:
        # Create temp directory for extraction
        temp_dir = Path(tempfile.mkdtemp(prefix="mozart_backfill_"))
        try:
            for archive in archives:
                print(f"Processing: {archive.name}")
                imported, skipped, errors = process_archive(archive, store, temp_dir)
                if imported > 0 or skipped > 0:
                    print(f"  Imported: {imported}, Skipped: {skipped}, Errors: {errors}")
                total_imported += imported
                total_skipped += skipped
                total_errors += errors
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    # Get final count
    with store._get_connection() as conn:
        after_count = conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0]

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Executions before: {before_count}")
    print(f"Executions after:  {after_count}")
    print(f"New imports:       {total_imported}")
    print(f"Skipped (dupe):    {total_skipped}")
    print(f"Errors:            {total_errors}")


if __name__ == "__main__":
    main()
