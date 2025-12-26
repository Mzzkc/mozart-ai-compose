#!/usr/bin/env python3
"""Reset a sheet for re-testing.

Usage:
    python scripts/reset-sheet.py STATE_FILE SHEET_NUM [--delete-files]

Example:
    python scripts/reset-sheet.py my-workspace/my-job.json 8
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


def reset_sheet(state_file: Path, sheet_num: int, delete_files: bool = False):
    """Reset a sheet to pending state for re-testing."""

    # Load state
    state = json.loads(state_file.read_text())

    sheet_key = str(sheet_num)
    if sheet_key not in state["sheets"]:
        print(f"Sheet {sheet_num} not found in state")
        return

    print(f"Resetting sheet {sheet_num}...")
    print(f"  Previous status: {state['sheets'][sheet_key]['status']}")
    print(f"  Previous attempts: {state['sheets'][sheet_key]['attempt_count']}")

    # Reset sheet state
    state["sheets"][sheet_key] = {
        "sheet_num": sheet_num,
        "status": "pending",
        "started_at": None,
        "completed_at": None,
        "attempt_count": 0,
        "exit_code": None,
        "error_message": None,
        "error_category": None,
        "validation_passed": False,
        "validation_details": None,
        "completion_attempts": 0,
        "passed_validations": [],
        "failed_validations": [],
        "last_pass_percentage": None,
        "execution_mode": None,
    }

    # Update job state
    state["last_completed_sheet"] = sheet_num - 1
    state["current_sheet"] = sheet_num
    state["status"] = "running"
    state["updated_at"] = datetime.utcnow().isoformat()

    # Write updated state
    state_file.write_text(json.dumps(state, indent=2))
    print(f"  State updated: sheet {sheet_num} reset to pending")

    if delete_files:
        # Find and delete sheet files
        workspace = state_file.parent
        patterns = [
            f"sheet{sheet_num}-*.md",
        ]
        for pattern in patterns:
            for f in workspace.glob(pattern):
                print(f"  Deleting: {f}")
                f.unlink()


def main():
    parser = argparse.ArgumentParser(description="Reset a sheet for re-testing")
    parser.add_argument("state_file", type=Path, help="Path to state JSON file")
    parser.add_argument("sheet_num", type=int, help="Sheet number to reset")
    parser.add_argument("--delete-files", action="store_true",
                       help="Also delete sheet output files")

    args = parser.parse_args()

    if not args.state_file.exists():
        print(f"State file not found: {args.state_file}")
        return 1

    reset_sheet(args.state_file, args.sheet_num, args.delete_files)
    print("\nTo run the sheet:")
    print(f"  cd {args.state_file.parent.parent}")
    print(f"  mozart run mozart-sheet-review.yaml")

    return 0


if __name__ == "__main__":
    exit(main())
