#!/usr/bin/env python3
"""Reset a batch for re-testing.

Usage:
    python scripts/reset-batch.py STATE_FILE BATCH_NUM [--delete-files]

Example:
    python scripts/reset-batch.py ~/Projects/Naurva/coordination-workspace/naurva-commit-review.json 8
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


def reset_batch(state_file: Path, batch_num: int, delete_files: bool = False):
    """Reset a batch to pending state for re-testing."""

    # Load state
    state = json.loads(state_file.read_text())

    batch_key = str(batch_num)
    if batch_key not in state["batches"]:
        print(f"Batch {batch_num} not found in state")
        return

    print(f"Resetting batch {batch_num}...")
    print(f"  Previous status: {state['batches'][batch_key]['status']}")
    print(f"  Previous attempts: {state['batches'][batch_key]['attempt_count']}")

    # Reset batch state
    state["batches"][batch_key] = {
        "batch_num": batch_num,
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
    state["last_completed_batch"] = batch_num - 1
    state["current_batch"] = batch_num
    state["status"] = "running"
    state["updated_at"] = datetime.utcnow().isoformat()

    # Write updated state
    state_file.write_text(json.dumps(state, indent=2))
    print(f"  State updated: batch {batch_num} reset to pending")

    if delete_files:
        # Find and delete batch files
        workspace = state_file.parent
        patterns = [
            f"batch{batch_num}-*.md",
        ]
        for pattern in patterns:
            for f in workspace.glob(pattern):
                print(f"  Deleting: {f}")
                f.unlink()


def main():
    parser = argparse.ArgumentParser(description="Reset a batch for re-testing")
    parser.add_argument("state_file", type=Path, help="Path to state JSON file")
    parser.add_argument("batch_num", type=int, help="Batch number to reset")
    parser.add_argument("--delete-files", action="store_true",
                       help="Also delete batch output files")

    args = parser.parse_args()

    if not args.state_file.exists():
        print(f"State file not found: {args.state_file}")
        return 1

    reset_batch(args.state_file, args.batch_num, args.delete_files)
    print("\nTo run the batch:")
    print(f"  cd {args.state_file.parent.parent}")
    print(f"  mozart run mozart-batch-review.yaml")

    return 0


if __name__ == "__main__":
    exit(main())
