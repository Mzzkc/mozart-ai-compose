SHEET: 7
PHASE: Runner Integration
RUNNER_CHANGES: Added _setup_isolation() and _cleanup_isolation() methods to JobRunner, modified run() to integrate worktree lifecycle
BACKEND_CHANGES: Backend working_directory override via getattr/setattr for type-safe dynamic access
TESTS_PASSING: yes
BACKWARD_COMPATIBLE: yes
COMMIT_HASH: 06d489b
IMPLEMENTATION_COMPLETE: yes
