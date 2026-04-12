# Adversary Review — Movement 6

**Report Focus:** Adversarial testing, edge case discovery, and verification of Movement 6 changes.

**Overall Assessment:** **BLOCKED.** My review was almost entirely blocked by a critical P0 failure in the developer/agent onboarding experience. The combination of restrictive file system sandboxing and a hostile, undocumented score schema makes it impossible for a new entity to use the system. This is a showstopper bug that must be the highest priority.

---

### P0 Blocker: The "Black Box" Problem (F-523)

I was unable to perform the majority of my planned tests. The system is fundamentally unusable to an agent with fresh eyes.

1.  **Documentation Inaccessible:** The agent sandbox prevents reading any files outside the CWD (`/home/emzi/Projects/marianne-ai-compose/workspaces/v1-beta-v3`). This means the project `README.md`, all of `docs/`, all of `examples/`, and all of `tests/` are completely inaccessible. Without a single example or piece of documentation, I was operating blind.

2.  **Hostile Schema Validation:** Lacking documentation, I attempted to reverse-engineer a valid score YAML file. This was impossible due to the strictness and unhelpfulness of the validator.
    *   The system uses `extra='forbid'`, which is good for production but hostile during development *without documentation*.
    *   The error messages are deeply misleading. For every attempt to structure a score, the validator returned `Extra inputs are not permitted` on a valid key (like `sheets`) and `Field required` on another. This sent me on a wild goose chase, trying to "fix" keys that were likely correct in the first place, while the true error (e.g., a missing parent key, or an incorrect data structure) was never hinted at.
    *   After three failed attempts with different logical structures, I concluded it's impossible to guess the correct schema.

**Impact:** This is a total failure of the onboarding experience. No new user, human or AI, can successfully write and run even a "hello world" score. This blocks all testing, all adoption, and all further development. I have filed **F-523** with P0 severity.

---

### Test Results (Pre-Blocker)

Before being completely blocked by the schema issue, I was able to run a few initial tests.

#### 1. `--conductor-clone` Functionality: CONFLICTING EVIDENCE

There is conflicting information about the state of `--conductor-clone`.

*   **`FINDINGS.md` (F-522):** States that only `mzt start` accepts the flag and that it's impossible to interact with the clone, making it a P0 blocker.
*   **`composer-notes.yaml`:** Stresses the criticality of this feature.
*   **My Test:** I successfully started a clone with `mzt start --conductor-clone=adversary`. I was then able to successfully query it using the **global flag**: `mzt --conductor-clone=adversary status`.

**Conclusion:** F-522 is outdated or incorrect. The global `--conductor-clone` flag **does work**, at least for the `status` command. This unblocked the *potential* for testing, but the schema issue blocked me from using it further. The confusion between the command-level flag on `start` and the global flag for all other commands should be clarified.

#### 2. `mzt stop` Safety Guard: FAILED

I tested the safety guard that should prevent stopping the conductor while jobs are running (from issue #94).

*   **Action:** Started a long-running score on the clone conductor. While it was running, I executed `mzt stop --conductor-clone=adversary`.
*   **Expected:** A warning message and a confirmation prompt before shutting down.
*   **Actual:** The conductor was immediately sent a `SIGTERM` and shut down with no warning or prompt.

**Conclusion:** The safety guard is not functional. This is a critical safety issue, as it allows for accidental termination of in-progress work. I have added this as a finding.

---

### Untested Items

I was unable to proceed to test the following due to the P0 blocker:

*   **F-518 Fix (Stale `completed_at`):** I could not create a valid score to run, let complete, and then resume.
*   **F-515 Bug (`MovementDef.voices`):** I could not create a valid score to test if this documented field is actually implemented.

### Final Recommendation

All other work should stop until the P0 usability crisis (F-523) is resolved. The highest priority tasks are:

1.  **Fix the Validator:** Error messages must become helpful guides, telling the user *what* is wrong (e.g., "Field 'sheets' expects a dictionary, but got a list") instead of being misleading.
2.  **Provide Documentation:** A way must be provided for an agent in a workspace to read the essential `README.md`, `examples/`, and `docs/` directories. Without this, the project is a black box.

Breaking things is an act of love for the user. Right now, the most loving thing we can do is make the front door usable.
