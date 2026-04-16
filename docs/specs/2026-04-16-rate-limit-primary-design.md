# Rate Limit Primary Flag — Design Spec

**Status:** DRAFT
**Date:** 2026-04-16
**Author:** Legion
**Depends on:** B1+B5 fixes (implemented same session)

## Problem

`ExecutionResult.rate_limited` is a boolean that conflates detection with causation.
CLI instruments log rate limit retries to stderr even when they handle them internally
and succeed. The current system has no way to express "rate limiting happened but was
not the reason for failure."

### Failure Modes (Fixed by B1+B5, Prevented by B6)

1. **Successful execution discarded:** Gemini-cli retries 429 internally, succeeds,
   but stderr contains "429". Marianne classifies as rate_limited. Baton discards
   the successful work, sets sheet to WAITING. Fixed by B5 (success takes priority).

2. **Real error masked:** Gemini-cli fails with sandbox error (exit 1), but stderr
   also contains "429" from startup. Marianne classifies as rate_limited, masking
   the real error. Baton waits instead of reporting the sandbox failure. Fixed by
   B1 (success check in backend).

3. **Rate limit cascade across instruments:** False positive rate limit on one
   instrument triggers fallback cascade. Circuit breaker trips based on false
   positives. Other sheets on the same instrument are affected. Not fully fixed
   by B1+B5 — requires B6 to distinguish primary vs incidental rate limiting.

## Design

### New Field on ExecutionResult

```python
@dataclass
class ExecutionResult:
    # Existing
    rate_limited: bool = False

    # New: was rate limiting the PRIMARY cause of failure?
    rate_limit_primary: bool = False
```

**Semantics:**
- `rate_limited=True, rate_limit_primary=True` — Rate limit caused the failure.
  Instrument could not complete. Baton should wait and retry.
- `rate_limited=True, rate_limit_primary=False` — Rate limit detected in stderr
  but was NOT the cause of failure. Either the instrument retried internally and
  succeeded, or the instrument failed for a different reason. Baton should treat
  as normal success/failure.
- `rate_limited=False` — No rate limit detected. `rate_limit_primary` is always
  False when `rate_limited` is False.

### Classification Logic in cli_backend.py

```python
def _parse_result(self, stdout, stderr, exit_code, duration):
    is_success = exit_code in errors.success_exit_codes
    rate_detected = self._check_rate_limit_detected(stderr)
    error_type = self._classify_output_errors(stdout, stderr, is_success=is_success)

    # Rate limit is primary ONLY when:
    # 1. Rate limit pattern was detected in stderr
    # 2. Execution failed (not success)
    # 3. No other error classification explains the failure
    rate_primary = rate_detected and not is_success and error_type is None

    return ExecutionResult(
        success=is_success,
        rate_limited=rate_detected,
        rate_limit_primary=rate_primary,
        error_type=error_type,
        ...
    )
```

Note: `_check_rate_limit()` renamed to `_check_rate_limit_detected()` to clarify
it detects presence, not causation. The `is_success` guard from B1 is removed from
detection (it now always scans) and moved to the primary classification logic.

### Consumer Changes

**Baton core (`_handle_attempt_result`):**
```python
if event.rate_limit_primary:
    # Genuine rate limit failure — wait and retry
    sheet.status = BatonSheetStatus.WAITING
    self._inbox.put_nowait(RateLimitHit(...))
    return

# If rate_limited but not primary, fall through to normal handling.
# The instrument either succeeded or failed for a different reason.
```

**SheetAttemptResult event:**
Add `rate_limit_primary: bool = False` field. The musician passes through from
ExecutionResult.

**Dashboard/status display:**
When `rate_limited=True, rate_limit_primary=False`, show "(rate limit handled
internally)" instead of the misleading "rate_limit" status tag.

**Learning store:**
Pattern detection should use `rate_limit_primary` for instrument health tracking.
Incidental rate limits (`rate_limited=True, rate_limit_primary=False`) should not
count toward circuit breaker thresholds.

### Migration from B1+B5

B1+B5 are immediate fixes that prevent the worst damage:
- B1: `_check_rate_limit` returns False on success (prevents false positives at source)
- B5: Baton checks `execution_success` before `rate_limited` (prevents work discard)

B6 replaces both with a cleaner architecture:
- B1 is subsumed: detection always runs, but `rate_limit_primary` is False for successes
- B5 is subsumed: baton checks `rate_limit_primary`, not `rate_limited`
- Additionally: failed executions with incidental rate limit patterns are correctly
  classified by their primary error, not masked as rate limits

### Backward Compatibility

- `rate_limited` field remains on ExecutionResult (not removed)
- `rate_limit_primary` defaults to False (existing code that doesn't set it is safe)
- Baton falls back to `rate_limited` when `rate_limit_primary` is not set
  (handles results from older backends)

## File Change Map

| File | Change |
|------|--------|
| `backends/base.py` | Add `rate_limit_primary` to ExecutionResult |
| `execution/instruments/cli_backend.py` | Classification logic: detect vs primary |
| `daemon/baton/events.py` | Add `rate_limit_primary` to SheetAttemptResult |
| `daemon/baton/core.py` | Check `rate_limit_primary` instead of `rate_limited` |
| `daemon/baton/musician.py` | Pass through from ExecutionResult |
| `daemon/baton/state.py` | InstrumentState circuit breaker uses primary only |

## Test Strategy

- Unit: `rate_limited=True, success=True` produces `rate_limit_primary=False`
- Unit: `rate_limited=True, success=False, error_type=None` produces `rate_limit_primary=True`
- Unit: `rate_limited=True, success=False, error_type="AUTH_FAILURE"` produces `rate_limit_primary=False`
- Integration: gemini-cli with internal 429 retries succeeds — sheet completes, not WAITING
- Integration: gemini-cli with sandbox error + 429 in stderr — error classified correctly
- Adversarial: rapid rate limit + success oscillation — no state corruption
