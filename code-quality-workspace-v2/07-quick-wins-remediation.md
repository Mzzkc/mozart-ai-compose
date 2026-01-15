# Movement III-A: Quick Wins Remediation

**Executed:** 2026-01-15
**Sheet:** 7 of 9

---

## Fixes Applied

| Fix | Location | Type | Verified |
|-----|----------|------|----------|
| Fix README batch→sheet terminology | `README.md:35-353` | terminology | ✓ |
| Standardize GitHub URL | `README.md:21` | URL fix | ✓ |
| Use RetryDelays constants | `retry_strategy.py:420,775` | magic numbers | ✓ |
| Fix logger bypass (notify) | `notifications/base.py:263-267` | logging consistency | ✓ |
| Fix logger bypass (close) | `notifications/base.py:425-429` | logging consistency | ✓ |
| Remove missing doc refs | `docs/getting-started.md:284-286` | dead links | ✓ |
| Extract timeout constants | `claude_cli.py:36-38,345,552` | magic numbers | ✓ |
| Expand TODO context | `runner.py:720-721` | documentation | ✓ |
| Create ErrorMessages class | `cli.py:51-57` | string constants | ✓ |
| Use ErrorMessages constants | `cli.py` (5 locations) | string duplication | ✓ |

---

## Details of Changes

### 1. README batch→sheet Terminology (High Priority)

The README Quick Start section incorrectly used `batch:` in YAML configs when the actual config schema uses `sheet:`. Updated 15+ occurrences:
- Config key: `batch:` → `sheet:`
- Variables: `batch_num` → `sheet_num`, `total_batches` → `total_sheets`
- Descriptions: "batch job" → "sheet job"
- File paths: `batch{batch_num}` → `sheet{sheet_num}`
- Events: `batch_failed` → `sheet_failed`

### 2. GitHub URL Standardization

Changed placeholder URL `github.com/yourusername/mozart-ai-compose` to actual repo URL `github.com/Mzzkc/mozart-ai-compose`.

### 3. RetryDelays Constants Usage

Imported and used `RetryDelays.API_RATE_LIMIT` instead of magic number `3600.0` in:
- `RetryStrategyConfig.max_delay` default
- `_recommend_rate_limit_retry()` fallback value

### 4. Logger Bypass Fixes

Replaced inline `import logging; logging.getLogger(__name__).warning()` with module-level `_logger.warning()` using Mozart's structured logging format in `notifications/base.py`.

### 5. Missing Doc References

Removed dead links to non-existent `configuration.md` and `best-practices.md` in `docs/getting-started.md`, replaced with references to `examples/` and `CLAUDE.md`.

### 6. Timeout Constants

Created named constants at module level:
- `GRACEFUL_TERMINATION_TIMEOUT = 5.0` - seconds to wait for graceful termination
- `STREAM_READ_TIMEOUT = 1.0` - seconds between stream read checks

### 7. TODO Enhancement

Expanded `TODO: Pass concert context for chaining` to include tag and reference:
`TODO(concert-chaining): Pass concert context from parent job to enable job chaining. See ConcertConfig for context structure.`

### 8. ErrorMessages Constants

Created `ErrorMessages` class with constants for user-facing error messages, then updated 5 occurrences of hardcoded "Job not found" string.

---

## Verification Results

- **mypy**: 7 errors (all pre-existing, not from this PR)
- **imports**: ✓ Pass (`from mozart.cli import app` works)
- **tests**: ✓ Pass (1103 passed, 3 warnings in 30.69s)

---

## Files Changed

1. `README.md` - Terminology and URL fixes
2. `docs/getting-started.md` - Removed dead links
3. `src/mozart/cli.py` - ErrorMessages constants
4. `src/mozart/backends/claude_cli.py` - Timeout constants
5. `src/mozart/execution/retry_strategy.py` - RetryDelays usage
6. `src/mozart/execution/runner.py` - TODO enhancement
7. `src/mozart/notifications/base.py` - Logger fixes

---

*Quick Wins Remediation completed by Quality Coordinator*
*Mozart AI Compose Code Quality Review v2 - Movement III-A*
