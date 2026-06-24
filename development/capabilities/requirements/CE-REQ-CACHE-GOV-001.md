# CE-REQ-CACHE-GOV-001 - Caching Runtime Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-CACHE-GOV-001 |
| obligation_type | runtime_behavior |
| claim_refs | CE-CAP-CACHE-001 |
| adr_refs | ADR-003 |
| status | active |
| verification_status | verified |
| tif_exemption | repository_policy |
| tif_exemption_rationale | Cache internals (key construction, LRU eviction, disabled-state behavior) are implementation details verified by unit tests targeting the cache layer directly; WrapCalibratedExplainer scenarios cannot observe cache policy at the required granularity. |

## Scope

Public cache behavior and runtime caching helpers governed by ADR-003: deterministic cache keys, opt-in disabled state, LRU eviction, and visible fallback behavior.

## Observable behavior

- Cache keys include namespace, version, and all key parts so different cache domains cannot collide silently.
- Calibrator cache operations preserve the disabled/default-off state.
- LRU cache behavior evicts entries when size limits are exceeded.
- Cache fallback behavior remains explicit and covered by cache fallback tests.

## Acceptance criterion

- `make_key(namespace, version, parts)` produces keys that encode namespace/version/parts distinctions.
- The calibrator cache disabled state avoids storing values while preserving observable metrics.
- LRU cache implementations evict the least-recently-used value when the configured memory budget is exceeded.
- Fallback cache logic is exercised by automated tests, not prose review.

## Verification method

Automated pytest tests for cache key construction, disabled-state behavior, LRU eviction, and fallback logic.

## Verification targets

- pytest: tests/unit/cache/test_cache_fallback.py::test_make_key_should_include_namespace_version_and_parts
- pytest: tests/unit/perf/test_cache.py::test_calibrator_cache_handles_disabled_state
- pytest: tests/unit/cache/test_cache_fallback.py::test_lru_cache_should_evict_when_exceeding_memory_budget
- pytest: tests/unit/cache/test_cache_fallback.py::test_cache_fallback_logic

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies runtime cache contracts named in `CE-CAP-CACHE-001`. It does not prove cache performance under production workloads or every possible cache backend implementation.
