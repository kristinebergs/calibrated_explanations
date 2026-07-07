# CE-REQ-EXPL-CONJ-001 — SUPERSEDED

## Status

**This requirement has been superseded.** It was a monolithic API contract that
combined API availability, return contract, rule semantics, and parameter
semantics into a single requirement without TIF decomposition.

It was replaced by the following focused requirements:

| New requirement | Covers |
|---|---|
| [CE-REQ-EXPL-CONJ-API-001](CE-REQ-EXPL-CONJ-API-001.md) | API availability on public explanation objects |
| [CE-REQ-EXPL-CONJ-RETURN-001](CE-REQ-EXPL-CONJ-RETURN-001.md) | Return type and collection cardinality contract |
| [CE-REQ-EXPL-CONJ-RULE-001](CE-REQ-EXPL-CONJ-RULE-001.md) | Multi-feature conjunction rule semantics |
| [CE-REQ-EXPL-CONJ-PARAM-001](CE-REQ-EXPL-CONJ-PARAM-001.md) | max_rule_size=1 parameter semantics |
| [CE-REQ-EXPL-CONJ-DOC-001](CE-REQ-EXPL-CONJ-DOC-001.md) | Documentation boundary |

**Do not reference this requirement in new claims, TIF interfaces, or tests.**
Use the replacements above.

## Metadata (historical — superseded, not evaluated by policy checkers)

| Former field | Value |
|---|---|
| requirement_id | CE-REQ-EXPL-CONJ-001 |
| claim_refs | CE-CAP-EXPL-CONJ-001 |
| adr_refs | ADR-008 |
| status | superseded |
| verification_status | superseded |
| superseded_by | CE-REQ-EXPL-CONJ-API-001, CE-REQ-EXPL-CONJ-RETURN-001, CE-REQ-EXPL-CONJ-RULE-001, CE-REQ-EXPL-CONJ-PARAM-001, CE-REQ-EXPL-CONJ-DOC-001 |
| superseded_reason | Monolithic requirement replaced by TIF-decomposed requirements (TIF architecture hardening) |
