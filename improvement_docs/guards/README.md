# Guards Expansion Analysis – Complete Documentation

**Analysis Date:** November 11, 2025
**Analyst:** GitHub Copilot
**Status:** ✅ COMPLETE

---

## Overview

This analysis provides an in-depth examination of the **Perturbation Guards expansion** for calibrated explanations, addressing three critical questions:

1. **Does the guards expansion need improvements?**
2. **What guarantees can be provided for in-distribution perturbations?**
3. **How can we evaluate when guarded CE succeeds vs unguarded CE fails?**

---

## Analysis Documents

### Document 1: Quick Reference
**File:** `GUARDS_QUICK_REFERENCE.md`
**Length:** 3 pages
**Audience:** Developers, stakeholders, quick decision-makers
**Contains:**
- Summary of three key questions + answers
- Status table (architecture, implementation, integration)
- Five key evaluation metrics with procedures
- Implementation checklist (Phase 1, 2, 3)
- Code locations reference

**Start here for:** Rapid overview, implementation plan, quick lookup

---

### Document 2: Visual Summary
**File:** `GUARDS_VISUAL_SUMMARY.md`
**Length:** 8 pages
**Audience:** Architects, evaluation researchers, documentation writers
**Contains:**
- Status dashboard (what works, what doesn't)
- Guarantee hierarchy (3 tiers of guarantees)
- Failure mode map (5 OOD failure scenarios)
- Test suite design (3 concrete test cases)
- Diagnostic tools (proposed additions)
- Roadmap (phases 1-3)

**Start here for:** Understanding architecture, evaluation design, failure modes

---

### Document 3: Executive Summary
**File:** `GUARDS_ANALYSIS_SUMMARY.md`
**Length:** 6 pages
**Audience:** Project leads, release managers, technical writers
**Contains:**
- High-level findings (gaps, issues, improvements needed)
- Formal problem statement for guarantees
- Evaluation framework with metrics + test cases
- Recommended action plan (short/medium/long-term)
- Stakeholder questions
- References to code and ADRs

**Start here for:** Decision-making, release planning, stakeholder communication

---

### Document 4: Detailed Analysis (Internal)
**File:** `GUARDS_EXPANSION_ANALYSIS.md`
**Length:** 15 pages
**Audience:** Implementation team, architecture review committee
**Contains:**
- Full system architecture walkthrough
- Line-by-line code analysis
- Theoretical gaps with formal definitions
- Complete test case specifications
- Performance optimization opportunities
- Integration requirements

**Start here for:** Implementation, code review, detailed technical planning

---

## Key Findings Summary

### Finding 1: Integration is Incomplete (CRITICAL)

**Status:** Helpers `_label_ctx()` and `_accept()` are defined but never called.

**Impact:** Guards are instantiated but non-functional. Perturbations are never filtered.

**Fix required:** Wire guards into perturbation generation loop.

### Finding 2: Formal Guarantees Need Clarification (HIGH)

**Issue:** Conformal calibration guarantees apply to random samples, not structured perturbations.

**Impact:** Users may have false confidence in "in-distribution" perturbations.

**Fix required:** Document guarantees formally (ADR-028) and implement martingale e-test.

### Finding 3: No Evaluation Framework (CRITICAL)

**Status:** No tests to show when unguarded CE fails vs guarded CE succeeds.

**Impact:** Cannot measure actual benefit or performance impact.

**Fix required:** Implement test suite (synthetic + real datasets) with 5 key metrics.

---

## Recommended Next Steps

### Immediate (v0.10.0)
1. Complete pipeline integration (wire guards into perturbation loop)
2. Implement martingale e-test (currently stubbed)
3. Add performance optimization (S_total caching)
4. Create unit tests

### Short-term (v0.11.0)
1. Write ADR-028 formalizing guarantees
2. Run full evaluation suite
3. Create user documentation (`docs/guards.md`)

### Long-term (Post-v1.0.0)
1. Make guards default (opt-out instead of opt-in)
2. Add advanced features (full covariance, adaptive alpha)

---

## Document Navigation

**For different roles:**

| Role | Start with | Then read |
|------|-----------|-----------|
| **Project Lead** | Quick Reference | Executive Summary |
| **Software Engineer** | Visual Summary | Detailed Analysis |
| **Researcher** | Executive Summary | Visual Summary |
| **Architect** | Quick Reference | Detailed Analysis |
| **Product Manager** | Quick Reference | Executive Summary |
| **QA/Test Engineer** | Visual Summary | Quick Reference (Metrics section) |

---

## Key Metrics for Evaluation

When guards are properly integrated, measure:

1. **Explanation Stability** (primary): 20%+ improvement in rule agreement
2. **OOD Detection Rate**: ~α (e.g., 10% for α=0.1)
3. **Fairness Parity**: <5pp disparity across classes
4. **Computational Overhead**: ≤20% slowdown
5. **Coverage Calibration**: Good agreement between estimated and actual coverage

---

## Code References

**Core Implementation (195 LOC):**
- `src/calibrated_explanations/guards/regions.py` – ConformalRegionOracle
- `src/calibrated_explanations/guards/__init__.py` – BaseGuard protocol

**Integration Points (4,100 LOC):**
- `src/calibrated_explanations/core/calibrated_explainer.py`
  - Lines 715–939: Guard initialization
  - Lines 945–953: Helper methods
  - Lines 1772–1803: Guard management

**Missing Implementation:**
- Perturbation filtering (must be added to explain pipeline)
- Martingale e-test (stubbed, needs completion)
- Evaluation framework (needs creation)

---

## Guarantee Framework (Recommended)

**Tier 1: Conformal Membership** (Default)
- Guarantee: Marginal coverage ≥ (1-α)
- Overhead: O(n_clusters)
- When: Always use as baseline

**Tier 2: Martingale E-Test** (Optional)
- Guarantee: Anytime-valid OOD test
- Overhead: O(1)
- When: High-stakes + small calibration sets
- Status: TODO

---

## Questions for Stakeholders

1. Should guards be **opt-in (current)** or **opt-out (v1.0+)**?
2. Is **90% coverage (α=0.1)** the right default?
3. **Martingale e-test:** v0.10.0 or deferrable?
4. **Evaluation scope:** Synthetic only, or real datasets?
5. **Production feedback:** Any reported OOD perturbation issues?

---

## Timeline Estimate

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1 (Integration) | 4-6 weeks | Functional guards + tests + benchmarks |
| Phase 2 (Guarantees) | 4-8 weeks | ADR-028 + evaluation report + user guide |
| Phase 3 (Enhancement) | Ongoing | Default setting, advanced features, etc. |

---

## How to Use These Documents

### For Quick Decisions (5 min)
→ Read **Quick Reference** (pages 1-2)

### For Understanding the Problem (20 min)
→ Read **Executive Summary** + **Visual Summary** (pages 1-5 each)

### For Implementation (1-2 hours)
→ Read **Quick Reference** (implementation checklist) + **Detailed Analysis** (architecture + code)

### For Evaluation Design (30 min)
→ Read **Visual Summary** (failure modes + metrics section)

### For Stakeholder Communication (10 min)
→ Use **Quick Reference** (summary table) + **Executive Summary** (finding 1-3)

---

## Conclusion

The guards expansion is **architecturally sound** but **functionally incomplete**. The core issue is that guards are instantiated but never actually used during explanation generation.

**Primary action:** Complete pipeline integration so guards actually filter perturbations.

**Secondary action:** Formalize guarantees and implement evaluation framework to measure benefits.

**Expected outcome:** More stable, trustworthy explanations that avoid spurious rules from OOD perturbations.

---

## Document Metadata

| Property | Value |
|----------|-------|
| Analysis Date | 2025-11-11 |
| Analysis Duration | 2-3 hours |
| Total Pages | 32 pages (across 4 documents) |
| Code Analyzed | ~200 LOC (guards) + ~4100 LOC (integration) |
| References | Implementation instructions, ADRs, release plan |
| Status | ✅ Complete, ready for review |

---

**All analysis documents are in `improvement_docs/` directory.**

**Analysis completed and ready for next phase: stakeholder review, implementation planning, or both.**
