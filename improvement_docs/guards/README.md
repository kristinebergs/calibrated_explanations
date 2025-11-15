# Guards Documentation Index

**Date:** November 13, 2025
**Status:** Complete & Final
**Scope:** Confidence-modulated conformal region guard

---

## Document Suite Overview

This directory contains comprehensive documentation for the guard feature, including design, mathematics, formal guarantees, and implementation details.

### 📋 Quick Navigation

**For implementation:**
- Start with → `GUARD_DESIGN_CONFIDENCE_MODULATION.md` (architecture & code)

**For theory:**
- Start with → `GUARD_MATHEMATICAL_FOUNDATIONS.md` (conformal prediction proofs)

**For guarantees:**
- Start with → `GUARD_FORMAL_GUARANTEES.md` (coverage theorems)

**For context:**
- Start with → `ANALYSIS_SUMMARY_CALIBRATED_PREDICTION_CONTEXT.md` (design evolution)

---

## Documents in This Suite

### 1. GUARD_DESIGN_CONFIDENCE_MODULATION.md
**Purpose:** Primary design and implementation specification
**Audience:** Developers, implementers
**Length:** ~800 lines
**Key Sections:**
- Conformal prediction overview (non-technical intro)
- Confidence modulation mechanism
- Implementation algorithms (pseudocode)
- API specification (signatures, parameters)
- Usage examples (classification & regression)
- Parameter selection guidance
- Integration with CalibratedExplainer
- Implementation checklist
- Edge cases and handling

**When to use:**
- Implementing the guard
- Understanding the architecture
- Learning how to use the guard
- Debugging integration issues

---

### 2. GUARD_MATHEMATICAL_FOUNDATIONS.md
**Purpose:** Rigorous mathematical theory and proofs
**Audience:** Researchers, theoreticians, advanced practitioners
**Length:** ~1000 lines
**Key Sections:**
- Exchangeability and i.i.d. assumptions
- Nonconformity measures (definition and examples)
- Conformal prediction sets (formal definition)
- Fundamental theorem (Vovk, Vapnik 1999)
- Proof sketch (exchangeability argument)
- Inductive Conformal Prediction (ICP) framework
- Mondrian conformal prediction (clustering)
- Distance metrics (Euclidean, Mahalanobis, diagonal)
- Confidence-modulated acceptance (framework)
- Modulation function requirements and examples
- Interval width as confidence proxy (VennAbers, CPD)
- Composition of guarantees (full guard)
- Failure modes and mitigations
- Extensions and future work

**When to use:**
- Understanding mathematical foundations
- Reviewing proofs and assumptions
- Writing papers or research using the guard
- Teaching conformal prediction concepts
- Debugging theoretical issues

---

### 3. GUARD_FORMAL_GUARANTEES.md
**Purpose:** Formal theorems and coverage guarantees
**Audience:** Researchers, implementers verifying correctness
**Length:** ~600 lines
**Key Sections:**
- Overview of three core guarantees (G1, G2, G3)
- G1: Conformal Coverage (finite-sample validity)
- G2: Modulation Preserves Coverage (monotonicity)
- G3: In-Distribution Acceptance (Mondrian validity)
- G4: Combined Guard Guarantee (optional extension)
- G5: Calibration Validity (classification-specific)
- When guarantees hold vs. fail (conditions)
- Mitigations for guarantee violations
- Empirical verification procedures
- Testing coverage in practice
- Diagnostic plots for debugging
- Summary table

**When to use:**
- Verifying correctness of implementation
- Writing formal proofs for papers
- Understanding coverage guarantees
- Debugging when coverage fails
- Assessing applicability to new datasets

---

### 4. ANALYSIS_SUMMARY_CALIBRATED_PREDICTION_CONTEXT.md
**Purpose:** Design evolution and justification
**Audience:** Decision-makers, design reviewers
**Length:** ~300 lines
**Key Sections:**
- Final design (confidence modulation)
- Key finding (why we chose this approach)
- Comparison table (vs. thresholds, categorical contexts)
- Implementation overview (API before/after)
- What changes in code
- Theoretical guarantee statement
- Design evolution summary (4 phases)
- Key advantages of final design
- Implementation roadmap
- Next steps for development

**When to use:**
- Understanding why this design was chosen
- Reviewing design decisions
- Comparing with alternative approaches
- Getting quick summary of approach
- Presenting to stakeholders

---

## How These Documents Relate

```
┌─────────────────────────────────────┐
│ ANALYSIS_SUMMARY                    │
│ (Why this design? Design evolution) │
└──────────────┬──────────────────────┘
               │
        ┌──────▼────────┐
        │   Design OK?  │ Yes
        └──────┬────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ GUARD_DESIGN_CONFIDENCE_MODULATION │
│ (How to implement it)               │
│ ├─ Architecture                     │
│ ├─ Algorithms (pseudocode)          │
│ ├─ API specification                │
│ ├─ Usage examples                   │
│ └─ Implementation checklist          │
└──────────────┬──────────────────────┘
               │
        ┌──────▼──────────┐
        │  Implement ok?  │ Yes
        └──────┬──────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ GUARD_FORMAL_GUARANTEES             │
│ (Does it actually work?)            │
│ ├─ G1: Conformal coverage           │
│ ├─ G2: Modulation preserves         │
│ ├─ G3: Mondrian validity            │
│ ├─ When guarantees hold/fail        │
│ └─ Empirical verification           │
└──────────────┬──────────────────────┘
               │
        ┌──────▼──────────┐
        │ Correctness ok? │ Yes
        └──────┬──────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ GUARD_MATHEMATICAL_FOUNDATIONS      │
│ (Why does it work theoretically?)   │
│ ├─ Conformal prediction theory      │
│ ├─ ICP framework                    │
│ ├─ Mondrian conformal prediction    │
│ ├─ Distance metrics                 │
│ ├─ Modulation functions             │
│ ├─ Calibration (VennAbers, CPD)     │
│ ├─ Proofs and derivations           │
│ └─ Extensions                       │
└─────────────────────────────────────┘
```

---

## Reading Paths for Different Audiences

### 👨‍💻 I'm a developer implementing the guard
1. **Start:** `ANALYSIS_SUMMARY_CALIBRATED_PREDICTION_CONTEXT.md` (context)
2. **Read:** `GUARD_DESIGN_CONFIDENCE_MODULATION.md` (implementation spec)
3. **Reference:** `GUARD_FORMAL_GUARANTEES.md` (correctness checks)
4. **Deep dive:** `GUARD_MATHEMATICAL_FOUNDATIONS.md` (if needed)

**Estimated time:** 3-4 hours

---

### 🔬 I'm a researcher reviewing the approach
1. **Start:** `ANALYSIS_SUMMARY_CALIBRATED_PREDICTION_CONTEXT.md` (design rationale)
2. **Read:** `GUARD_FORMAL_GUARANTEES.md` (theorems and proofs)
3. **Deep dive:** `GUARD_MATHEMATICAL_FOUNDATIONS.md` (complete theory)
4. **Reference:** `GUARD_DESIGN_CONFIDENCE_MODULATION.md` (practical details)

**Estimated time:** 2-3 hours

---

### 📊 I'm a practitioner using the guard
1. **Start:** `ANALYSIS_SUMMARY_CALIBRATED_PREDICTION_CONTEXT.md` (quick intro)
2. **Read:** `GUARD_DESIGN_CONFIDENCE_MODULATION.md` (usage examples section)
3. **Reference:** `GUARD_FORMAL_GUARANTEES.md` (when to trust results)
4. **Skip:** `GUARD_MATHEMATICAL_FOUNDATIONS.md` (unless curious)

**Estimated time:** 1-2 hours

---

### 🧪 I want to verify correctness
1. **Start:** `GUARD_FORMAL_GUARANTEES.md` (guarantees & when they hold)
2. **Read:** `GUARD_DESIGN_CONFIDENCE_MODULATION.md` (implementation details)
3. **Deep dive:** `GUARD_MATHEMATICAL_FOUNDATIONS.md` (proofs)
4. **Reference:** `ANALYSIS_SUMMARY_CALIBRATED_PREDICTION_CONTEXT.md` (design context)

**Estimated time:** 2-3 hours

---

## Key Concepts Across Documents

### Conformal Prediction
- **Summary:** `GUARD_DESIGN_CONFIDENCE_MODULATION.md`, Part 1
- **Foundation:** `GUARD_MATHEMATICAL_FOUNDATIONS.md`, Part 2
- **Guarantees:** `GUARD_FORMAL_GUARANTEES.md`, Guarantee G1

### Confidence Modulation
- **Design:** `GUARD_DESIGN_CONFIDENCE_MODULATION.md`, Part 2
- **Framework:** `GUARD_MATHEMATICAL_FOUNDATIONS.md`, Part 6
- **Guarantees:** `GUARD_FORMAL_GUARANTEES.md`, Guarantee G2
- **Why needed:** `ANALYSIS_SUMMARY_CALIBRATED_PREDICTION_CONTEXT.md`, Key Advantages

### Clustering (Mondrian)
- **Algorithm:** `GUARD_DESIGN_CONFIDENCE_MODULATION.md`, Part 3
- **Theory:** `GUARD_MATHEMATICAL_FOUNDATIONS.md`, Part 4
- **Guarantees:** `GUARD_FORMAL_GUARANTEES.md`, Guarantee G3

### Interval Width as Confidence
- **VennAbers:** `GUARD_MATHEMATICAL_FOUNDATIONS.md`, Part 8.1
- **CPD:** `GUARD_MATHEMATICAL_FOUNDATIONS.md`, Part 8.2
- **Advantages:** `GUARD_MATHEMATICAL_FOUNDATIONS.md`, Part 8.3
- **Usage:** `GUARD_DESIGN_CONFIDENCE_MODULATION.md`, Part 2

### Implementation Details
- **API spec:** `GUARD_DESIGN_CONFIDENCE_MODULATION.md`, Part 5
- **Algorithms:** `GUARD_DESIGN_CONFIDENCE_MODULATION.md`, Part 3
- **Pseudocode:** `GUARD_DESIGN_CONFIDENCE_MODULATION.md`, Part 3
- **Checklist:** `GUARD_DESIGN_CONFIDENCE_MODULATION.md`, Part 10

---

## Cross-References for Key Topics

| Topic | Document | Section |
|-------|----------|---------|
| Why eliminate thresholds? | ANALYSIS_SUMMARY | Why This Is Better |
| API design | GUARD_DESIGN | Part 5 |
| fit() pseudocode | GUARD_DESIGN | Part 3 |
| accept() pseudocode | GUARD_DESIGN | Part 3 |
| Modulation function | GUARD_MATHEMATICAL_FOUNDATIONS | Part 7 |
| Mahalanobis distance | GUARD_MATHEMATICAL_FOUNDATIONS | Part 5 |
| Conformal validity | GUARD_FORMAL_GUARANTEES | Guarantee G1 |
| Coverage preservation | GUARD_FORMAL_GUARANTEES | Guarantee G2 |
| In-distribution property | GUARD_FORMAL_GUARANTEES | Guarantee G3 |
| When guarantees fail | GUARD_FORMAL_GUARANTEES | When Guarantees Hold vs. Fail |
| Empirical verification | GUARD_FORMAL_GUARANTEES | Empirical Verification |
| Parameter selection | GUARD_DESIGN | Part 4 |
| Integration with CalibratedExplainer | GUARD_DESIGN | Part 6 |
| Usage examples | GUARD_DESIGN | Part 7 |
| Edge cases | GUARD_DESIGN | Part 8 |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Nov 13, 2025 | Initial complete suite |

**Related versions:**
- Design document v1.0 (ready for implementation)
- Mathematical foundations v1.0 (complete proofs)
- Guarantees v1.0 (formal theorems)
- Analysis v1.0 (design evolution)

---

## Implementation Status

| Phase | Status | Document |
|-------|--------|----------|
| 1. Design & theory | ✅ Complete | All documents |
| 2. Implementation | ⏳ Pending | Use `GUARD_DESIGN_CONFIDENCE_MODULATION.md` as spec |
| 3. Testing | ⏳ Pending | Use `GUARD_FORMAL_GUARANTEES.md` for verification |
| 4. Documentation | ⏳ Pending | Update quickstart & API docs |
| 5. Benchmarking | ⏳ Pending | Use `GUARD_DESIGN_CONFIDENCE_MODULATION.md` Part 4 |

---

## How to Use This Index

**Scenario 1: I just arrived and don't know where to start**
→ Read this document top-to-bottom, then follow "Reading Paths" section

**Scenario 2: I know what I want but need to find it**
→ Use "Cross-References" table or "Quick Navigation" section

**Scenario 3: I'm implementing and need code details**
→ Go to `GUARD_DESIGN_CONFIDENCE_MODULATION.md` Part 3 (Algorithms) and Part 5 (API)

**Scenario 4: Something broke and I need to debug**
→ Check `GUARD_FORMAL_GUARANTEES.md` "When Guarantees Hold vs. Fail" section

**Scenario 5: I need to write a paper**
→ Use `GUARD_MATHEMATICAL_FOUNDATIONS.md` and `GUARD_FORMAL_GUARANTEES.md`

---

## Integration with Codebase

### Source Files to Modify
- `src/calibrated_explanations/guards/regions.py` — Main `ConformalRegionOracle` class
- `src/calibrated_explanations/core/calibrated_explainer.py` — Integration point (`set_guard()`)

### Test Files to Create
- `tests/unit/guards/test_confidence_modulation.py` — Modulation function tests
- `tests/integration/guards/test_guard_integration.py` — End-to-end tests

### Documentation Files to Update
- `docs/api/guards.md` — API reference
- `notebooks/quickstart_guard.ipynb` — Usage examples
- `README.md` — Feature overview

**See:** `GUARD_DESIGN_CONFIDENCE_MODULATION.md` Part 5 for exact specification

---

## Questions & Troubleshooting

**Q: Which document should I read first?**
A: Depends on your role. See "Reading Paths for Different Audiences" section above.

**Q: What's the difference between guarantee G1 and G2?**
A: G1 is base conformal validity; G2 proves modulation preserves it. See `GUARD_FORMAL_GUARANTEES.md`.

**Q: Why no threshold parameter?**
A: It was arbitrary. We use calibrated interval width instead. See `ANALYSIS_SUMMARY_CALIBRATED_PREDICTION_CONTEXT.md`.

**Q: Can I use this for time-series data?**
A: Conformal requires exchangeability. Time-series violates it. See `GUARD_MATHEMATICAL_FOUNDATIONS.md` Part 11.1.

**Q: How do I verify the guard works on my data?**
A: See `GUARD_FORMAL_GUARANTEES.md` "Empirical Verification" section.

**Q: What's the computational cost?**
A: See `GUARD_DESIGN_CONFIDENCE_MODULATION.md` Part 4 (parameter selection & scaling).

---

## Document Maintenance

**Linting Status:** Minor markdown formatting warnings (no content errors)

**To-Do for Finalization:**
- [ ] Fix markdown list formatting warnings (MD032)
- [ ] Add code language tags to pseudocode blocks (MD040)
- [ ] Verify all cross-references are valid
- [ ] Add any missing mathematical references

**Owner:** Guard Feature Team
**Last Updated:** November 13, 2025
**Next Review:** After implementation complete

---

**Document Suite Status:** ✅ COMPLETE AND FINAL

All four core documents are complete, cross-referenced, and ready for:
1. Implementation (use GUARD_DESIGN_CONFIDENCE_MODULATION.md as spec)
2. Code review (use GUARD_FORMAL_GUARANTEES.md for validation)
3. Testing (use GUARD_FORMAL_GUARANTEES.md for test design)
4. Publication (use GUARD_MATHEMATICAL_FOUNDATIONS.md for theory)
