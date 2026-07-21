# Research Directions

This page tracks open research questions related to calibrated explanations.
These are **research directions, not release commitments**: none carry a
target version, and none imply a guarantee beyond what is currently
established (see {doc}`../foundations/concepts/calibrated_interval_semantics`
for the calibration guarantees CE currently makes and their explicit
non-guarantees).

Engineering work that is scoped, evidenced, and actively planned lives in
`development/current-work/RELEASE_PLAN.md`, not here.

## Open research questions

### Distribution-free coverage guarantees for feature-importance rankings

CE provides calibrated prediction intervals (classification probabilities and
regression intervals). It does not currently provide, and does not claim,
distribution-free coverage guarantees for feature-importance rankings or
simultaneous feature comparisons. Whether such a guarantee is even
theoretically attainable for rule-based feature weights is an open question
connected to conformal prediction theory and Venn-Abers calibration.

### Explanation stability across calibration samples

Whether repeated calibration draws (same model, different calibration split)
produce stable feature-weight rankings is not currently measured or
guaranteed. This is a prerequisite question for any future rank-confidence
claim.

### Multi-calibration across intersectional protected attributes

CE currently supports Mondrian/conditional categorizer-based calibration (see
{doc}`../foundations/concepts/index` and ADR-039) for subgroup-aware
uncertainty. Formal multi-calibration guarantees across intersectional
protected attributes (Hébert-Johnson et al., 2018) are not implemented and
would require a dedicated ADR for any fairness-primitive API surface before
implementation could be considered.

### Higher-order feature interaction search

`add_conjunctions()` already builds conjunctive rules iteratively up to a
caller-supplied `max_rule_size`. The open research question is whether a
computationally feasible search exists for interactions beyond what the
current greedy pairwise-growth algorithm covers, and whether calibration
guarantees survive such a search (connections to Shapley interaction indices
and functional ANOVA decompositions).

### Adaptive binning strategies

CE currently uses fixed or Mondrian-based discretization. Whether adaptive
binning (optimizing jointly for explanation fidelity and computational cost)
is worthwhile, and what the right optimization criterion would be, is an open
question connected to conformal-prediction adaptive-binning literature and
information-theoretic discretization.

## Contributing research

If you are working on any of these questions or have a related idea:

1. Open a discussion in [GitHub Discussions](https://github.com/Moffran/calibrated_explanations/discussions).
2. Reference the relevant ADR(s) in `development/adrs/` if your idea implies an
   architectural or API change.
3. Consider prototyping as a plugin (see ADR-006, ADR-013, ADR-015, ADR-026) —
   OSS CE's extension contracts do not require a core change to experiment.
4. For published work, cite the calibrated-explanations framework; see
   {doc}`../citing`.
