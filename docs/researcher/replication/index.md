# Replication workflow

The
[Calibrated Explanations Studies](https://github.com/kristinebergs/calibrated-explanations-studies)
repository is authoritative for reproduction commands, environments, datasets,
random seeds, historical CE requirements, and expected result artefacts.

## Handoff workflow

1. Open the
   [studies repository](https://github.com/kristinebergs/calibrated-explanations-studies).
2. Select the study corresponding to the paper or result.
3. Follow that study's README.
4. Install the CE version and dependencies specified by the study.
5. Run the study-specific scripts or notebooks.
6. Compare the outputs with the bundled result artefacts.

Historical studies may require historical CE versions. The latest CE release
is not guaranteed to reproduce an older published result exactly.

General method documentation and formal explanation semantics remain in this
core documentation. For the theory and its limits, see
{doc}`../../foundations/concepts/calibrated_interval_semantics`.

Entry-point tier: Tier 2.
