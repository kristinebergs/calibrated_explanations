# Guard design: threshold-free, calibrated-context-only

This document describes the proposed implementation for the Guard subsystem.
This implementation intentionally drops backward compatibility. It does not use class-specific logic for classification, nor fixed numeric thresholds for regression. It only supports "calibrated context" mode: the guard requires calibration/holdout data to compute its metrics.

The design is based on the comprehensive documentation in `improvement_docs/guards/`, particularly:

- `GUARD_DESIGN_CONFIDENCE_MODULATION.md`: Primary design and implementation specification for confidence-modulated conformal regions.

- `GUARD_MATHEMATICAL_FOUNDATIONS.md`: Rigorous mathematical theory, conformal prediction proofs, and formal guarantees.

- `ANALYSIS_SUMMARY_CALIBRATED_PREDICTION_CONTEXT.md`: Historical analysis and design evolution, justifying the elimination of thresholds and categorical contexts.

- `GUARD_FORMAL_GUARANTEES.md`: Coverage theorems and validity guarantees.

- `GUARD_CALIBRATED_PREDICTION_CONTEXT_ANALYSIS.md`: Detailed analysis of calibrated prediction contexts.

## New Solution: Confidence-Modulated Conformal Regions

The guard implements **confidence-modulated conformal regions** to filter out-of-distribution perturbations during explanation generation. It uses **continuous calibrated confidence** to modulate the conformal acceptance criterion, eliminating the need for thresholds or categorical contexts.

### Key Principles

- **No Thresholds:** Regression guard works without any fixed numeric threshold; uses calibrated predictions and intervals directly.

- **No Class Dependence:** Classification logic is class-agnostic; same mechanism applies to both classification and regression.

- **Calibrated Context Only:** Requires calibrated predictions with uncertainty intervals (from `CalibratedExplainer`).

- **Fitting with Training Instances:** The guard must be fitted with training data (`X_train`, `y_train`), a fitted model, and an interval learner to define the conformal regions.

- **Single Global Clustering:** Clusters data in feature space to capture heteroscedasticity and compute per-cluster conformal radii.

- **Confidence Modulation:** Uses interval width from calibrated predictions to adapt acceptance strictness per-instance.

### Mathematical Foundation

Based on conformal prediction theory (see `GUARD_MATHEMATICAL_FOUNDATIONS.md`):

- Conformal prediction provides finite-sample coverage guarantees without distributional assumptions.

- Nonconformity measures (e.g., Mahalanobis distance) quantify how unusual a point is.

- Inductive conformal prediction splits data into proper set (for training) and calibration set (for radius computation).

- Confidence modulation adjusts the effective radius based on model uncertainty: wider intervals (lower confidence) allow more lenient acceptance.

### Implementation Overview

The `ConformalRegionOracle` class:

- **Fit Method:** Accepts `X_train`, `y_train`, `model`, and `interval_learner` to fit clustering, compute nonconformity scores, and derive conformal radii per cluster.

- **Evaluate/Accept Method:** For a new point, computes Mahalanobis distance to nearest cluster, modulates radius by calibrated interval width, and accepts if distance ≤ effective radius.

- **No Threshold Parameter:** Eliminates `threshold` for regression; no `context_mode` or categorical binning.

- **Unified API:** Same implementation for classification and regression; only requires calibrated intervals.

### API Sketch

```python
class ConformalRegionOracle:
    def __init__(self, alpha: float = 0.1, relaxation_factor: float = 1.0, n_clusters: int = 5):
        # alpha: conformal miscalibration level
        # relaxation_factor: controls leniency for uncertain predictions
        # n_clusters: number of feature-space clusters

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, model, interval_learner):
        # Fit clustering on X_train, compute radii using calibration split
        # interval_learner provides calibrated intervals for confidence modulation

    def accept(self, x: np.ndarray, calibrated_prediction) -> bool:
        # calibrated_prediction: (pred, [L, U]) from CalibratedExplainer
        # Compute effective radius modulated by interval width
        # Return True if within conformal region
```

### Fitting with Training Instances

It must be possible to submit training instances to define/fit the guard. The `fit` method takes `X_train` and `y_train` (full training data), splits internally into proper and calibration sets, trains clustering on proper set, and computes radii on calibration set. This ensures the guard learns representative regions and nonconformity distributions from the training distribution.

### Edge Cases & Handling

- **Insufficient Data:** If training data is too small, raise error or flag insufficient data.

- **No Calibrated Intervals:** Raise UsageError if interval_learner not provided or intervals unavailable.

- **High-Dimensional Data:** Use covariance regularization (e.g., shrunk covariance) to handle ill-conditioned matrices.

- **Single Cluster:** Fallback to global radius if clustering fails.

### Testing Strategy

- Unit tests: Verify radius computation, modulation function, and acceptance logic with synthetic data.

- Integration tests: Fit guard on training data, test acceptance on in-distribution and OOD points.

- Test files: Append to nearest existing guard test files per repo rules.

### Implementation Files

- Update `src/calibrated_explanations/guards/regions.py` (current implementation) to implement `ConformalRegionOracle` as described.

- Reference `GUARD_DESIGN_CONFIDENCE_MODULATION.md` for detailed pseudocode and parameter guidance.

### Next Steps

1. Implement `ConformalRegionOracle` skeleton with fit and accept methods.

2. Add unit tests for core logic.

3. Integrate with `CalibratedExplainer` for interval access.

4. Run tests and iterate.


