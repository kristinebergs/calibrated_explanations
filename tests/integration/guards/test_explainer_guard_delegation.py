import numpy as np
from sklearn.tree import DecisionTreeClassifier

from calibrated_explanations import CalibratedExplainer


class DummyGuardAlwaysReject:
    def __init__(self):
        self._fitted = True

    def accept(self, x_new, calibrated_prediction=None):
        return False

    def accept_batch(self, x_new_batch, calibrated_predictions=None):
        return np.zeros(len(x_new_batch), dtype=bool)


def test_explainer_set_guard_and_delegate():
    # Small calibration set
    x_cal = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    y_cal = np.array([0, 1, 0, 1])

    model = DecisionTreeClassifier()
    model.fit(x_cal, y_cal)

    explainer = CalibratedExplainer(model, x_cal, y_cal, mode="classification")

    # Set dummy guard via public API
    dummy = DummyGuardAlwaysReject()
    explainer.set_guard(dummy)

    # Public property returns delegator guard
    assert explainer.guard is dummy

    # _accept should delegate to the orchestrator and thus return False
    assert explainer._accept(x_cal[0]) is False
