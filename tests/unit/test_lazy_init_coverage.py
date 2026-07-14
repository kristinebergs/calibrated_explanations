import subprocess
import sys

import pytest


def test_plotting_deprecation_warning(monkeypatch) -> None:
    """Test that package attribute access no longer exposes deprecated plotting alias."""
    import calibrated_explanations

    monkeypatch.delitem(calibrated_explanations.__dict__, "plotting", raising=False)
    with pytest.raises(AttributeError):
        _ = calibrated_explanations.plotting

    import calibrated_explanations.plotting as plotting_module

    assert plotting_module is not None


def test_import_does_not_register_process_wide_mappingproxy_reducer() -> None:
    script = """
import copyreg
import pickle
from types import MappingProxyType

before = copyreg.dispatch_table.get(MappingProxyType)
import calibrated_explanations
after = copyreg.dispatch_table.get(MappingProxyType)
assert before is after
try:
    pickle.dumps(MappingProxyType({'demo': 1}))
except TypeError:
    pass
else:
    raise AssertionError('standalone MappingProxyType unexpectedly became picklable')
print('ok')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "ok"


def test_unknown_package_attribute_raises_attribute_error() -> None:
    import calibrated_explanations

    with pytest.raises(AttributeError):
        _ = calibrated_explanations.this_symbol_does_not_exist


def test_lazy_package_reject_policy_spec_symbol_is_available() -> None:
    import calibrated_explanations

    assert calibrated_explanations.RejectPolicySpec is not None
