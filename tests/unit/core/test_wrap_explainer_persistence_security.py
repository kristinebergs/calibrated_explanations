"""Security-focused regression tests for the WrapCalibratedExplainer state-persistence
trust boundary (ADR-031 schema v3).

These tests prove the mandatory security invariant: normal
``WrapCalibratedExplainer.load_state()`` must never execute ``pickle.loads``/
``pickle.load``/``joblib.load``/``cloudpickle.loads`` on artifact-provided
bytes, even when every checksum in the artifact is internally consistent.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from calibrated_explanations.core import wrap_explainer as wrap_explainer_module
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
from calibrated_explanations.utils.exceptions import IncompatibleStateError

# --------------------------------------------------------------------------
# Harmless "exploit" payload: a __reduce__ hook that, if ever unpickled,
# writes a sentinel marker file. No weaponized payload is committed; this
# only proves whether load_state() reached pickle.loads() at all.
# --------------------------------------------------------------------------


def _sentinel_side_effect(marker_path: str) -> int:
    Path(marker_path).write_text("pwned", encoding="utf-8")
    return 0


class _MaliciousReducer:
    """Picklable object whose __reduce__ triggers a detectable side effect."""

    def __init__(self, marker_path: str) -> None:
        self.marker_path = marker_path

    def __reduce__(self):
        return (_sentinel_side_effect, (self.marker_path,))


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_saved_wrapper(tmp_path: Path, name: str) -> tuple[Path, RandomForestClassifier]:
    """Produce a real, safe-schema saved artifact to use as a base for tampering."""
    x, y = make_classification(
        n_samples=64,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=1,
    )
    learner = RandomForestClassifier(n_estimators=12, random_state=1)
    wrapper = WrapCalibratedExplainer(learner)
    wrapper.fit(x[:32], y[:32])
    wrapper.calibrate(x[32:48], y[32:48], seed=1)
    state_dir = tmp_path / name
    wrapper.save_state(state_dir)
    return state_dir, learner


# --------------------------------------------------------------------------
# Exploit-regression test
# --------------------------------------------------------------------------


def test_load_state_rejects_legacy_artifact_before_unpickling_malicious_wrapper(
    tmp_path: Path,
) -> None:
    """A self-consistent legacy (schema v1) artifact with a malicious wrapper.pkl and a
    correctly recomputed checksum must be rejected before any unpickling occurs."""
    marker_path = tmp_path / "sentinel-marker.txt"
    malicious_bytes = pickle_dumps_for_test(_MaliciousReducer(str(marker_path)))

    state_dir = tmp_path / "legacy_exploit_state"
    state_dir.mkdir()
    (state_dir / "wrapper.pkl").write_bytes(malicious_bytes)
    manifest = {
        "schema_version": 1,
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "artifact_type": "wrap_calibrated_explainer_state",
        "files": {"wrapper.pkl": _sha256(malicious_bytes)},
    }
    _write_manifest(state_dir / "manifest.json", manifest)

    with pytest.raises(IncompatibleStateError, match="legacy"):
        WrapCalibratedExplainer.load_state(state_dir)

    assert not marker_path.exists(), (
        "load_state() must reject legacy artifacts before any pickle byte is decoded; "
        "the sentinel side effect proves pickle.loads() was never reached."
    )


def pickle_dumps_for_test(obj: Any) -> bytes:
    """Local import indirection: pickle is used here only to construct a harmless,
    local proof-of-concept exploit payload for the test -- never by production code."""
    import pickle  # local: test-only, constructing an attack fixture, not production usage

    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


# --------------------------------------------------------------------------
# Nested calibrator regression
# --------------------------------------------------------------------------


def test_load_state_rejects_python_pickle_calibrator_primitive(tmp_path: Path) -> None:
    """A calibrator_primitive.json declaring calibrator_type='python_pickle' with a
    correct checksum must be rejected without decoding or unpickling the payload."""
    state_dir, learner = _make_saved_wrapper(tmp_path, "nested_pickle_state")
    marker_path = tmp_path / "nested-sentinel.txt"
    malicious_bytes = pickle_dumps_for_test(_MaliciousReducer(str(marker_path)))
    primitive = {
        "schema_version": 3,
        "calibrator_type": "python_pickle",
        "checksums": {"sha256": _sha256(malicious_bytes)},
        "payload": {"pickle_b64": base64.b64encode(malicious_bytes).decode("ascii")},
    }
    _patch_calibrator_primitive(state_dir, primitive)

    with pytest.raises(IncompatibleStateError, match="python_pickle"):
        WrapCalibratedExplainer.load_state(state_dir, learner=learner)

    assert not marker_path.exists()


def test_load_state_rejects_python_pickle_nested_inside_fast_collection(tmp_path: Path) -> None:
    """The same rejection must hold when 'python_pickle' is nested inside a
    fast_collection calibrator primitive."""
    state_dir, learner = _make_saved_wrapper(tmp_path, "nested_pickle_fast_state")
    marker_path = tmp_path / "nested-fast-sentinel.txt"
    malicious_bytes = pickle_dumps_for_test(_MaliciousReducer(str(marker_path)))
    child = {
        "schema_version": 3,
        "calibrator_type": "python_pickle",
        "checksums": {"sha256": _sha256(malicious_bytes)},
        "payload": {"pickle_b64": base64.b64encode(malicious_bytes).decode("ascii")},
    }
    children = [child]
    child_bytes = json.dumps(children, sort_keys=True).encode("utf-8")
    primitive = {
        "schema_version": 3,
        "calibrator_type": "fast_collection",
        "parameters": {"size": 1},
        "checksums": {"sha256": _sha256(child_bytes)},
        "calibrators": children,
    }
    _patch_calibrator_primitive(state_dir, primitive)

    with pytest.raises(IncompatibleStateError, match="python_pickle"):
        WrapCalibratedExplainer.load_state(state_dir, learner=learner)

    assert not marker_path.exists()


def test_load_state_rejects_unknown_calibrator_type(tmp_path: Path) -> None:
    """An unrecognized calibrator_type must fail closed rather than being resolved
    via dynamic import of an artifact-provided module path."""
    state_dir, learner = _make_saved_wrapper(tmp_path, "unknown_calibrator_state")
    primitive = {
        "schema_version": 3,
        "calibrator_type": "totally_unregistered_type",
        "parameters": {"module": "os", "class_name": "system"},
    }
    _patch_calibrator_primitive(state_dir, primitive)

    with pytest.raises(IncompatibleStateError, match="[Uu]nknown|[Uu]nsupported"):
        WrapCalibratedExplainer.load_state(state_dir, learner=learner)


def _patch_calibrator_primitive(state_dir: Path, primitive: dict[str, Any]) -> None:
    """Overwrite calibrator_primitive.json in a saved artifact and fix up its manifest
    checksum so the artifact remains internally self-consistent."""
    primitive_bytes = json.dumps(primitive, indent=2, sort_keys=True).encode("utf-8")
    (state_dir / "calibrator_primitive.json").write_bytes(primitive_bytes)
    manifest_path = state_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["calibrator_primitive.json"] = _sha256(primitive_bytes)
    _write_manifest(manifest_path, manifest)


# --------------------------------------------------------------------------
# Artifact parsing hardening
# --------------------------------------------------------------------------


def test_load_state_rejects_manifest_that_is_not_a_json_object(tmp_path: Path) -> None:
    state_dir = tmp_path / "malformed_manifest_state"
    state_dir.mkdir()
    (state_dir / "manifest.json").write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(IncompatibleStateError, match="JSON object"):
        WrapCalibratedExplainer.load_state(state_dir)


def test_load_state_rejects_missing_declared_file(tmp_path: Path) -> None:
    state_dir, learner = _make_saved_wrapper(tmp_path, "missing_file_state")
    (state_dir / "calibrator_primitive.json").unlink()

    with pytest.raises(IncompatibleStateError, match="missing"):
        WrapCalibratedExplainer.load_state(state_dir, learner=learner)


def test_load_state_rejects_absolute_path_in_manifest(tmp_path: Path) -> None:
    state_dir, learner = _make_saved_wrapper(tmp_path, "absolute_path_state")
    manifest_path = state_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    absolute_name = str((tmp_path / "outside.json").resolve())
    manifest["files"][absolute_name] = "0" * 64
    _write_manifest(manifest_path, manifest)

    with pytest.raises(IncompatibleStateError):
        WrapCalibratedExplainer.load_state(state_dir, learner=learner)


def test_load_state_rejects_path_traversal_in_manifest(tmp_path: Path) -> None:
    state_dir, learner = _make_saved_wrapper(tmp_path, "traversal_state")
    manifest_path = state_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["../escaped.json"] = "0" * 64
    _write_manifest(manifest_path, manifest)

    with pytest.raises(IncompatibleStateError):
        WrapCalibratedExplainer.load_state(state_dir, learner=learner)


def test_load_state_rejects_duplicate_file_after_normalization(tmp_path: Path) -> None:
    state_dir, learner = _make_saved_wrapper(tmp_path, "duplicate_state")
    manifest_path = state_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # "./explainer_state.json" normalizes to the same on-disk file as the
    # already-declared "explainer_state.json" entry.
    manifest["files"]["./explainer_state.json"] = manifest["files"]["explainer_state.json"]
    _write_manifest(manifest_path, manifest)

    with pytest.raises(IncompatibleStateError):
        WrapCalibratedExplainer.load_state(state_dir, learner=learner)


def test_load_state_rejects_unexpected_wrapper_pkl_in_safe_schema_artifact(
    tmp_path: Path,
) -> None:
    """A schema_version=3 artifact must reject a wrapper.pkl file even if it is
    listed in the manifest with a correctly recomputed checksum -- executable
    payload files are rejected by name, independent of checksum validity."""
    state_dir, learner = _make_saved_wrapper(tmp_path, "unexpected_pkl_state")
    payload = b"not-really-a-pickle-but-irrelevant"
    (state_dir / "wrapper.pkl").write_bytes(payload)

    manifest_path = state_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["wrapper.pkl"] = _sha256(payload)
    _write_manifest(manifest_path, manifest)

    with pytest.raises(IncompatibleStateError):
        WrapCalibratedExplainer.load_state(state_dir, learner=learner)


def test_load_state_rejects_unexpected_file_not_declared_in_manifest(tmp_path: Path) -> None:
    """Any on-disk file not declared in the manifest is rejected, even if it is
    never referenced by name -- not just the specific 'wrapper.pkl' case."""
    state_dir, learner = _make_saved_wrapper(tmp_path, "undeclared_file_state")
    (state_dir / "sneaky_extra.json").write_text("{}", encoding="utf-8")

    with pytest.raises(IncompatibleStateError, match="unexpected"):
        WrapCalibratedExplainer.load_state(state_dir, learner=learner)


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="Creating symlinks on Windows requires elevated privileges/dev mode",
)
def test_load_state_rejects_symlink_escaping_artifact_root(tmp_path: Path) -> None:
    state_dir, learner = _make_saved_wrapper(tmp_path, "symlink_state")
    outside_target = tmp_path / "outside_payload.json"
    outside_target.write_text(
        (state_dir / "explainer_state.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    link_path = state_dir / "escape_link.json"
    try:
        os.symlink(outside_target, link_path)
    except OSError:
        pytest.skip("Symlink creation not permitted in this environment")

    manifest_path = state_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["escape_link.json"] = _sha256(outside_target.read_bytes())
    _write_manifest(manifest_path, manifest)

    with pytest.raises(IncompatibleStateError):
        WrapCalibratedExplainer.load_state(state_dir, learner=learner)


# --------------------------------------------------------------------------
# Static safety guard: the safe persistence module must never call an
# arbitrary Python object deserializer.
# --------------------------------------------------------------------------

_FORBIDDEN_MODULES = {"pickle", "joblib", "cloudpickle"}
_FORBIDDEN_PAIRS = {
    ("pickle", "loads"),
    ("pickle", "load"),
    ("joblib", "load"),
    ("cloudpickle", "loads"),
}


def test_wrap_explainer_module_never_calls_unsafe_deserializers() -> None:
    """AST-based guard: calibrated_explanations.core.wrap_explainer must not call
    pickle.loads/pickle.load/joblib.load/cloudpickle.loads anywhere in the module
    (this is the module implementing the public safe-load call graph).

    This intentionally does not ban pickling repo-wide: WrapCalibratedExplainer
    remains picklable via __getstate__/__setstate__ for direct, user-controlled
    pickle.dump/joblib.dump use, which is a different, well-understood trust
    boundary than the ADR-031 state-artifact format guarded here.
    """
    source = inspect.getsource(wrap_explainer_module)
    tree = ast.parse(source)

    module_aliases: dict[str, str] = {}
    from_aliases: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _FORBIDDEN_MODULES:
                    module_aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module in _FORBIDDEN_MODULES:
            for alias in node.names:
                from_aliases[alias.asname or alias.name] = (node.module, alias.name)

    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            module_name = module_aliases.get(func.value.id)
            if module_name is not None and (module_name, func.attr) in _FORBIDDEN_PAIRS:
                violations.append(f"{module_name}.{func.attr}() at line {node.lineno}")
        elif isinstance(func, ast.Name) and func.id in from_aliases:
            module_name, original_name = from_aliases[func.id]
            if (module_name, original_name) in _FORBIDDEN_PAIRS:
                violations.append(
                    f"{module_name}.{original_name}() (imported as {func.id}) at line {node.lineno}"
                )

    assert violations == [], (
        "wrap_explainer.py implements the public safe-load call graph and must never "
        f"call an arbitrary Python object deserializer; found: {violations}"
    )
