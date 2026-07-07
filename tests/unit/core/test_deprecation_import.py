def test_should_resolve_core_lazy_exports_when_accessed() -> None:
    """Exercise calibrated_explanations.core.__getattr__ branches for coverage."""
    import importlib
    import types

    core = importlib.import_module("calibrated_explanations.core")

    # Arrange/Act: Access lazy exports.
    _ = core.CalibratedExplainer
    _ = core.WrapCalibratedExplainer
    _ = core.assign_threshold
    _ = core.CalibratedError
    _ = core.ValidationError
    _ = core.explain_exception
    explain_mod = core.explain

    # Assert: explain resolves to a module.
    assert isinstance(explain_mod, types.ModuleType)
