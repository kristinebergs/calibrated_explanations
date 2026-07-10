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


def test_should_resolve_core_lazy_submodule_attributes_when_accessed() -> None:
    """Exercise the core package compatibility submodule lazy-import branch."""
    import importlib
    import types

    core = importlib.import_module("calibrated_explanations.core")
    core.__dict__.pop("discretizer_config", None)

    discretizer_config = core.discretizer_config

    assert isinstance(discretizer_config, types.ModuleType)
    assert discretizer_config is core.__dict__["discretizer_config"]


def test_should_raise_attribute_error_for_unknown_core_lazy_export() -> None:
    """Exercise the unknown-attribute branch of core.__getattr__."""
    import importlib

    core = importlib.import_module("calibrated_explanations.core")

    try:
        _ = core.not_a_real_core_export
    except AttributeError as exc:
        assert str(exc) == "not_a_real_core_export"
    else:
        raise AssertionError("Expected AttributeError for unknown core lazy export")
