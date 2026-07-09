"""Parameter canonicalization utilities (ADR-002).

This module centralizes lightweight argument normalization and consistency
checks, enabling downstream validators and plugins to rely on a stable
parameter contract.

Notes
-----
- Alias mapping enables backward compatibility without behavior drift.
- Combination validation enforces ADR-compliant constraints (e.g., conflicting
  parameter exclusivity).
- Canonicalization only maps known aliases to canonical keys when the
  canonical key is not already provided.

See ADR-002 for context.
"""

from __future__ import annotations

from typing import Any

from ..utils.exceptions import ConfigurationError

# Removed aliases (v0.11.0).
REMOVED_ALIAS_MAP: dict[str, str] = {
    "alpha": "low_high_percentiles",
    "alphas": "low_high_percentiles",
    "n_jobs": "parallel_workers",
}

REMOVED_GUARDED_KWARG_MAP: dict[str, str] = {
    "guarded": "guarded_options=GuardedOptions()",
    "significance": "guarded_options=GuardedOptions(confidence=1-significance)",
    "n_neighbors": "guarded_options=GuardedOptions(n_neighbors=...)",
    "normalize_guard": "guarded_options=GuardedOptions(normalize=...)",
    "merge_adjacent": "guarded_options=GuardedOptions(merge_adjacent=...)",
}

REMOVED_REJECT_KWARG_MAP: dict[str, str] = {
    "confidence": "reject_confidence",
}

REMOVED_NORMALIZATION_KWARG_MAP: dict[str, str] = {
    "normalize": "normalization=NormalizationStrategy.<MEMBER>",
}

UNSUPPORTED_NARRATIVE_KWARG_MAP: dict[str, str] = {
    "format": 'output_format=<"text"|"dataframe"|"html"|"dict"|"markdown"> and expertise_level=<"beginner"|"intermediate"|"advanced">',
}

# Kept for API compatibility; no active alias mapping remains after v0.11.0.
ALIAS_MAP: dict[str, str] = {}

# Parameter combinations that are mutually exclusive or conflicting.
EXCLUSIVE_PARAM_GROUPS: list[tuple[str, ...]] = [
    # Example: threshold and confidence_level are alternative ways to specify coverage.
    # Callers should choose one or the other, not both.
    ("threshold", "confidence_level"),
]


def canonicalize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of kwargs with known aliases mapped to canonical keys.

    Notes
    -----
    - If an alias exists in ``kwargs`` and the canonical key is absent, copy the
      value to the canonical key.
    - If both alias and canonical are present, keep the canonical value and do not
      overwrite it.
    - Always preserve original keys; we do not delete aliases to avoid
      any chance of behavior drift. Callers should read canonical keys first.
    - Unknown keys are left untouched.
    """
    return dict(kwargs)


def reject_removed_aliases(kwargs: dict[str, Any]) -> None:
    """Reject aliases removed in v0.11.0 with actionable migration guidance."""
    used = {alias: canonical for alias, canonical in REMOVED_ALIAS_MAP.items() if alias in kwargs}
    if not used:
        return
    formatted = ", ".join(f"'{alias}' -> '{canonical}'" for alias, canonical in used.items())
    raise ConfigurationError(
        "Deprecated parameter aliases were removed in v0.11.0. "
        f"Use canonical names instead: {formatted}.",
        details={
            "removed_aliases": list(used.keys()),
            "canonical_replacements": used,
            "removed_in": "v0.11.0",
        },
    )


def reject_removed_guarded_kwargs(kwargs: dict[str, Any]) -> None:
    """Reject guarded kwargs removed in v0.11.5 with actionable migration guidance."""
    used = {
        name: replacement
        for name, replacement in REMOVED_GUARDED_KWARG_MAP.items()
        if name in kwargs
    }
    if not used:
        return
    formatted = ", ".join(f"'{name}' -> '{replacement}'" for name, replacement in used.items())
    raise ConfigurationError(
        "Guarded keyword arguments were removed in v0.11.5. "
        f"Use GuardedOptions instead: {formatted}.",
        details={
            "removed_kwargs": list(used.keys()),
            "replacements": used,
            "removed_in": "v0.11.5",
        },
    )


def reject_removed_reject_kwargs(kwargs: dict[str, Any]) -> None:
    """Reject reject-policy kwargs removed in v0.11.5 with migration guidance."""
    used = {
        name: replacement
        for name, replacement in REMOVED_REJECT_KWARG_MAP.items()
        if name in kwargs
    }
    if not used:
        return
    formatted = ", ".join(f"'{name}' -> '{replacement}'" for name, replacement in used.items())
    raise ConfigurationError(
        "Reject-policy keyword arguments were removed in v0.11.5. "
        f"Use canonical names instead: {formatted}.",
        details={
            "removed_kwargs": list(used.keys()),
            "replacements": used,
            "removed_in": "v0.11.5",
        },
    )


def reject_removed_normalization_kwarg(kwargs: dict[str, Any]) -> None:
    """Reject the legacy ``normalize=`` bool passthrough removed in v0.11.5.

    ``normalize=`` looked like a synonym of the live ``normalization=`` parameter
    but was actually a removed alias: any presence raises, regardless of value.
    Previously this was only caught deep inside ``VennAbers.predict_proba`` via
    ``ValidationError``, well past the public wrapper boundary; catching it here
    gives a `ConfigurationError` consistent with every other removed alias (ADR-038
    5B).
    """
    used = {
        name: replacement
        for name, replacement in REMOVED_NORMALIZATION_KWARG_MAP.items()
        if name in kwargs
    }
    if not used:
        return
    formatted = ", ".join(f"'{name}' -> '{replacement}'" for name, replacement in used.items())
    raise ConfigurationError(
        "The legacy `normalize=` bool passthrough was removed in v0.11.5. "
        f"Use canonical names instead: {formatted}.",
        details={
            "removed_kwargs": list(used.keys()),
            "replacements": used,
            "removed_in": "v0.11.5",
        },
    )


def reject_unknown_public_kwargs(
    kwargs: dict[str, Any], *, allowed: frozenset[str] | set[str], surface: str
) -> None:
    """Reject unrecognized public keyword arguments (ADR-038 D3: fail-fast).

    Replaces the pre-v0.11.6 behavior of warning and silently forwarding
    unknown public kwargs; see ``development/current-work/v0.11.6_plan.md``
    (D3 resolution) for the CHANGELOG-documented reversal from v0.11.4 Task 15.
    """
    unknown = sorted(set(kwargs) - allowed)
    if not unknown:
        return
    raise ConfigurationError(
        f"{surface} received unknown keyword arguments: {unknown}.",
        details={"surface": surface, "unknown_kwargs": unknown},
    )


def reject_unsupported_narrative_kwargs(kwargs: dict[str, Any], *, surface: str) -> None:
    """Reject unsupported ``to_narrative`` kwargs that were previously ignored.

    The CE-first docs mistakenly taught ``format=...`` while the runtime only
    supported ``output_format=...`` plus ``expertise_level=...``. Prior to
    v0.11.6 Task 12, ``format`` was silently forwarded and ignored, yielding the
    default dataframe output instead of the requested narrative style.
    """
    used = {
        name: replacement
        for name, replacement in UNSUPPORTED_NARRATIVE_KWARG_MAP.items()
        if name in kwargs
    }
    if not used:
        return
    formatted = ", ".join(f"'{name}' -> '{replacement}'" for name, replacement in used.items())
    raise ConfigurationError(
        "Unsupported narrative keyword arguments were provided. "
        f"Use the public narrative API instead: {formatted}.",
        details={
            "surface": surface,
            "unsupported_kwargs": list(used.keys()),
            "replacements": used,
        },
    )


def reject_cross_surface_kwargs(
    kwargs: dict[str, Any],
    *,
    allowed: frozenset[str] | set[str],
    closed_surface_names: frozenset[str] | set[str],
    surface: str,
) -> None:
    """Reject kwargs known on another closed surface but not valid here (ADR-038 5C).

    For experimental plugin-forwarding surfaces (ADR-038 §3 exception), a name
    genuinely unknown anywhere is treated as plugin-defined and passed through
    silently. Only names that are recognized on a *closed* surface (e.g.
    ``CalibratedExplainer.__init__``/``predict``/``predict_proba``) but not valid
    on this experimental surface are rejected -- this preserves the plugin
    extensibility the §3 exception grants while still closing the cross-method
    silent-no-op class of bug (ADR-038 5B/5C).
    """
    out_of_scope = sorted((set(kwargs) & closed_surface_names) - allowed)
    if not out_of_scope:
        return
    raise ConfigurationError(
        f"{surface} received keyword arguments that are recognized on another"
        f" method but not valid here: {out_of_scope}.",
        details={
            "surface": surface,
            "out_of_scope_kwargs": out_of_scope,
            "allowed_kwargs": sorted(allowed),
        },
    )


def validate_param_combination(kwargs: dict[str, Any]) -> None:
    """Perform basic consistency checks for parameter combinations (ADR-002).

    Enforces mutual exclusivity and conflict constraints. Raises
    ``ConfigurationError`` when violations are detected.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments to validate.

    Raises
    ------
    ConfigurationError
        When conflicting parameter combinations are detected.
    """
    # Check mutual exclusivity groups
    for param_group in EXCLUSIVE_PARAM_GROUPS:
        present = [p for p in param_group if p in kwargs and kwargs[p] is not None]
        if len(present) > 1:
            raise ConfigurationError(
                f"Parameters {present} are mutually exclusive; specify at most one.",
                details={
                    "conflict": param_group,
                    "provided": present,
                    "requirement": "choose one or none",
                },
            )


__all__ = [
    "ALIAS_MAP",
    "REMOVED_ALIAS_MAP",
    "REMOVED_GUARDED_KWARG_MAP",
    "REMOVED_REJECT_KWARG_MAP",
    "REMOVED_NORMALIZATION_KWARG_MAP",
    "UNSUPPORTED_NARRATIVE_KWARG_MAP",
    "canonicalize_kwargs",
    "reject_removed_guarded_kwargs",
    "reject_removed_reject_kwargs",
    "reject_removed_aliases",
    "reject_removed_normalization_kwarg",
    "reject_unsupported_narrative_kwargs",
    "reject_unknown_public_kwargs",
    "reject_cross_surface_kwargs",
    "validate_param_combination",
]


def warn_on_aliases(kwargs: dict[str, Any]) -> None:
    """Compatibility wrapper for the removed alias guard.

    Notes
    -----
    - `warn_on_aliases` historically emitted deprecation warnings.
    - Since aliases were removed in v0.11.0, this now fails fast.
    """
    reject_removed_aliases(kwargs)


__all__.append("warn_on_aliases")
