"""Plugin base interfaces (ADR-006 skeleton).

Minimal interfaces to support a registry of third-party explainers. This is an
opt-in surface; users should understand that loading external plugins executes
arbitrary code. We will document risks and keep the registry explicit.

Contract (v0.2, unstable):
- Each plugin module exposes a ``plugin_meta`` dict with at least:
    {
        "schema_version": 1,
        "capabilities": ["explain"],
        "name": str,
        "version": str,
        "provider": str,
        "trust": bool | Mapping[str, Any],
    }
- Each plugin exposes two callables:
    supports(model) -> bool
    explain(model, X, **kwargs) -> Any  # typically an Explanation or legacy dict

This mirrors ADR-006 minimal capability metadata and keeps behavior opt-in.

ADR-015 refines this layer with dedicated explanation, interval, and plotting
protocols. They build on the lightweight ``PluginMeta`` typing alias and the
validation helper exported from this module.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Protocol, TypeAlias


PluginMeta: TypeAlias = Mapping[str, Any]


class ExplainerPlugin(Protocol):
    """Protocol for explainer plugins.

    Implementations are expected to provide:
    - plugin_meta: Dict[str, Any]
    - supports(model) -> bool
    - explain(model, X, **kwargs) -> Any
    """

    plugin_meta: PluginMeta

    def supports(self, model: Any) -> bool:  # pragma: no cover - protocol
        ...

    def explain(self, model: Any, X: Any, **kwargs: Any) -> Any:  # pragma: no cover - protocol
        ...


def validate_plugin_meta(meta: Dict[str, Any]) -> None:
    """Validate minimal plugin metadata.

    Required keys: schema_version (int), capabilities (list[str]), name (str),
    version (str), provider (str), trust (bool or mapping)
    """

    if not isinstance(meta, dict):
        raise ValueError("plugin_meta must be a dict")
    required = (
        ("schema_version", int),
        ("capabilities", list),
        ("name", str),
        ("version", str),
        ("provider", str),
    )
    for key, typ in required:
        if key not in meta:
            raise ValueError(f"plugin_meta missing required key: {key}")
        if not isinstance(meta[key], typ):
            raise ValueError(f"plugin_meta[{key!r}] must be {typ.__name__}")
    if "trust" not in meta:
        raise ValueError("plugin_meta missing required key: trust")
    trust_value = meta["trust"]
    if not isinstance(trust_value, (bool, Mapping)):
        raise ValueError("plugin_meta['trust'] must be a bool or mapping")
    checksum = meta.get("checksum")
    if checksum is not None and not isinstance(checksum, str):
        raise ValueError("plugin_meta['checksum'] must be a string when provided")


__all__ = ["ExplainerPlugin", "validate_plugin_meta"]
