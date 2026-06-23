"""Shared helpers for TIF evidence payload construction.

Utility functions used by TIF executables to build evidence payloads.
Not a registry. Not a manifest. Pure utility code.

TIF executable modules import these helpers inside build_evidence_payload()
so the import is deferred until evidence generation time.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any


def obs_to_dict(value: Any) -> dict[str, Any]:
    """Convert an observation dataclass or dict-like to a plain dict."""
    return asdict(value) if hasattr(value, "__dataclass_fields__") else dict(value)


def acceptance_entry(
    criterion_ref: str,
    field: str,
    expected: Any,
    observed: Any,
) -> dict[str, Any]:
    """Build a single acceptance check entry."""
    return {
        "criterion_ref": criterion_ref,
        "field": field,
        "expected": expected,
        "observed": observed,
        "result": "pass" if observed == expected else "fail",
    }


def scenario_entry(
    scenario_id: str,
    observations: dict[str, Any],
    acceptance: list[dict[str, Any]],
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a single scenario entry."""
    payload: dict[str, Any] = {
        "scenario_id": scenario_id,
        "observations": observations,
        "result": "pass" if all(a["result"] == "pass" for a in acceptance) else "fail",
        "acceptance": acceptance,
    }
    if parameters:
        payload["parameters"] = parameters
    return payload


def overall_result(scenarios: list[dict[str, Any]]) -> str:
    """Derive top-level result from scenario results."""
    return "pass" if all(s["result"] == "pass" for s in scenarios) else "fail"


def build_payload(
    evidence_key: str,
    *,
    claim_ids: list[str],
    requirement_ids: list[str],
    adr_refs: list[str],
    tif_ids: list[str],
    verification_type: str,
    dataset_id: str,
    scenarios: list[dict[str, Any]],
    commit_sha: str,
    timestamp: str,
    date_suffix: str,
    package_version: str,
    python_version: str,
    platform_str: str,
    configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete, serialisable evidence payload dict."""
    evidence_id = f"CE-EVID-{evidence_key}-{date_suffix}"
    return {
        "evidence_id": evidence_id,
        "claim_ids": claim_ids,
        "requirement_ids": requirement_ids,
        "adr_refs": adr_refs,
        "standard_refs": [],
        "tif_ids": tif_ids,
        "verification_type": verification_type,
        "result": overall_result(scenarios),
        "timestamp": timestamp,
        "commit_sha": commit_sha,
        "package_version": package_version,
        "python_version": python_version,
        "platform": platform_str,
        "dataset_id": dataset_id,
        "random_seed": 42,
        "configuration": configuration or {},
        "scenarios": scenarios,
        "artifacts": {"logs": None, "raw_output": None},
    }
