import importlib.resources as resources
import json
import pytest

PLOTSPEC_SCHEMA_PACKAGE = "calibrated_explanations.schemas.v1"
PRIMITIVES_SCHEMA_PACKAGE = "development.schemas"


def load_schema(package: str, name: str):
    with resources.files(package).joinpath(name).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_plotspec_and_primitives_schemas_exist():
    plotspec = resources.files(PLOTSPEC_SCHEMA_PACKAGE).joinpath("plotspec_schema.json")
    primitives = resources.files(PRIMITIVES_SCHEMA_PACKAGE).joinpath("primitives_schema.json")
    assert plotspec.is_file(), "Packaged plotspec_schema.json must exist"
    assert primitives.is_file(), "primitives_schema.json must exist"


def test_schema_declares_interval_and_save_requirements():
    plotspec = load_schema(PLOTSPEC_SCHEMA_PACKAGE, "plotspec_schema.json")
    plot_spec = plotspec["properties"]["plot_spec"]
    required = set(plot_spec.get("required", ()))
    assert {"kind", "mode"}.issubset(required)
    feature_entries = (
        plot_spec.get("properties", {})
        .get("feature_entries", {})
        .get("items", {})
        .get("properties", {})
    )
    assert "low" in feature_entries and "high" in feature_entries
    save_behavior = plot_spec.get("properties", {}).get("save_behavior", {}).get("properties", {})
    assert "default_exts" in save_behavior
    assert "style" in plot_spec.get("properties", {})
    assert "uncertainty" in plot_spec.get("properties", {})


def test_plotspec_schema_declares_feature_entry_requirements():
    schema = load_schema(PLOTSPEC_SCHEMA_PACKAGE, "plotspec_schema.json")
    props = schema["properties"]["plot_spec"].get("properties", {})
    assert "feature_entries" in props
    feature_entries = props["feature_entries"]
    items = feature_entries.get("items", {})
    required = set(items.get("required", []))
    assert {"name", "weight"}.issubset(required)
    assert "index" in items.get("properties", {})


def test_plotspec_schema_defines_feature_contract():
    schema = load_schema(PLOTSPEC_SCHEMA_PACKAGE, "plotspec_schema.json")
    plot_spec = schema["properties"]["plot_spec"]

    assert "feature_order" in plot_spec["properties"]
    feature_order = plot_spec["properties"]["feature_order"]
    assert "array" in feature_order["type"]
    assert feature_order["items"]["type"] == "integer"

    feature_entries = plot_spec["properties"]["feature_entries"]
    entry_schema = feature_entries["items"]
    assert set(entry_schema["required"]) == {"name", "weight"}
    low_type = entry_schema["properties"]["low"]["type"]
    high_type = entry_schema["properties"]["high"]["type"]
    assert "number" in low_type
    assert "number" in high_type


@pytest.mark.skipif(
    pytest.importorskip("jsonschema", exc_type=ImportError) is None,
    reason="jsonschema not installed in this environment; run locally with jsonschema to validate schemas",
)
def test_runtime_plotspec_payloads_validate_against_packaged_schema():
    """Runtime PlotSpec envelopes should validate against the packaged schema."""
    import jsonschema
    from calibrated_explanations.viz.builders import (
        build_global_plotspec,
        build_triangular_plotspec,
    )
    from calibrated_explanations.viz.plotspec import (
        BarHPanelSpec,
        BarItem,
        IntervalHeaderSpec,
        PlotSpec,
    )
    from calibrated_explanations.viz.serializers import (
        global_plotspec_to_dict,
        plotspec_to_dict,
        triangular_plotspec_to_dict,
    )

    plotspec_schema = load_schema(PLOTSPEC_SCHEMA_PACKAGE, "plotspec_schema.json")
    primitives_schema = load_schema(PRIMITIVES_SCHEMA_PACKAGE, "primitives_schema.json")

    factual = plotspec_to_dict(
        PlotSpec(
            title="Example",
            header=IntervalHeaderSpec(pred=0.65, low=0.2, high=0.8),
            body=BarHPanelSpec(
                bars=[BarItem(label="f0", value=0.35, interval_low=0.2, interval_high=0.5)],
                xlabel="Feature weights",
                ylabel="Features",
            ),
            kind="factual_probabilistic",
            mode="classification",
            feature_order=(0,),
        )
    )
    triangular = triangular_plotspec_to_dict(
        build_triangular_plotspec(
            title="tri",
            proba=[0.2, 0.8],
            uncertainty=[0.1, 0.2],
            rule_proba=[0.6],
            rule_uncertainty=[0.3],
            num_to_show=1,
            is_probabilistic=True,
        )
    )
    global_payload = global_plotspec_to_dict(
        build_global_plotspec(
            title="glob",
            proba=[0.1, 0.7],
            predict=None,
            low=[0.05, 0.6],
            high=[0.15, 0.8],
            uncertainty=[0.1, 0.2],
            y_test=[0, 1],
            is_regularized=True,
        )
    )

    for payload in (factual, triangular, global_payload):
        jsonschema.validate(instance=payload, schema=plotspec_schema)
    assert factual["plot_spec"]["feature_entries"][0]["name"] == "f0"
    assert triangular["plot_spec"]["triangular"]["num_to_show"] == 1
    assert global_payload["plot_spec"]["axis_hints"]["xlim"] == [0.1, 0.7]

    export_wrapper = {
        "plot_spec": factual["plot_spec"],
        "primitives": [
            {
                "id": "p1",
                "axis_id": "header.pos",
                "type": "fill_between",
                "coords": {"x": [0, 1], "y1": [0.2, 0.2], "y2": [0.5, 0.5]},
                "style": {"color": "#ff0000", "alpha": 0.2},
                "semantic": "probability_fill",
            }
        ],
    }
    jsonschema.validate(instance=export_wrapper, schema=primitives_schema)
