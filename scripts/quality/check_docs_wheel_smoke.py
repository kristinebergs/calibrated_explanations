"""Run a wheel-only smoke check for public docs snippets.

This task-15 guard proves the affected docs examples work from a built wheel
without repository-only ``tests`` helpers on ``sys.path``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

SMOKE_CODE = r"""
import importlib
import importlib.util
import sys

site_dir = sys.argv[1]
sys.path.insert(0, site_dir)

assert importlib.util.find_spec("tests") is None
for module_name in ("tests.helpers.doc_utils", "tests.helpers.model_utils"):
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError(f"{module_name} unexpectedly importable in wheel-only smoke")

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from calibrated_explanations import CalibratedExplainer, WrapCalibratedExplainer
from calibrated_explanations.api.config import ExplainerBuilder
from calibrated_explanations.plotting import resolve_plot_style_chain

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
x_temp, x_test, y_temp, _ = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_cal, y_train, y_cal = train_test_split(
    x_temp, y_temp, test_size=0.4, random_state=42
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_cal = scaler.transform(x_cal)
x_test = scaler.transform(x_test)

model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

explainer = CalibratedExplainer(model, x_cal, y_cal, plot_style="plot_spec.default")
chain = resolve_plot_style_chain(explainer, explicit_style="plot_spec.default")
assert chain[0] == "plot_spec.default"

assert hasattr(explainer.explain_factual(x_test[:3]), "plot")

builder = ExplainerBuilder(model)
config = builder.perf_cache(True, max_items=64, namespace="docs-smoke").build_config()
wrapped = WrapCalibratedExplainer.from_config(config)
assert wrapped is not None

print("docs-wheel-smoke: ok")
"""


def _build_wheel(dist_dir: Path) -> Path:
    subprocess.run(  # noqa: S603
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)],
        check=True,
        cwd=REPO_ROOT,
    )
    wheels = sorted(dist_dir.glob("calibrated_explanations-*.whl"))
    if not wheels:
        raise FileNotFoundError(f"No wheel built under {dist_dir}")
    return wheels[-1]


def _install_wheel(wheel_path: Path, site_dir: Path) -> None:
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(site_dir),
            str(wheel_path),
        ],
        check=True,
        cwd=REPO_ROOT,
    )


def run_docs_wheel_smoke() -> int:
    if shutil.which("pip") is None:
        print("ERROR: pip is unavailable in the current environment.")
        return 1

    with tempfile.TemporaryDirectory(prefix="ce-docs-wheel-smoke-") as tmpdir:
        root = Path(tmpdir)
        dist_dir = root / "dist"
        site_dir = root / "site"
        run_dir = root / "run"
        dist_dir.mkdir()
        site_dir.mkdir()
        run_dir.mkdir()

        wheel_path = _build_wheel(dist_dir)
        _install_wheel(wheel_path, site_dir)

        result = subprocess.run(  # noqa: S603
            [sys.executable, "-I", "-c", SMOKE_CODE, str(site_dir)],
            check=False,
            cwd=run_dir,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
        return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        return run_docs_wheel_smoke()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
