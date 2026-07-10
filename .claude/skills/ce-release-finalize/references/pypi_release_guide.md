# PyPI Release Guide for calibrated-explanations

## Pre-requisites (once)

- Read the official packaging tutorial: https://packaging.python.org/en/latest/tutorials/packaging-projects/
- Install tooling: `python -m pip install --upgrade build twine`
- Create a PyPI account and API token
- Configure `.pypirc`:
	- macOS/Linux: `~/.pypirc`
	- Windows: `%USERPROFILE%\\.pypirc`

```ini
[pypi]
	username = __token__
	password = pypi-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Notes:

- PyPI uploads are immutable per version.
- This project uses PEP 440 version (`X.Y.Z`) and display/tag version with leading `v` (`vX.Y.Z`).

## Files that must be updated for a release

When releasing `vX.Y.Z`:

1. Packaging/runtime version

- `pyproject.toml`: set `[project].version = "X.Y.Z"`
- `src/calibrated_explanations/__init__.py`: set `__version__ = "vX.Y.Z"`

2. Citation/docs version

- `CITATION.cff`: set `version: vX.Y.Z` and `date-released: 'YYYY-MM-DD'`
- `docs/conf.py`:
	- `release = "X.Y.Z"`
	- `version = "X.Y"`
- `docs/citing.md`: update BibTeX `version = {vX.Y.Z}` and month/year if needed

3. Changelog

- `CHANGELOG.md`: move `[Unreleased]` items to a new version section and update compare links

4. Repo metadata

- `METADATA.json`: set `version` to `"X.Y.Z"`

5. Planning/release tracking

- `development/current-work/RELEASE_PLAN_v1.md`: update running version and release notes
- `development/current-work/vX.Y.Z_plan.md`: create next release plan

Not required (usually):

- `*.egg-info/*`, `dist/*`: generated artifacts, do not edit manually
- Notebook version strings: update only when producing release evidence outputs (required gate below)

## Release steps (every release)

1. Checkout and pull `main`.

```bash
git checkout main
git pull
```

2. Confirm CI is green on `main`.

Suggested local check:

```bash
make ci-local-new
```

3. Update `CHANGELOG.md`.

- Add `## [vX.Y.Z](https://github.com/Moffran/calibrated_explanations/releases/tag/vX.Y.Z) - YYYY-MM-DD`
- Add full changelog compare link from previous tag to `vX.Y.Z`
- Move relevant `[Unreleased]` bullets into new section
- Update `[Unreleased]` compare link to `compare/vX.Y.Z...main`

4. Bump all release version files listed above.

5. Install release version locally and execute notebooks in-place (required pre-upload gate).

```bash
python -m pip install -e .[dev]
python -c "import calibrated_explanations as ce; print(ce.__version__)"
```

Execute and save notebooks in-place:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=600 \
		notebooks/quickstart.ipynb \
		notebooks/quickstart_guarded.ipynb \
		notebooks/quickstart_tiny.ipynb \
		notebooks/core_demos/*.ipynb \
		notebooks/miscellaneous/*.ipynb \
		notebooks/paper_based/*.ipynb \
		notebooks/advanced/demo_conditional.ipynb \
		notebooks/advanced/demo_config_management.ipynb \
		notebooks/advanced/demo_narrative_explanations.ipynb \
		notebooks/advanced/demo_plugin_wiring.ipynb \
		notebooks/advanced/demo_reject.ipynb \
		notebooks/advanced/demo_under_the_hood.ipynb
```

Run the heavy notebook separately:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=5400 \
		notebooks/advanced/fast_feature_filtering_demo.ipynb
```

Spot-check outputs for `vX.Y.Z`. `python scripts/run_all_notebooks.py` is pass/fail only unless `--inplace` behavior is explicitly added.

6. Sanity-check version consistency across all release files.

7. Remove build artifacts.

```bash
# Bash
rm -rf dist/ build/ *.egg-info/

# PowerShell
Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force *.egg-info -ErrorAction SilentlyContinue
```

8. Build distributions.

```bash
python -m build
```

9. Validate distributions.

```bash
python -m twine check dist/*
```

10. Smoke-test the wheel locally (recommended).

```bash
python -m venv venv-wheel
# Windows (PowerShell): .\\venv-wheel\\Scripts\\Activate.ps1
# macOS/Linux: source venv-wheel/bin/activate
python -m pip install --upgrade pip
pip install calibrated_explanations --find-links dist/
python -c "import calibrated_explanations as ce; print(ce.__version__)"
```

Strict local pre-tag automation gate:

```bash
make release-preflight
```

Immediately before tagging:

```bash
make release-finalize
```

Do not continue unless both commands exit 0.

11. Commit and tag semantic version.

```bash
git add .
git commit -m 'calibrated-explanations vX.Y.Z'
git tag vX.Y.Z
git push
git push --tags
```

12. Publish docs on Read the Docs.

- Confirm tag exists remotely: `git ls-remote --tags origin vX.Y.Z`
- Verify RTD tag build passes
- Activate `vX.Y.Z`
- Set stable to `vX.Y.Z`
- Spot-check `https://calibrated-explanations.readthedocs.io/en/stable/`

13. Upload artifacts.

```bash
python -m twine upload --repository testpypi dist/*
python -m twine upload --repository pypi dist/*
```

14. Verify PyPI project page:

- https://pypi.org/project/calibrated-explanations/

15. Test install in a clean environment.

```bash
python -m venv venv-release
# Windows (PowerShell): .\\venv-release\\Scripts\\Activate.ps1
# macOS/Linux: source venv-release/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade calibrated-explanations==X.Y.Z
python -c "import calibrated_explanations as ce; print(ce.__version__)"
```

16. Prepare next release plan.

- Update `development/current-work/RELEASE_PLAN_v1.md` with release notes
- Create/update `development/current-work/vX.Y.Z_plan.md` for next version

17. Bump to next `-dev` version in `pyproject.toml` and `src/calibrated_explanations/__init__.py`.

## Critical notes

- Never upload before `twine check` passes.
- Never tag without a green CI/build state.
- `make release-preflight` and `make release-finalize` are mandatory release gates.
- RTD uses `.readthedocs.yaml` and `docs/requirements-doc.txt`; notebook execution is intentionally disabled on RTD.
