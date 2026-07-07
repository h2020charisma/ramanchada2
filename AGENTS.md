# AGENTS.md

## Sources

- Prefer `setup.py`, `tox.ini`, `pytest.ini`, `mypy.ini`, `environment.yml`, and `.github/workflows/*.yml` over prose when commands or supported versions conflict.
- This repo has no `pyproject.toml`; it is a setuptools `src/` layout package.

## Project Shape

- Package code lives under `src/ramanchada2`; tests live under `tests`.
- Main user-facing type is `ramanchada2.spectrum.Spectrum`.
- Spectrum constructors, filters, and methods are dynamically attached by decorators in `src/ramanchada2/misc/spectrum_deco/`; importing `ramanchada2.spectrum` registers subpackage functions onto `Spectrum`.
- Add Spectrum transforms with the existing decorators: `add_spectrum_constructor`, `add_spectrum_filter`, or `add_spectrum_method`; filters should preserve the old/new Spectrum pattern, processing history, and cache behavior.
- Calibration and twinning workflows live under `src/ramanchada2/protocols/`; packaged reference spectra live under `src/ramanchada2/auxiliary/spectra/datasets2`.

## Commands

- Editable install: `pip install -e .`.
- Conda setup: `conda env create && conda activate ramanchada2`.
- Full CI-style run: `tox`.
- Quick test subset: `tox -e quick`.
- Focused test: `pytest tests/path/to/test_file.py::test_name`.
- Typecheck: `tox -e mypy`.
- Lint: `tox -e flake8`.
- Build docs: `tox -e docs`.
- `tox -e black` and `tox -e usort` format only the explicit file lists in `tox.ini`; they are not repo-wide check-only commands.

## Test Notes

- Run tests from the repo root; some tests use relative paths such as `./tests/data/...`.
- `tox -e quick` skips the slow/noisy Pearson4 tests, twinning test, and calibration-model protocol tests listed in `tox.ini`.
- Some tests intentionally write ignored root artifacts such as `pearson4*.png`, `pearson4*.csv`, `test_twinning_*.png`, `reference.csv`, and `twinned.csv`; do not commit these.
- CI runs Python `3.9` through `3.13`; mypy, black, usort, flake8, and docs run on Python `3.13`.

## Packaging And Docs

- Package version is read from `src/ramanchada2/__init__.py`; PyPI long description is `README.pypi`.
- `setup.py` packages auxiliary `*.txt` and `*.dat` data; `MANIFEST.in` separately includes `src/ramanchada2/protocols/calibration/config_certs.json`.
- If adding runtime data files with other extensions, update packaging rules and verify an installed package can load them.
- `tox -e docs` runs `scripts/gen-decorated-docs.py`, which regenerates `spectrum_functions.md`, then writes pdoc output to `docs/_build`.

## Style

- Flake8 max line length is `119`.
- Mypy uses the Pydantic plugin, Python `3.13`, and `ignore_missing_imports = True`.
- Preserve existing small, explicit module imports; many `__init__.py` imports are needed to register decorated Spectrum APIs.
