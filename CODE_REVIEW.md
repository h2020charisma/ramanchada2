# Code Review — ramanchada2

- **Date:** 2026-07-06
- **Reviewed at:** branch `issue233`, commit `a88c8deb` ("Add rbfinverse (monotone polyharmonic spline) + robust anchor filter")
- **Scope:** full-repo review with emphasis on the `protocols/calibration` package (most actively developed), the `Spectrum` core, the dynamic-decorator machinery, I/O entry points, packaging/CI, and tests. ~137 source files, ~11.5k lines.

---

## Overall assessment

ramanchada2 is a well-scoped scientific package with a clear purpose (harmonising Raman spectra) and several genuinely good engineering decisions: an immutable-by-convention `Spectrum` with recorded processing provenance, a plugin-style registration of filters/creators, wide file-format support, a real CI matrix (Python 3.9–3.13 with flake8, mypy, black/usort, coverage, docs), and tests that exercise real experimental data end-to-end.

The main risks are concentrated in the calibration subsystem, which is evolving fast and shows research-code habits leaking into library code: duplicated function definitions, unconditional `print` diagnostics, mutable default arguments that are actually mutated, commented-out code, and inconsistent `Literal` option sets between API layers. There is also one class of genuine correctness bug caused by the `Spectrum.x` property returning a copy (in-place mutation silently does nothing), and two security-relevant patterns (`pickle.load` for models, `eval` for certificate equations) that deserve an explicit trust-boundary statement.

Priority summary:

| Priority | Theme |
|---|---|
| P1 | In-place mutation of `Spectrum.x` copies is a silent no-op (extrapolation masking) |
| P1 | Mutable default dicts are mutated across calls (`derive_model_x`) |
| P1 | Duplicate function definitions in `qmatch.py` |
| P2 | `eval`/`pickle` trust boundary; `model_construct` bypasses validation |
| P2 | API-surface inconsistencies in `match_method`/`interpolator_method` literals and defaults |
| P3 | Library hygiene: prints, dead code, typos, missing f-strings, legacy `setup.py` |

---

## 1. Correctness findings

### 1.1 Assigning into `Spectrum.x` element-wise is a silent no-op — P1

`Spectrum.x` (and `.y`) getters return a **fresh copy** on every access (`return np.array(self._xdata)`, `src/ramanchada2/spectrum/spectrum.py:134`). Any element-wise assignment through the property mutates the temporary copy and is discarded:

- `src/ramanchada2/protocols/calibration/xcalibration.py:87` — in `XCalibrationComponent.process`, the `extrapolate=False` path does `new_spe.x[out_of_bounds] = np.nan`. This has **no effect**: out-of-bounds points are *not* NaN-ed, so `extrapolate=False` behaves like `extrapolate=True` for the RBF model. The neighbouring non-monotonic patch (lines 96–100) does it correctly (build `_newx`, assign, then `new_spe.x = _newx`); the same pattern is needed here.

Recommendation: fix the call site, then grep the codebase and notebooks for `\.x\[` / `\.y\[` assignments. Consider making the returned array `writeable=False` instead of returning a copy (see §3.2), so this class of bug raises instead of passing silently.

### 1.2 Mutated mutable default arguments — P1

`CalibrationModel.derive_model_x` (`calibration_model.py:92`) declares `find_kw={"wlen": 200, "width": 1}` and then executes `find_kw["prominence"] = spe_neon.y_noise_MAD() * self.prominence_coeff` (line 117). Because the default dict is created once at import, **the computed prominence of one call leaks into every subsequent call** that relies on the default — including calls for a different spectrum or laser wavelength. The same anti-pattern appears at:

- `calibration_model.py:100` — `ref_sil={520.45: 1}` (shared default dict)
- `calibration_model.py:191–192` — `find_kw={}`, `fit_peaks_kw={}` in `derive_model_curve`
- `xcalibration.py:447` — `fit_peaks()` does `find_kw.update(dict(sharpening=None))`, mutating the **caller's** dict
- `from_local_file.py:27` — `custom_meta: Dict = {}`

Other modules already use the correct `None` + `if x is None:` idiom, so this is an easy, mechanical cleanup. Adding flake8-bugbear (B006/B008) to the flake8 env would prevent regressions.

### 1.3 Duplicate function definitions in `qmatch.py` — P1

`estimate_median_limit_from_data` is defined at `qmatch.py:54` **and** `qmatch.py:105`; `linear_residual_filter` at `qmatch.py:77` **and** `qmatch.py:130`. The second definitions silently shadow the first (which differ slightly — the first logs via `logger.info`, the second via `print`). This is copy-paste drift; delete one pair. flake8's F811 catches this, which suggests the current branch has not been run through the lint env.

### 1.4 Default parameter value is a typing object — P2

`CalibrationModel._derive_model_curve` (`calibration_model.py:152`) declares `ref=Dict[float, float]` — that assigns the *typing construct* as the default value instead of annotating the parameter. It is masked because line 167 checks `if ref is None`… which is never true for the broken default, so a caller omitting `ref` gets `reference_peaks = Dict[float, float]` passed into `XCalibrationComponent`. Should be `ref: Optional[Dict[float, float]] = None`.

### 1.5 Error paths that cannot fire or misreport — P2

- `spectrum.py:99` — `raise ValueError('Unknown algorithm {algorithm}')`: missing `f` prefix; users see the literal placeholder. Same at `spectrum.py:206` and `spectrum.py:223` (x_err/y_err shape messages).
- `calibration_component.py:83` — `raise Exception("Unsupported conversion {} to {}", spe_unit, newspe_unit)`: `.format()` never applied; the exception carries a tuple. Also, prefer `ValueError`.
- `ycalibration.py:121` — `YCalibrationCertificate.model_construct(**certificate_data)`: `model_construct` **bypasses pydantic validation entirely**, so the `except ValidationError` at line 125 is dead code and malformed certificates load silently. Use `model_validate(certificate_data)`.
- `xcalibration.py:281–284` — in `LazerZeroingComponent.derive_model`, if the fit dataframe has neither a `position` nor a `center` column, `zero_peak_nm` is unbound and the subsequent use raises a confusing `UnboundLocalError`. Add an explicit `else: raise`.
- `from_local_file.py:82–90` — when `backend is None`, `except Exception: spe = load_rc1()` masks the native loader's real failure (a `FileNotFoundError` or a malformed-CSV error resurfaces as an unrelated rc1-parser error). Catch narrowly, or chain (`raise … from err`) / log the native failure before falling back. Also, an invalid `backend` string would leave `spe` unbound (currently prevented only by the `Literal` validation).

### 1.6 Unit conversion for `"pixel"` silently does nothing — P2

`CalibrationComponent.convert_units` (`calibration_component.py:69–81`) returns an unmodified copy when `spe_unit == "pixel"`, with the actual conversion left as a commented-out block. Callers asking for `pixel → nm` get untouched pixel data with no warning. If pixel support is intentionally deferred, raise `NotImplementedError` or log a warning rather than silently passing data through; the downstream matching then operates on incommensurable axes.

### 1.7 `Spectrum` setter side effect on caller data — P3

The `x`/`y` setters (`spectrum.py:136–158`) store the array *by reference* and set `val.flags.writeable = False`. That freezes the **caller's** array too — a user who does `spe.x = my_array; my_array[0] = 5` gets an unexpected `ValueError: assignment destination is read-only` on their own data. Copy on ingest (`self._xdata = np.asarray(val, dtype=float).copy()`) to keep the immutability contract from leaking outward. The setters also don't cross-check x/y length (only `__init__` does), so `spe.y = shorter_array` succeeds and fails later at plot/compute time.

### 1.8 Provenance loss in calibration processing — P3

`LazerZeroingComponent.process` (`xcalibration.py:315`) and `YCalibrationComponent.process` (`ycalibration.py:230–234`) build `Spectrum(x, y, metadata=…)` directly, discarding `applied_processings`. Elsewhere the package works hard to track processing history (that's the point of the `.cha` cache design); calibration — arguably the most important transformation to audit — is the step that loses it.

---

## 2. Security / robustness

### 2.1 `eval` on certificate strings — P2

`YCalibrationCertificate.response_function` (`ycalibration.py:47–58`) `eval()`s both `params` and `equation` strings. For the packaged `config_certs.json` this is trusted input, but the class is public API and the docstring encourages users to construct certificates with arbitrary strings — anyone loading a *shared* certificate file executes arbitrary code. At minimum document the trust boundary; better, parse the polynomial coefficients (`A0…A5`) with a small regex/`ast.literal_eval` and evaluate with `numpy.polynomial` instead of `eval`.

### 2.2 Pickle model persistence — P2

`CalibrationModel.save/from_file` (`calibration_model.py:77–90`) use `pickle`. Besides the standard arbitrary-code-execution caveat when loading shared model files, pickled scipy interpolator subclasses are fragile across scipy/numpy/ramanchada2 versions — a calibration model produced by one environment may not load in another. The interpolators already implement `to_dict`/`from_dict` JSON round-trips; extending that to the whole `CalibrationModel` (components + metadata + version stamp) would give portable, inspectable model files. Note the JSON path has its own bugs today: `CustomCubicSplineInterpolator.to_dict` (`interpolators.py:100–106`) references `self.bc_type`, which scipy's `CubicSpline` does not store (AttributeError), and both it and `CustomRBFInterpolator.to_dict` return raw numpy arrays that `json.dump` cannot serialize.

### 2.3 Silent exception swallowing in HSDS I/O — P3

`io/HSDS.py` contains five bare `except: pass` blocks (lines 26–61, flagged `noqa: E722`). Cache/write failures disappear without a trace, which makes .cha cache bugs very hard to diagnose. Catch the specific h5py exceptions and `logger.warning` them.

**Note (project direction):** the `.cha` format is being retired in favour of **NeXus**. That lowers the priority of fixing the .cha cache internals — instead, plan a deprecation path: mark `write_cha`/`from_chada`/`write_cache` and the `cachefile` mechanism as deprecated (with `DeprecationWarning`), make `write_nexus` the documented persistence route, and update the package landing docstring (`src/ramanchada2/__init__.py`), which currently presents the `.cha` cache as a core concept.

---

## 3. Design & API observations

### 3.1 Dynamic method registration — a deliberate trade-off worth documenting

Filters and creators self-register onto `Spectrum` via `@add_spectrum_filter` / `@add_spectrum_constructor` (`misc/spectrum_deco/`). The mechanism is clean (guards against redefinition, wraps provenance + caching uniformly) and gives a pleasant fluent API. The cost: **no static visibility** — mypy, IDEs and pdoc can't see `spe.dropna()` et al. (`scripts/gen-decorated-docs.py` patches the docs case). Consider generating a `.pyi` stub for `Spectrum` in the same script so type checkers and autocomplete benefit too.

### 3.2 Copy-on-read properties

Every `spe.x` access allocates a full copy. Besides the §1.1 bug class, this is quadratic-ish overhead in loops (`fit_peaks` and friends access `.x`/`.y` repeatedly). Since `_xdata` is already frozen with `writeable=False`, returning the frozen array directly (documented as read-only) would be both faster and *safer* — in-place mutation attempts would raise instead of silently vanishing.

### 3.3 Inconsistent option sets and defaults across calibration layers — P2

The `match_method` / `interpolator_method` `Literal`s disagree between layers:

| Layer | match_method options / default | interpolator options / default |
|---|---|---|
| `XCalibrationComponent.__init__` (`xcalibration.py:37–38`) | incl. `monotonic`; default `qargmin2d` | incl. `pchipinverse`, `pchippolyinverse`, `poly`; default `pchipinverse` |
| `CalibrationModel.derive_model_x` (`calibration_model.py:105–106`) | no `monotonic`; default `cluster` | only `rbf`, `pchip`, `cubic_spline`; default `rbf` |
| `calibration_model_factory` (`calibration_model.py:319–320`) | default `argmin2d` | default `pchip` |
| `get_interpolator` (`interpolators.py:321`) | — | also accepts `rbfinverse`, not exposed in any `Literal` |

So the newest, recommended methods (`rbfinverse`, `pchipinverse`) are unreachable through the high-level API without violating its type hints, and three entry points have three different defaults. Define the option sets once (module-level `Literal` aliases or an `Enum`) and pass through; align defaults deliberately.

### 3.4 Class-level miscellany

- `CalibrationModel` (`calibration_model.py:16–21`): the intended class docstring is placed *after* the `nonmonotonic` attribute, so it is not a docstring at all — `CalibrationModel.__doc__` is `None` and pdoc loses it.
- `calibration_model.py:58–59`: `super(ProcessingModel, self).__init__()` + `super(Plottable, self).__init__()` deliberately *skip* the named classes in the MRO — with `ProcessingModel.__init__` empty this happens to work, but it's fragile and confusing; plain `super().__init__()` (or explicit `ProcessingModel.__init__(self)` / `Plottable.__init__(self)`) says what is meant.
- `CalibrationModel._plot` (`calibration_model.py:303–306`): a `for … break` that only plots the first component — write `if self.components: self.components[0]._plot(ax, **kwargs)` or plot all.
- `interpolators.py:30`: `self.inverse = inverse` — inside the `if inverse:` branch, `inverse` has been rebound to a `PchipInterpolator` instance, so the attribute is sometimes a bool and sometimes an interpolator. Keep the flag and the helper object in separate attributes.
- `get_interpolator` `rbf` branch (`interpolators.py:336`): `neighbors=int(len(x_spe)/3)` is `0` for fewer than 3 anchors → scipy error. Guard small-n (the offset fallback in `derive_model` only covers n == 1).

---

## 4. Code hygiene — P3 but pervasive in the calibration package

- **`print` in library code:** `qmatch.py` has ~40 unconditional `print()` calls (e.g. lines 72–125, 207, 280–351); also `calibration_component.py:190` (`pixels_to_wavenumber`) and `ycalibration.py:126`. The package otherwise uses `logging` consistently — these should be `logger.debug/info`. As-is, every calibration run spams stdout of downstream applications and notebooks.
- **Commented-out code** left in place: `xcalibration.py:51–56` (`from_json`), 168–173 (`_plot_peaks` skeleton), 197–202, 276, 303–318; `calibration_component.py:72–81, 88`; `ycalibration.py:186–200`. Git already remembers; delete.
- **Typos in public names/messages:** `LazerZeroingComponent` (Lazer→Laser; it's public API, so deprecate-alias if renamed), `'x and y shold have same dimentions'` (`spectrum.py:59`).
- **Lint status:** the branch currently has F811 duplicates (§1.3), 4 lines > 119 chars in the calibration package, and trailing whitespace — i.e. `tox -e flake8` (part of CI) will fail. Running the lint env locally before pushing (or a pre-commit hook) would keep the branch releasable.
- **Formatting policy is a whitelist:** `tox.ini` runs black/usort only on an enumerated subset of files. Pragmatic during migration, but the list is already stale relative to the fast-moving files (`xcalibration.py` is usort-checked but not black-checked). Consider flipping to repo-wide with a short exclusion list.

---

## 5. Testing

Genuine strengths: tests cover io (experimental + simulated), filters, peak fitting, protocols, and end-to-end flows against real instrument data; CI runs the matrix on 3.9–3.13 with coverage artifacts and a PR coverage comment; heavy tests are deselectable via `tox -e quick`.

Gaps worth closing:

1. **No unit tests for `qmatch.py` / the `match_peaks` dispatch** (`xcalibration.py:329`) — the highest-churn, highest-risk code. Small synthetic-anchor tests per `match_method` (including the `pixel` special-casing and outlier-filter behaviour) would have caught §1.1 and §1.3.
2. **`extrapolate=False` has no regression test** — the no-op bug in §1.1 is invisible to the current suite.
3. **Serialization round-trips**: `to_dict`/`from_dict` for all interpolators (would catch the `bc_type` AttributeError immediately), plus a `CalibrationModel.save/from_file` round-trip.
4. `tests/protocols/test_calibrationmodel.py` builds all fixtures in a module-scoped `SetupModule` class rather than pytest fixtures — failures in one setup member fail everything with less useful reporting; plain `@pytest.fixture(scope="module")` would integrate better.
5. Committed `tests/__pycache__/*.pyc` files are absent from git (good) but present in the tree; harmless, though a `pytest --cache-clear`/clean occasionally avoids confusion with stale `.pyc` from other Python versions.

---

## 6. Packaging, docs, repo layout

- **Legacy `setup.py`**: works, but PEP 621 `pyproject.toml` is the current standard (the sibling project already migrated). The AST-parse of `__init__.py` for the version is clever but replaceable by `[project] dynamic = ["version"]` + setuptools' `attr:` directive.
- **Dependency pins** (`==1.*`-style ceilings) are a reasonable middle ground for a scientific stack; note `numpy>=1.0,<3.0` is broad enough that numpy-2-incompatible transitive behaviour (e.g. `np.float_` removal in dependencies) is worth a CI job on the oldest supported pins.
- **Docs**: module docstrings and the pdoc pipeline are good. The main README/landing docstring (in `src/ramanchada2/__init__.py`) still says the software is "in early development stage" at version 1.3.1 with a "Production/Stable" classifier — pick one.
- **Repo root clutter**: generated PNG/CSV artifacts (`pearson4*.{png,csv}`, `test_twinning_*.png`, `reference.csv`, `twinned.csv`) are correctly gitignored but written to the repo root by tests. Pointing test outputs at `tmp_path` (or a gitignored `test-output/` dir) keeps the working tree clean and the `.gitignore` from enumerating file-by-file. A 24 MB local `build/` directory likewise suggests builds happen in-tree; `python -m build` with an out-of-tree venv or a periodic clean helps.

---

## 7. Suggested action plan

1. **Now (small, high-value):** fix `xcalibration.py:87` no-op NaN masking; delete duplicate `qmatch.py` functions; convert mutable default args to `None`-idiom; fix `ref=Dict[float, float]`; add the missing f-strings / `.format` calls; `model_construct` → `model_validate`.
2. **Next:** demote all calibration `print`s to `logger`; unify `match_method`/`interpolator_method` literals and defaults; make `tox -e flake8` green and add flake8-bugbear; guard `LazerZeroingComponent.derive_model` column fallback; raise on unimplemented pixel conversion.
3. **Then:** JSON (versioned) persistence for `CalibrationModel`; replace `eval` in `YCalibrationCertificate`; unit tests for matching/interpolator round-trips and `extrapolate=False`; consider `.pyi` stub generation for `Spectrum`; migrate to `pyproject.toml`; start the `.cha` → NeXus deprecation path (see §2.3 note).

---

*Review performed with Claude Code. Findings were verified by reading the code at the stated commit; line numbers refer to that commit.*
