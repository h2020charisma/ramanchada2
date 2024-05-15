# ramanchada2

[![CI](https://github.com/h2020charisma/ramanchada2/workflows/CI/badge.svg)](https://github.com/h2020charisma/ramanchada2/actions/workflows/ci.yml)
[![docs](https://github.com/h2020charisma/ramanchada2/workflows/docs/badge.svg)](https://h2020charisma.github.io/ramanchada2/index.html)

ramanchada2 is a Python library for Raman spectroscopy harmonization. It is meant to fill the gap between the theoretical Raman analysis and the experimental Raman spectroscopy by providing means to compare data of different origin.

## Quick start

```sh
pip install ramanchada2
```

- 📖 [Documentation](https://h2020charisma.github.io/ramanchada2/ramanchada2.html)
- ⚗️ [Examples](https://github.com/h2020charisma/ramanchada2/tree/main/examples)

## Quick start with Jupyter notebook examples

[Install Poetry](https://python-poetry.org/docs/#installation).

```sh
git clone https://github.com/h2020charisma/ramanchada2.git
cd ramanchada2
poetry install --with=jupyter
poetry run jupyter notebook examples
```

The browser should open Jupyter automatically.

## Development

This project uses [Poetry](https://python-poetry.org/) for dependency management. Make sure to [have it installed](https://python-poetry.org/docs/#installation). Also, using [pyenv](https://github.com/pyenv/pyenv) on UNIX/MacOS or [pyenv-win](https://github.com/pyenv-win/pyenv-win) for Windows is recommended for Python version management (the default Python version for this project is set in the `.python-version` file).

For better Visual Studio Code integration it may be helpful to set `poetry config virtualenvs.in-project true`.

### Setting up

```sh
git clone https://github.com/h2020charisma/ramanchada2.git
cd ramanchada2
poetry install
```

### Basic usage

```
poetry shell
python
>>> import ramanchada2
```

### Running the linters & tests

Everything:
```
poetry run tox
```

Linter only:
```
poetry run tox -e flake8
```

### Playing with the Jupyter notebooks

```
poetry install --with=jupyter
```
then
```
poetry run jupyter notebook examples
```
or
```
poetry run jupyter lab examples
```
or
```
poetry shell
```
and running `jupyter notebook` or `jupyter lab` from there.

## Acknowledgements

🇪🇺 This project has received funding from the European Union’s Horizon 2020 research and innovation program under [grant agreement No. 952921](https://cordis.europa.eu/project/id/952921).
