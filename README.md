# ramanchada2

[![build](https://github.com/h2020charisma/ramanchada2/workflows/build/badge.svg)](https://github.com/h2020charisma/ramanchada2/actions/workflows/build.yml)
[![docs](https://github.com/h2020charisma/ramanchada2/workflows/docs/badge.svg)](https://h2020charisma.github.io/ramanchada2/index.html)
[![DOI](https://zenodo.org/badge/476228306.svg)](https://zenodo.org/doi/10.5281/zenodo.10255172)

Harmonising Raman spectroscopy: meant to fill the gap between the theoretical Raman analysis and the experimental Raman spectroscopy by providing means to compare data of different origin.

- 📖 [Documentation](https://h2020charisma.github.io/ramanchada2/ramanchada2.html)
- ⚗️ [Examples](https://github.com/h2020charisma/ramanchada2/tree/main/examples)

If you find *ramanchada2* useful, please consider giving it a ⭐ on [GitHub](https://github.com/h2020charisma/ramanchada2)!

## Mini quick start with Conda

**NOTICE**: See the next section for more details and examples with venv-managed virtual environments.

[Install Miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#download) if you don't have Conda already installed, then run the following:
```
git clone https://github.com/h2020charisma/ramanchada2.git
cd ramanchada2
conda env create
conda activate ramanchada2
jupyter notebook
```

## Quick start

Clone the repo using https
```bash
git clone https://github.com/h2020charisma/ramanchada2.git
```
or by using ssh
```
git clone git@github.com:h2020charisma/ramanchada2.git
```


and go inside
```bash
cd ramanchada2  # make sure you are in ramanchada2 directory
```

Make sure you have virtualenv module and create a virtual environment
```bash
virtualenv .venv  # create virtual environment

# activate the virtual environment
source .venv/bin/activate  # on linux
.venv\Scripts\activate  # on windows
```

Install the package in editable mode and its development dependencies by running:

```bash
pip install -e .
pip install autopep8 flake8 jupyter mypy pdoc pytest tox
hash -r  # make sure the newly created environment is in use
```

In order to create a jupyter kernel, from the already activated virtual environment execute following command:

```bash
ipython kernel install --name=ramanchada2 --user  # set up a new jupyter kernel
```

The kernel can be removed by:
```bash
jupyter kernelspec remove ramanchada2
```

A jupyter server can be started from anywhere -- no need to activate the virtual environment:
```bash
jupyter-notebook
```
or
```bash
jupyter-lab
```

A web browser with jupyter should start automaticaly.

## Credits

If you use *ramanchada2* in your research, please cite the following paper:

> Georgiev, G., Coca-Lopez, N., Lellinger, D., Iliev, L., Marinov, E., Tsoneva, S., Kochev, N., Bañares, M. A., Portela, R. and Jeliazkova, N. (2025), Open Source for Raman Spectroscopy Data Harmonization. J Raman Spectrosc. https://doi.org/10.1002/jrs.6789

```bibtex
@article{georgiev2025ramanchada2,
author = {Georgiev, G. and Coca-Lopez, N. and Lellinger, D. and Iliev, L. and Marinov, E. and Tsoneva, S. and Kochev, N. and Bañares, M. A. and Portela, R. and Jeliazkova, N.},
title = {Open Source for Raman Spectroscopy Data Harmonization},
journal = {Journal of Raman Spectroscopy},
keywords = {calibration, data processing, NeXus, Orange data mining, Python},
doi = {https://doi.org/10.1002/jrs.6789},
url = {https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/jrs.6789},
eprint = {https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/pdf/10.1002/jrs.6789}
}
```

## Acknowledgement

🇪🇺 This project has received funding from the European Union’s Horizon 2020 research and innovation program under grant agreement [No. 952921](https://cordis.europa.eu/project/id/952921).
