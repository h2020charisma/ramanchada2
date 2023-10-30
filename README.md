# ramanchada2

[![build](https://github.com/h2020charisma/ramanchada2/workflows/build/badge.svg)](https://github.com/h2020charisma/ramanchada2/actions/workflows/build.yml)
[![docs](https://github.com/h2020charisma/ramanchada2/workflows/docs/badge.svg)](https://h2020charisma.github.io/ramanchada2/index.html)

Harmonising Raman spectroscopy: meant to fill the gap between the theoretical Raman analysis and the experimental Raman spectroscopy by providing means to compare data of different origin.

- 📖 [Documentation](https://h2020charisma.github.io/ramanchada2/ramanchada2.html)
- ⚗️ [Examples](https://github.com/h2020charisma/ramanchada2/tree/main/examples)


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

Ramanchada package and all dependencies can be installed by runing:

```bash
pip install -r requirements-dev.txt  # install development environment
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


## Quick start with Conda

[Install Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and, optionally, Mamba:
```
conda install mamba -n base -c conda-forge
```

Run the following. If you haven't installed Mamba, replace `mamba` with `conda`.
```
mamba env update -f environment.yml
conda activate ramanchada2
jupyter notebook
```
---
🇪🇺 This project has received funding from the European Union’s Horizon 2020 research and innovation program under [grant agreement No. 952921](https://cordis.europa.eu/project/id/952921).
