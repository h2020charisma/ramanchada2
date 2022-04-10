Harmonising Raman Spectroscopy
==============================

RamanChada v2
--------------

Clone the repo and go inside
```bash
git clone git@github.com:h2020charisma/ramanchada2.git
cd ramanchada2 #make sure you are in ramanchada2 directory
```

Create a virtual environment
```bash
virtualenv .venv # create virtual environment
source .venv/bin/activate # activate the virtual environment
```

Ramanchada package can be installed in editable mode by runing

```bash
pip install -e . # install ramanchada2
pip install jupyter  # install jupyter
hash -r # make sure the newly created environment is in use
```

In order to create a jupyter kernel, from the already activated virtual environment execute following commands:

```bash
ipython kernel install --name=ramanchada2 --user # set up a new jupyter kernel
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
