from setuptools import setup, find_packages

setup(
    name="ramanchada2",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "matplotlib",
        "h5py",
        "h5pyd",
        "scipy",
        "uncertainties",
        "pydantic",
        "lmfit",
        "pandas",
    ],
)
